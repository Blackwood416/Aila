#include "Qwen35HybridTextBackend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
int round_up_seq(int v, int granularity) {
    return ((v + granularity - 1) / granularity) * granularity;
}
}

float Qwen35HybridTextBackend::softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float Qwen35HybridTextBackend::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void Qwen35HybridTextBackend::head_l2_norm_inplace(std::vector<float>& x,
                                                    int seq_len, int num_heads,
                                                    int head_dim, float eps) {
    for (int s = 0; s < seq_len; ++s) {
        float* row = x.data() + static_cast<size_t>(s) * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            float* v = row + static_cast<size_t>(h) * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum_sq += v[d] * v[d];
            float inv = 1.0f / std::sqrt(sum_sq + eps);
            for (int d = 0; d < head_dim; ++d) v[d] *= inv;
        }
    }
}

void Qwen35HybridTextBackend::head_rms_norm_and_silu_gate(std::vector<float>& x,
                                                           const std::vector<float>& norm_weight,
                                                           const std::vector<float>& z,
                                                           int seq_len, int num_heads,
                                                           int head_dim, float eps) {
    for (int s = 0; s < seq_len; ++s) {
        float* row = x.data() + static_cast<size_t>(s) * num_heads * head_dim;
        const float* z_row = z.data() + static_cast<size_t>(s) * num_heads * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            float* v = row + static_cast<size_t>(h) * head_dim;
            const float* zg = z_row + static_cast<size_t>(h) * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; ++d) sum_sq += v[d] * v[d];
            float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + eps);
            for (int d = 0; d < head_dim; ++d) {
                float n = v[d] * inv_rms * norm_weight[d];
                v[d] = n * silu(zg[d]);
            }
        }
    }
}

void Qwen35HybridTextBackend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.hidden = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)hidden_size_});
    buf_.normed = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)hidden_size_});
    buf_.qkv = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_qkv_dim_});
    buf_.linear_all = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_all_dim_});
    buf_.q = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.k = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.v = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.z = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.a = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_kv_heads_});
    buf_.b = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)linear_kv_heads_});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)max_attn_dim_});
    buf_.full_qkv = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)full_fused_qkv_dim_});
    buf_.gate_up = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * ff_dim_)});
    buf_.gate = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});
    buf_.up = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3.5] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen35HybridTextBackend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(
        ctx,
        {(int64_t)max_attn_heads_, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);
    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3.5] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen35HybridTextBackend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;

    int new_seq_cap = std::max(seq_len, incr_prefill_seq_cap_);
    int new_total_cap = std::max(total_len, incr_prefill_total_cap_);
    new_seq_cap = round_up_seq(new_seq_cap, 16);
    new_total_cap = round_up_seq(new_total_cap, 64);

    buf_.incr_scores = Tensor::allocate(
        ctx,
        {(int64_t)max_attn_heads_, (int64_t)new_seq_cap, (int64_t)new_total_cap},
        dnnl::memory::data_type::f32);

    incr_prefill_seq_cap_ = new_seq_cap;
    incr_prefill_total_cap_ = new_total_cap;
    AILA_LOG_INFO("[Qwen3.5] Incremental prefill score buffer resized: seq_cap=%d total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

Qwen35HybridTextBackend::~Qwen35HybridTextBackend() {
    clear_mrope_positions();
}

bool Qwen35HybridTextBackend::load(Context& ctx,
                                   ModelWeights& weights,
                                   const ModelSpec& spec,
                                   int max_seq_len,
                                   std::string* error_message) {
    if (spec.family != ModelFamily::Qwen35Hybrid) {
        if (error_message) *error_message = "Qwen35HybridTextBackend: invalid model family";
        return false;
    }

    spec_ = spec;
    cfg_ = spec.qwen35_text;

    hidden_size_ = cfg_.hidden_size;
    ff_dim_ = cfg_.intermediate_size;
    max_seq_len_ = max_seq_len;
    current_len_ = 0;

    full_q_heads_ = cfg_.num_attention_heads;
    full_kv_heads_ = cfg_.num_key_value_heads;
    full_head_dim_ = cfg_.head_dim;
    full_q_dim_ = full_q_heads_ * full_head_dim_;
    full_kv_dim_ = full_kv_heads_ * full_head_dim_;
    full_q_proj_dim_ = cfg_.attn_output_gate ? (2 * full_q_dim_) : full_q_dim_;
    full_fused_qkv_dim_ = full_q_proj_dim_ + 2 * full_kv_dim_;

    linear_q_heads_ = cfg_.linear_num_key_heads;
    linear_kv_heads_ = cfg_.linear_num_value_heads;
    linear_head_dim_ = cfg_.linear_key_head_dim;
    linear_q_dim_ = linear_q_heads_ * linear_head_dim_;
    linear_kv_dim_ = linear_kv_heads_ * cfg_.linear_value_head_dim;
    linear_qkv_dim_ = linear_q_dim_ + linear_q_dim_ + linear_kv_dim_;
    linear_z_dim_ = linear_kv_dim_;
    linear_all_dim_ = linear_qkv_dim_ + linear_z_dim_ + 2 * linear_kv_heads_;
    linear_conv_kernel_dim_ = std::max(1, cfg_.linear_conv_kernel_dim);
    linear_conv_channels_ = linear_qkv_dim_;
    use_delta_linear_ = aila::env::read_flag("AILA_Q35_LINEAR_DELTA", true);

    max_attn_heads_ = std::max(full_q_heads_, linear_q_heads_);
    max_qkv_dim_ = std::max(full_q_proj_dim_, linear_qkv_dim_);
    max_attn_dim_ = std::max(full_q_dim_, linear_kv_dim_);

    auto transpose_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);
        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };
    auto plus_one_norm_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        if (src.dtype() != dnnl::memory::data_type::bf16) {
            return &src;
        }
        size_t n = (size_t)src.numel();
        std::vector<bf16> host(n);
        ctx.memcpy_d2h(host.data(), src.data(), n * sizeof(bf16));
        for (size_t i = 0; i < n; ++i) {
            host[i] = bf16(static_cast<float>(host[i]) + 1.0f);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), src.dtype());
        ctx.memcpy_h2d(dst.data(), host.data(), n * sizeof(bf16));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    embed_weight_ = &weights.get("model.language_model.embed_tokens.weight");

    layers_.clear();
    layer_caches_.clear();
    fused_weights_.clear();
    layers_.resize(cfg_.num_hidden_layers);
    layer_caches_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 3 + 1);

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

    auto fuse_four_cols = [&](Tensor& a, Tensor& b, Tensor& c, Tensor& d) {
        int64_t rows = a.shape(0);
        int64_t a_cols = a.shape(1);
        int64_t b_cols = b.shape(1);
        int64_t c_cols = c.shape(1);
        int64_t d_cols = d.shape(1);
        int64_t total_cols = a_cols + b_cols + c_cols + d_cols;

        Tensor out = Tensor::allocate(ctx, {rows, total_cols}, a.dtype());
        bf16* a_ptr = static_cast<bf16*>(a.data());
        bf16* b_ptr = static_cast<bf16*>(b.data());
        bf16* c_ptr = static_cast<bf16*>(c.data());
        bf16* d_ptr = static_cast<bf16*>(d.data());
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
                } else if (cidx < a_cols + b_cols + c_cols) {
                    int cc = cidx - static_cast<int>(a_cols + b_cols);
                    o_ptr[out_idx] = c_ptr[static_cast<int64_t>(r) * c_cols + cc];
                } else {
                    int dc = cidx - static_cast<int>(a_cols + b_cols + c_cols);
                    o_ptr[out_idx] = d_ptr[static_cast<int64_t>(r) * d_cols + dc];
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

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        auto& cache = layer_caches_[i];
        std::string prefix = "model.language_model.layers." + std::to_string(i) + ".";

        // Qwen3.5 RMSNorm uses (1 + weight) instead of direct weight multiply.
        layer.input_ln_weight = plus_one_norm_weight(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = plus_one_norm_weight(prefix + "post_attention_layernorm.weight");

        bool is_linear = true;
        if (i < static_cast<int>(cfg_.layer_types.size())) {
            is_linear = (cfg_.layer_types[(size_t)i] == "linear_attention");
        } else {
            is_linear = ((i + 1) % 4 != 0);
        }
        layer.is_linear = is_linear;

        if (is_linear) {
            Tensor* qkv_w = transpose_weight(prefix + "linear_attn.in_proj_qkv.weight");
            Tensor* z_w = transpose_weight(prefix + "linear_attn.in_proj_z.weight");
            Tensor* o_w = transpose_weight(prefix + "linear_attn.out_proj.weight");
            Tensor* a_w = transpose_weight(prefix + "linear_attn.in_proj_a.weight");
            Tensor* b_w = transpose_weight(prefix + "linear_attn.in_proj_b.weight");
            layer.linear_qkv_proj.init(ctx, *qkv_w, hidden_size_, linear_qkv_dim_, true);
            layer.linear_z_proj.init(ctx, *z_w, hidden_size_, linear_z_dim_, true);
            layer.linear_o_proj.init(ctx, *o_w, linear_kv_dim_, hidden_size_, true);
            layer.linear_a_proj.init(ctx, *a_w, hidden_size_, linear_kv_heads_, true);
            layer.linear_b_proj.init(ctx, *b_w, hidden_size_, linear_kv_heads_, true);
            fused_weights_.push_back(fuse_four_cols(*qkv_w, *z_w, *a_w, *b_w));
            layer.linear_all_proj.init(ctx, fused_weights_.back(), hidden_size_, linear_all_dim_, true);

            layer.linear_norm_weight = &weights.get(prefix + "linear_attn.norm.weight");
            layer.linear_A_log = &weights.get(prefix + "linear_attn.A_log");
            layer.linear_dt_bias = &weights.get(prefix + "linear_attn.dt_bias");
            layer.linear_conv1d_weight = &weights.get(prefix + "linear_attn.conv1d.weight");

            layer.host_linear_A_negexp.resize((size_t)linear_kv_heads_);
            layer.host_linear_dt_bias.resize((size_t)linear_kv_heads_);
            layer.host_linear_norm.resize((size_t)cfg_.linear_value_head_dim);
            layer.host_linear_conv.resize((size_t)linear_conv_channels_ * linear_conv_kernel_dim_);

            std::vector<float> host_a((size_t)linear_kv_heads_);
            ctx.memcpy_d2h(host_a.data(), layer.linear_A_log->data(),
                           host_a.size() * sizeof(float));
            for (int h = 0; h < linear_kv_heads_; ++h) {
                layer.host_linear_A_negexp[(size_t)h] = -std::exp(host_a[(size_t)h]);
            }

            {
                std::vector<bf16> host_dt((size_t)linear_kv_heads_);
                ctx.memcpy_d2h(host_dt.data(), layer.linear_dt_bias->data(),
                               host_dt.size() * sizeof(bf16));
                for (int h = 0; h < linear_kv_heads_; ++h) {
                    layer.host_linear_dt_bias[(size_t)h] = static_cast<float>(host_dt[(size_t)h]);
                }
            }

            {
                std::vector<float> host_norm((size_t)cfg_.linear_value_head_dim);
                ctx.memcpy_d2h(host_norm.data(), layer.linear_norm_weight->data(),
                               host_norm.size() * sizeof(float));
                layer.host_linear_norm = std::move(host_norm);
            }

            {
                size_t n_conv = (size_t)linear_conv_channels_ * (size_t)linear_conv_kernel_dim_;
                std::vector<bf16> conv_raw(n_conv);
                ctx.memcpy_d2h(conv_raw.data(), layer.linear_conv1d_weight->data(),
                               n_conv * sizeof(bf16));
                for (size_t idx = 0; idx < n_conv; ++idx) {
                    layer.host_linear_conv[idx] = static_cast<float>(conv_raw[idx]);
                }
            }

            if (use_delta_linear_) {
                cache.k = Tensor();
                cache.v = Tensor();
                cache.linear_state = Tensor::allocate(
                    ctx,
                    {(int64_t)linear_kv_heads_, (int64_t)linear_head_dim_, (int64_t)cfg_.linear_value_head_dim},
                    dnnl::memory::data_type::f32);
                if (linear_conv_kernel_dim_ > 1) {
                    cache.linear_conv_state = Tensor::allocate(
                        ctx,
                        {(int64_t)(linear_conv_kernel_dim_ - 1), (int64_t)linear_conv_channels_},
                        dnnl::memory::data_type::f32);
                } else {
                    cache.linear_conv_state = Tensor();
                }
                if (cache.linear_state.numel() > 0) {
                    ctx.queue().memset(cache.linear_state.data(), 0, cache.linear_state.size_bytes());
                }
                if (cache.linear_conv_state.numel() > 0) {
                    ctx.queue().memset(cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
                }
                cache.host_linear_state.assign(
                    (size_t)linear_kv_heads_ * (size_t)linear_head_dim_ * (size_t)cfg_.linear_value_head_dim,
                    0.0f);
                cache.host_linear_conv_state.assign(
                    (size_t)std::max(0, linear_conv_kernel_dim_ - 1) * (size_t)linear_conv_channels_,
                    0.0f);
            } else {
                cache.k = Tensor::allocate(ctx,
                                           {(int64_t)linear_q_heads_, (int64_t)max_seq_len_, (int64_t)linear_head_dim_},
                                           dnnl::memory::data_type::bf16);
                cache.v = Tensor::allocate(ctx,
                                           {(int64_t)linear_kv_heads_, (int64_t)max_seq_len_, (int64_t)cfg_.linear_value_head_dim},
                                           dnnl::memory::data_type::bf16);
                cache.linear_state = Tensor();
                cache.linear_conv_state = Tensor();
                cache.host_linear_state.clear();
                cache.host_linear_conv_state.clear();
            }
        } else {
            Tensor* q_w = transpose_weight(prefix + "self_attn.q_proj.weight");
            Tensor* k_w = transpose_weight(prefix + "self_attn.k_proj.weight");
            Tensor* v_w = transpose_weight(prefix + "self_attn.v_proj.weight");
            Tensor* o_w = transpose_weight(prefix + "self_attn.o_proj.weight");
            layer.q_proj.init(ctx, *q_w, hidden_size_, full_q_proj_dim_, true);
            layer.k_proj.init(ctx, *k_w, hidden_size_, full_kv_dim_, true);
            layer.v_proj.init(ctx, *v_w, hidden_size_, full_kv_dim_, true);
            fused_weights_.push_back(fuse_three_cols(*q_w, *k_w, *v_w));
            layer.qkv_proj.init(ctx, fused_weights_.back(), hidden_size_, full_fused_qkv_dim_, true);
            layer.o_proj.init(ctx, *o_w, full_q_dim_, hidden_size_, true);
            layer.q_norm_weight = plus_one_norm_weight(prefix + "self_attn.q_norm.weight");
            layer.k_norm_weight = plus_one_norm_weight(prefix + "self_attn.k_norm.weight");

            cache.k = Tensor::allocate(ctx,
                                       {(int64_t)full_kv_heads_, (int64_t)max_seq_len_, (int64_t)full_head_dim_},
                                       dnnl::memory::data_type::bf16);
            cache.v = Tensor::allocate(ctx,
                                       {(int64_t)full_kv_heads_, (int64_t)max_seq_len_, (int64_t)full_head_dim_},
                                       dnnl::memory::data_type::bf16);
            cache.linear_state = Tensor();
            cache.linear_conv_state = Tensor();
            cache.host_linear_state.clear();
            cache.host_linear_conv_state.clear();
        }

        Tensor* gate_w = transpose_weight(prefix + "mlp.gate_proj.weight");
        Tensor* up_w = transpose_weight(prefix + "mlp.up_proj.weight");
        Tensor* down_w = transpose_weight(prefix + "mlp.down_proj.weight");
        layer.gate_proj.init(ctx, *gate_w, hidden_size_, ff_dim_, true);
        layer.up_proj.init(ctx, *up_w, hidden_size_, ff_dim_, true);
        fused_weights_.push_back(fuse_two_cols(*gate_w, *up_w));
        layer.gate_up_proj.init(ctx, fused_weights_.back(), hidden_size_, 2 * ff_dim_, true);
        layer.down_proj.init(ctx, *down_w, ff_dim_, hidden_size_, true);
    }

    final_norm_weight_ = plus_one_norm_weight("model.language_model.norm.weight");

    if (weights.has("lm_head.weight")) {
        Tensor* lm_w = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3.5] lm_head loaded (standalone)");
    } else {
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.put("model.language_model.lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("model.language_model.lm_head.weight_preprocessed"),
                      hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3.5] lm_head (tied, preprocessed copy)");
    }

    ctx.synchronize();

    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    incr_prefill_seq_cap_ = 0;
    incr_prefill_total_cap_ = 0;
    clear_embedding_overrides();
    ensure_runtime_buffers(ctx, 1);

    buf_.logits = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
    buf_.decode_scores = Tensor::allocate(
        ctx, {(int64_t)max_attn_heads_, (int64_t)max_seq_len_},
        dnnl::memory::data_type::f32);

    AILA_LOG_INFO("[Qwen3.5] Hybrid text backend loaded: layers=%d hidden=%d vocab=%d",
                  cfg_.num_hidden_layers, hidden_size_, cfg_.vocab_size);
    AILA_LOG_INFO("[Qwen3.5] Linear mode: %s (set AILA_Q35_LINEAR_DELTA=0 to force legacy-attn fallback)",
                  use_delta_linear_ ? "delta-host" : "legacy-attn");
    return true;
}

void Qwen35HybridTextBackend::set_embedding_overrides(
    const std::vector<int>& positions,
    const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
    int hidden_size) {
    if (positions.empty() || embeddings.empty()) {
        clear_embedding_overrides();
        return;
    }
    if (hidden_size <= 0) {
        clear_embedding_overrides();
        return;
    }
    if (embeddings.size() != static_cast<size_t>(positions.size()) * static_cast<size_t>(hidden_size)) {
        clear_embedding_overrides();
        return;
    }

    embed_override_positions_ = positions;
    embed_override_values_ = embeddings;
    embed_override_hidden_size_ = hidden_size;
}

void Qwen35HybridTextBackend::clear_embedding_overrides() {
    embed_override_positions_.clear();
    embed_override_values_.clear();
    embed_override_hidden_size_ = 0;
}

void Qwen35HybridTextBackend::set_mrope_positions(Context& ctx,
                                                  const std::vector<int>& pos_t,
                                                  const std::vector<int>& pos_h,
                                                  const std::vector<int>& pos_w,
                                                  int text_pos_delta) {
    clear_mrope_positions();
    if (pos_t.empty() || pos_h.empty() || pos_w.empty()) {
        return;
    }
    if (pos_t.size() != pos_h.size() || pos_t.size() != pos_w.size()) {
        return;
    }

    const size_t bytes = pos_t.size() * sizeof(int);
    mrope_ctx_ = &ctx;
    mrope_prompt_len_ = static_cast<int>(pos_t.size());
    mrope_text_pos_delta_ = text_pos_delta;

    mrope_pos_t_ = static_cast<int*>(ctx.alloc_device(bytes));
    mrope_pos_h_ = static_cast<int*>(ctx.alloc_device(bytes));
    mrope_pos_w_ = static_cast<int*>(ctx.alloc_device(bytes));
    ctx.memcpy_h2d(mrope_pos_t_, pos_t.data(), bytes);
    ctx.memcpy_h2d(mrope_pos_h_, pos_h.data(), bytes);
    ctx.memcpy_h2d(mrope_pos_w_, pos_w.data(), bytes);
}

void Qwen35HybridTextBackend::clear_mrope_positions() {
    if (mrope_ctx_) {
        if (mrope_pos_t_) mrope_ctx_->free_device(mrope_pos_t_);
        if (mrope_pos_h_) mrope_ctx_->free_device(mrope_pos_h_);
        if (mrope_pos_w_) mrope_ctx_->free_device(mrope_pos_w_);
    }
    mrope_pos_t_ = nullptr;
    mrope_pos_h_ = nullptr;
    mrope_pos_w_ = nullptr;
    mrope_prompt_len_ = 0;
    mrope_text_pos_delta_ = 0;
    mrope_ctx_ = nullptr;
}

void Qwen35HybridTextBackend::run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                                                    Tensor& qkv_src, Tensor& z_src,
                                                    Tensor& a_src, Tensor& b_src,
                                                    int seq_len) {
    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int num_heads = linear_kv_heads_;
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;

    auto& scratch = linear_delta_scratch_;
    scratch.h_qkv.resize((size_t)seq_len * qkv_dim);
    scratch.h_z.resize((size_t)seq_len * z_dim);
    scratch.h_a.resize((size_t)seq_len * num_heads);
    scratch.h_b.resize((size_t)seq_len * num_heads);
    ctx.memcpy_d2h(scratch.h_qkv.data(), qkv_src.data(), scratch.h_qkv.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_z.data(), z_src.data(), scratch.h_z.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_a.data(), a_src.data(), scratch.h_a.size() * sizeof(bf16));
    ctx.memcpy_d2h(scratch.h_b.data(), b_src.data(), scratch.h_b.size() * sizeof(bf16));

    std::vector<float>& h_conv_state = cache.host_linear_conv_state;
    if (h_conv_state.size() != (size_t)std::max(0, kernel - 1) * (size_t)linear_conv_channels_) {
        h_conv_state.assign((size_t)std::max(0, kernel - 1) * (size_t)linear_conv_channels_, 0.0f);
    }

    std::vector<float>& state = cache.host_linear_state;
    if (state.size() != (size_t)num_heads * (size_t)head_k_dim * (size_t)head_v_dim) {
        state.assign((size_t)num_heads * (size_t)head_k_dim * (size_t)head_v_dim, 0.0f);
    }

    scratch.conv_in.resize((size_t)seq_len * qkv_dim);
    scratch.z.resize((size_t)seq_len * z_dim);
    scratch.a.resize((size_t)seq_len * num_heads);
    scratch.b.resize((size_t)seq_len * num_heads);
    for (size_t i = 0; i < scratch.conv_in.size(); ++i) scratch.conv_in[i] = static_cast<float>(scratch.h_qkv[i]);
    for (size_t i = 0; i < scratch.z.size(); ++i) scratch.z[i] = static_cast<float>(scratch.h_z[i]);
    for (size_t i = 0; i < scratch.a.size(); ++i) scratch.a[i] = static_cast<float>(scratch.h_a[i]);
    for (size_t i = 0; i < scratch.b.size(); ++i) scratch.b[i] = static_cast<float>(scratch.h_b[i]);

    scratch.conv_out.assign((size_t)seq_len * qkv_dim, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = scratch.conv_in.data() + (size_t)t * qkv_dim;
        float* y_t = scratch.conv_out.data() + (size_t)t * qkv_dim;
        for (int c = 0; c < qkv_dim; ++c) {
            const float* w = layer.host_linear_conv.data() + (size_t)c * kernel;
            float v = 0.0f;
            for (int j = 0; j < kernel - 1; ++j) {
                if (kernel > 1) {
                    v += h_conv_state[(size_t)j * qkv_dim + c] * w[j];
                }
            }
            v += x_t[c] * w[kernel - 1];
            y_t[c] = silu(v);
        }
        if (kernel > 1) {
            if (kernel - 2 > 0) {
                std::memmove(h_conv_state.data(),
                             h_conv_state.data() + qkv_dim,
                             (size_t)(kernel - 2) * qkv_dim * sizeof(float));
            }
            std::memcpy(h_conv_state.data() + (size_t)(kernel - 2) * qkv_dim,
                        x_t, (size_t)qkv_dim * sizeof(float));
        }
    }

    scratch.q.resize((size_t)seq_len * linear_q_dim_);
    scratch.k.resize((size_t)seq_len * linear_q_dim_);
    scratch.v.resize((size_t)seq_len * linear_kv_dim_);
    for (int t = 0; t < seq_len; ++t) {
        const float* src = scratch.conv_out.data() + (size_t)t * qkv_dim;
        std::memcpy(scratch.q.data() + (size_t)t * linear_q_dim_, src, (size_t)linear_q_dim_ * sizeof(float));
        std::memcpy(scratch.k.data() + (size_t)t * linear_q_dim_, src + linear_q_dim_, (size_t)linear_q_dim_ * sizeof(float));
        std::memcpy(scratch.v.data() + (size_t)t * linear_kv_dim_, src + linear_q_dim_ + linear_q_dim_,
                    (size_t)linear_kv_dim_ * sizeof(float));
    }

    head_l2_norm_inplace(scratch.q, seq_len, linear_q_heads_, head_k_dim, cfg_.rms_norm_eps);
    head_l2_norm_inplace(scratch.k, seq_len, linear_q_heads_, head_k_dim, cfg_.rms_norm_eps);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    scratch.out.assign((size_t)seq_len * linear_kv_dim_, 0.0f);
    scratch.sk.resize((size_t)head_v_dim);
    scratch.d.resize((size_t)head_v_dim);

    for (int t = 0; t < seq_len; ++t) {
        for (int hv = 0; hv < num_heads; ++hv) {
            int hk = hv % std::max(1, linear_q_heads_);
            const float* q_h = scratch.q.data() + (size_t)t * linear_q_dim_ + (size_t)hk * head_k_dim;
            const float* k_h = scratch.k.data() + (size_t)t * linear_q_dim_ + (size_t)hk * head_k_dim;
            const float* v_h = scratch.v.data() + (size_t)t * linear_kv_dim_ + (size_t)hv * head_v_dim;

            float beta = 1.0f / (1.0f + std::exp(-scratch.b[(size_t)t * num_heads + hv]));
            float alpha = scratch.a[(size_t)t * num_heads + hv];
            float g_pre = softplus(alpha + layer.host_linear_dt_bias[(size_t)hv]) *
                          layer.host_linear_A_negexp[(size_t)hv];
            float decay = std::exp(g_pre);

            // S: [key_dim, value_dim] per head (row-major)
            float* S = state.data() + ((size_t)hv * head_k_dim * head_v_dim);
            for (int idx = 0; idx < head_k_dim * head_v_dim; ++idx) S[idx] *= decay;

            for (int dv = 0; dv < head_v_dim; ++dv) {
                float acc = 0.0f;
                for (int j = 0; j < head_k_dim; ++j) {
                    acc += S[(size_t)j * head_v_dim + dv] * k_h[j];
                }
                scratch.sk[(size_t)dv] = acc;
                scratch.d[(size_t)dv] = (v_h[dv] - acc) * beta;
            }

            for (int j = 0; j < head_k_dim; ++j) {
                float kj = k_h[j];
                float* row = S + (size_t)j * head_v_dim;
                for (int dv = 0; dv < head_v_dim; ++dv) {
                    row[dv] += kj * scratch.d[(size_t)dv];
                }
            }

            float* o_h = scratch.out.data() + (size_t)t * linear_kv_dim_ + (size_t)hv * head_v_dim;
            for (int dv = 0; dv < head_v_dim; ++dv) {
                float acc = 0.0f;
                for (int j = 0; j < head_k_dim; ++j) {
                    acc += S[(size_t)j * head_v_dim + dv] * q_h[j];
                }
                o_h[dv] = acc * scale;
            }
        }
    }

    head_rms_norm_and_silu_gate(scratch.out, layer.host_linear_norm, scratch.z, seq_len, num_heads,
                                head_v_dim, cfg_.rms_norm_eps);

    scratch.out_bf16.resize(scratch.out.size());
    for (size_t i = 0; i < scratch.out.size(); ++i) scratch.out_bf16[i] = bf16(scratch.out[i]);
    ctx.memcpy_h2d(buf_.attn_out.data(), scratch.out_bf16.data(), scratch.out_bf16.size() * sizeof(bf16));
}

Tensor& Qwen35HybridTextBackend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("Qwen35HybridTextBackend::forward: seq_len must be positive");
    }
    int start_pos = current_len_;
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen35HybridTextBackend::forward: context window exceeded");
    }

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, cached_len);
        }
    }

    ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, hidden_size_);
    if (!embed_override_positions_.empty() && embed_override_hidden_size_ == hidden_size_) {
        bf16* hidden_ptr = static_cast<bf16*>(buf_.hidden.data());
        for (size_t i = 0; i < embed_override_positions_.size(); ++i) {
            int pos = embed_override_positions_[i];
            if (pos < start_pos || pos >= start_pos + seq_len) continue;
            int local_row = pos - start_pos;
            const bf16* src = embed_override_values_.data() + i * static_cast<size_t>(hidden_size_);
            bf16* dst = hidden_ptr + static_cast<size_t>(local_row) * static_cast<size_t>(hidden_size_);
            ctx.queue().memcpy(dst, src, static_cast<size_t>(hidden_size_) * sizeof(bf16));
        }
    }
    ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight, cfg_.rms_norm_eps,
                  buf_.normed, seq_len, hidden_size_);

    int full_rotary_dim = std::max(2, (int)std::floor(full_head_dim_ * cfg_.rope.partial_rotary_factor));
    full_rotary_dim = std::min(full_head_dim_, full_rotary_dim);
    if (full_rotary_dim & 1) --full_rotary_dim;
    if (full_rotary_dim <= 0) full_rotary_dim = std::min(2, full_head_dim_);
    bool debug_layer_stats = aila::env::read_flag("AILA_DEBUG_Q35_LAYER_STATS", false);
    int debug_layer_detail_idx = aila::env::read_int_raw("AILA_DEBUG_Q35_LAYER_DETAIL", -1);
    auto log_row_stats = [&](const char* tag, Tensor& t, int row, int width) {
        if (row < 0) return;
        bf16* ptr = static_cast<bf16*>(t.data()) + (size_t)row * width;
        std::vector<bf16> h_row((size_t)width);
        ctx.memcpy_d2h(h_row.data(), ptr, h_row.size() * sizeof(bf16));
        double mean = 0.0;
        double abs_mean = 0.0;
        float max_abs = 0.0f;
        for (int d = 0; d < width; ++d) {
            float v = static_cast<float>(h_row[(size_t)d]);
            mean += v;
            abs_mean += std::abs(v);
            max_abs = std::max(max_abs, std::abs(v));
        }
        mean /= (double)width;
        abs_mean /= (double)width;
        float v0 = static_cast<float>(h_row[0]);
        float v1 = static_cast<float>(h_row[1]);
        float v2 = static_cast<float>(h_row[2]);
        float v3 = static_cast<float>(h_row[3]);
        AILA_LOG_INFO("[Q35L0] %s mean=%.6f abs_mean=%.6f max_abs=%.6f v0=%.6f v1=%.6f v2=%.6f v3=%.6f",
                      tag, mean, abs_mean, max_abs, v0, v1, v2, v3);
    };

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        auto& cache = layer_caches_[i];
        int dbg_row = seq_len - 1;
        bool debug_this_layer = (debug_layer_detail_idx >= 0 && debug_layer_detail_idx == i);
        if (debug_this_layer && seq_len > 0) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "layer%d_input_ln_out", i);
            log_row_stats(tag, buf_.normed, dbg_row, hidden_size_);
        }

        if (layer.is_linear) {
            if (use_delta_linear_) {
                if (seq_len == 1) {
                    layer.linear_all_proj.forward(ctx, buf_.normed, buf_.linear_all, seq_len);
                    bf16* fused_ptr = static_cast<bf16*>(buf_.linear_all.data());
                    Tensor qkv_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)linear_qkv_dim_});
                    Tensor z_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_, {1, (int64_t)linear_z_dim_});
                    Tensor a_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_ + linear_z_dim_,
                                                 {1, (int64_t)linear_kv_heads_});
                    Tensor b_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_ + linear_z_dim_ + linear_kv_heads_,
                                                 {1, (int64_t)linear_kv_heads_});
                    run_linear_delta_host(ctx, layer, cache, qkv_view, z_view, a_view, b_view, seq_len);
                } else {
                    layer.linear_qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
                    layer.linear_z_proj.forward(ctx, buf_.normed, buf_.z, seq_len);
                    layer.linear_a_proj.forward(ctx, buf_.normed, buf_.a, seq_len);
                    layer.linear_b_proj.forward(ctx, buf_.normed, buf_.b, seq_len);
                    run_linear_delta_host(ctx, layer, cache, buf_.qkv, buf_.z, buf_.a, buf_.b, seq_len);
                }
            } else {
                layer.linear_qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
                ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v,
                               seq_len, linear_q_dim_, linear_q_dim_);

                {
                    std::vector<bf16> hq((size_t)seq_len * linear_q_dim_);
                    std::vector<bf16> hk((size_t)seq_len * linear_q_dim_);
                    std::vector<float> fq((size_t)seq_len * linear_q_dim_);
                    std::vector<float> fk((size_t)seq_len * linear_q_dim_);
                    ctx.memcpy_d2h(hq.data(), buf_.q.data(), hq.size() * sizeof(bf16));
                    ctx.memcpy_d2h(hk.data(), buf_.k.data(), hk.size() * sizeof(bf16));
                    for (size_t z = 0; z < fq.size(); ++z) {
                        fq[z] = static_cast<float>(hq[z]);
                        fk[z] = static_cast<float>(hk[z]);
                    }
                    head_l2_norm_inplace(fq, seq_len, linear_q_heads_, linear_head_dim_, cfg_.rms_norm_eps);
                    head_l2_norm_inplace(fk, seq_len, linear_q_heads_, linear_head_dim_, cfg_.rms_norm_eps);
                    for (size_t z = 0; z < fq.size(); ++z) {
                        hq[z] = bf16(fq[z]);
                        hk[z] = bf16(fk[z]);
                    }
                    ctx.memcpy_h2d(buf_.q.data(), hq.data(), hq.size() * sizeof(bf16));
                    ctx.memcpy_h2d(buf_.k.data(), hk.data(), hk.size() * sizeof(bf16));
                }

                ops::copy_to_cache(ctx, buf_.k, cache.k, seq_len, start_pos,
                                   linear_q_heads_, linear_head_dim_, max_seq_len_);
                ops::copy_to_cache(ctx, buf_.v, cache.v, seq_len, start_pos,
                                   linear_kv_heads_, cfg_.linear_value_head_dim, max_seq_len_);

                if (seq_len == 1) {
                    ops::attention_decode(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.decode_scores,
                                          linear_q_heads_, linear_kv_heads_, linear_head_dim_, cached_len);
                } else if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores,
                                           seq_len, linear_q_heads_, linear_kv_heads_, linear_head_dim_);
                } else {
                    ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores,
                                                  seq_len, start_pos, linear_q_heads_, linear_kv_heads_,
                                                  linear_head_dim_, max_seq_len_);
                }

                layer.linear_z_proj.forward(ctx, buf_.normed, buf_.z, seq_len);
                ops::sigmoid_mul(ctx, buf_.attn_out, buf_.z, buf_.attn_out, seq_len * linear_kv_dim_);
            }
            layer.linear_o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);
        } else {
            Tensor k_decode_view;
            Tensor v_decode_view;
            Tensor* k_for_attn = &buf_.k;
            Tensor* v_for_attn = &buf_.v;

            if (seq_len == 1) {
                layer.qkv_proj.forward(ctx, buf_.normed, buf_.full_qkv, seq_len);
                bf16* fused_ptr = static_cast<bf16*>(buf_.full_qkv.data());
                Tensor q_proj_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)full_q_proj_dim_});
                k_decode_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_, {1, (int64_t)full_kv_dim_});
                v_decode_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_ + full_kv_dim_, {1, (int64_t)full_kv_dim_});
                k_for_attn = &k_decode_view;
                v_for_attn = &v_decode_view;
                if (cfg_.attn_output_gate) {
                    ops::split_q_gate(ctx, q_proj_view, buf_.q, buf_.z, seq_len, full_q_heads_, full_head_dim_);
                } else {
                    ops::copy_tensor(ctx, q_proj_view, buf_.q, full_q_dim_);
                }
            } else {
                layer.q_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
                ops::split_q_gate(ctx, buf_.qkv, buf_.q, buf_.z, seq_len, full_q_heads_, full_head_dim_);
                layer.k_proj.forward(ctx, buf_.normed, buf_.k, seq_len);
                layer.v_proj.forward(ctx, buf_.normed, buf_.v, seq_len);
            }
            if (debug_this_layer && seq_len > 0) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_gate_z", i);
                log_row_stats(tag, buf_.z, dbg_row, full_q_dim_);
            }

            ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                               cfg_.rms_norm_eps, seq_len, full_q_heads_, full_head_dim_);
            ops::head_rms_norm(ctx, *k_for_attn, *layer.k_norm_weight,
                               cfg_.rms_norm_eps, seq_len, full_kv_heads_, full_head_dim_);

            ops::apply_rope_partial(ctx, buf_.q, *k_for_attn, seq_len, start_pos,
                                    full_q_heads_, full_kv_heads_, full_head_dim_,
                                    full_rotary_dim, cfg_.rope.rope_theta,
                                    cfg_.rope.mrope_interleaved,
                                    mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
                                    mrope_prompt_len_, mrope_text_pos_delta_,
                                    cfg_.rope.mrope_section[0],
                                    cfg_.rope.mrope_section[1],
                                    cfg_.rope.mrope_section[2]);
            if (debug_this_layer && seq_len > 0) {
                char tag_q[64];
                char tag_k[64];
                char tag_v[64];
                std::snprintf(tag_q, sizeof(tag_q), "layer%d_full_q_after_rope", i);
                std::snprintf(tag_k, sizeof(tag_k), "layer%d_full_k_after_rope", i);
                std::snprintf(tag_v, sizeof(tag_v), "layer%d_full_v", i);
                log_row_stats(tag_q, buf_.q, dbg_row, full_q_dim_);
                log_row_stats(tag_k, *k_for_attn, dbg_row, full_kv_dim_);
                log_row_stats(tag_v, *v_for_attn, dbg_row, full_kv_dim_);
            }

            ops::copy_to_cache(ctx, *k_for_attn, cache.k, seq_len, start_pos,
                               full_kv_heads_, full_head_dim_, max_seq_len_);
            ops::copy_to_cache(ctx, *v_for_attn, cache.v, seq_len, start_pos,
                               full_kv_heads_, full_head_dim_, max_seq_len_);

            if (seq_len == 1) {
                ops::attention_decode(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.decode_scores,
                                      full_q_heads_, full_kv_heads_, full_head_dim_, cached_len);
            } else if (start_pos == 0) {
                ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores,
                                       seq_len, full_q_heads_, full_kv_heads_, full_head_dim_);
            } else {
                ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores,
                                              seq_len, start_pos, full_q_heads_, full_kv_heads_,
                                              full_head_dim_, max_seq_len_);
            }
            if (debug_this_layer && seq_len > 0) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_pre_gate", i);
                log_row_stats(tag, buf_.attn_out, dbg_row, full_q_dim_);
            }

            if (cfg_.attn_output_gate) {
                ops::sigmoid_mul(ctx, buf_.attn_out, buf_.z, buf_.attn_out, seq_len * full_q_dim_);
                if (debug_this_layer && seq_len > 0) {
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_post_gate", i);
                    log_row_stats(tag, buf_.attn_out, dbg_row, full_q_dim_);
                }
            }
            layer.o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);
        }
        if (debug_this_layer && seq_len > 0) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "layer%d_token_mixer_out", i);
            log_row_stats(tag, buf_.gate, dbg_row, hidden_size_);
        }

        ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate, *layer.post_attn_ln_weight,
                                cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
        if (debug_this_layer && seq_len > 0) {
            char tag1[64];
            char tag2[64];
            std::snprintf(tag1, sizeof(tag1), "layer%d_after_attn_residual", i);
            std::snprintf(tag2, sizeof(tag2), "layer%d_post_attn_ln_out", i);
            log_row_stats(tag1, buf_.hidden, dbg_row, hidden_size_);
            log_row_stats(tag2, buf_.normed, dbg_row, hidden_size_);
        }

        if (seq_len == 1) {
            layer.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, ff_dim_);
        } else {
            layer.gate_proj.forward(ctx, buf_.normed, buf_.gate, seq_len);
            layer.up_proj.forward(ctx, buf_.normed, buf_.up, seq_len);
            ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * ff_dim_);
        }
        layer.down_proj.forward(ctx, buf_.gate, buf_.up, seq_len);

        if (i < cfg_.num_hidden_layers - 1) {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up, *layers_[i + 1].input_ln_weight,
                                    cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
        } else {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up, *final_norm_weight_,
                                    cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
        }
        if (debug_this_layer && seq_len > 0) {
            char tag1[64];
            char tag2[64];
            std::snprintf(tag1, sizeof(tag1), "layer%d_after_mlp_residual", i);
            std::snprintf(tag2, sizeof(tag2), "layer%d_next_input_ln_out", i);
            log_row_stats(tag1, buf_.hidden, dbg_row, hidden_size_);
            log_row_stats(tag2, buf_.normed, dbg_row, hidden_size_);
        }

        if (debug_layer_stats && seq_len > 0) {
            const int row = seq_len - 1;
            bf16* hidden_ptr = static_cast<bf16*>(buf_.hidden.data()) + (size_t)row * hidden_size_;
            std::vector<bf16> h_row((size_t)hidden_size_);
            ctx.memcpy_d2h(h_row.data(), hidden_ptr, h_row.size() * sizeof(bf16));

            double mean = 0.0;
            double abs_mean = 0.0;
            float max_abs = 0.0f;
            for (int d = 0; d < hidden_size_; ++d) {
                float v = static_cast<float>(h_row[(size_t)d]);
                mean += v;
                abs_mean += std::abs(v);
                max_abs = std::max(max_abs, std::abs(v));
            }
            mean /= (double)hidden_size_;
            abs_mean /= (double)hidden_size_;
            float v0 = static_cast<float>(h_row[0]);
            float v1 = static_cast<float>(h_row[1]);
            float v2 = static_cast<float>(h_row[2]);
            float v3 = static_cast<float>(h_row[3]);
            AILA_LOG_INFO("[Q35LayerStat] layer=%d mean=%.6f abs_mean=%.6f max_abs=%.6f v0=%.6f v1=%.6f v2=%.6f v3=%.6f",
                          i, mean, abs_mean, max_abs, v0, v1, v2, v3);
        }
    }

    if (seq_len > 1) {
        bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * hidden_size_;
        Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)hidden_size_});
        lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
    } else {
        lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
    }

    current_len_ += seq_len;
    return buf_.logits;
}

void Qwen35HybridTextBackend::reset() {
    current_len_ = 0;
    if (!use_delta_linear_) return;
    for (size_t i = 0; i < layer_caches_.size(); ++i) {
        auto& layer = layers_[i];
        auto& cache = layer_caches_[i];
        if (!layer.is_linear) continue;
        if (cache.linear_state.valid()) {
            cache.linear_state.context()->queue().memset(
                cache.linear_state.data(), 0, cache.linear_state.size_bytes());
        }
        if (cache.linear_conv_state.valid()) {
            cache.linear_conv_state.context()->queue().memset(
                cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
        }
        std::fill(cache.host_linear_state.begin(), cache.host_linear_state.end(), 0.0f);
        std::fill(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(), 0.0f);
    }
}

void Qwen35HybridTextBackend::truncate_kv_cache(int new_len) {
    if (new_len >= current_len_) return;
    if (new_len == 0) {
        reset();
        return;
    }
    // Recurrent linear layers cannot be safely "trimmed" without replay, so reset conservatively.
    reset();
}
