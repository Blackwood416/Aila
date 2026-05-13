#include "Qwen35HybridBnb4Backend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <array>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>

using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl::ext::oneapi::experimental::matrix;

namespace {
int round_up_seq(int v, int granularity) {
    return ((v + granularity - 1) / granularity) * granularity;
}

enum class ProfileStage : int {
    EmbedNorm = 0,
    LinearProj,
    LinearDelta,
    LinearOProj,
    FullQkvProj,
    FullSplit,
    QkNormRope,
    KvCache,
    Attention,
    AttnGate,
    FullOProj,
    PostAttnNorm,
    FfnProj,
    FfnAct,
    DownProj,
    PostMlpNorm,
    LmHead,
    Count,
};

struct StageProfileTotals {
    std::array<double, static_cast<size_t>(ProfileStage::Count)> stage_ms{};
    int calls = 0;
    int tokens = 0;

    void reset() {
        stage_ms.fill(0.0);
        calls = 0;
        tokens = 0;
    }
};

bool supports_jm_bf16_f32(const sycl::device& dev, int m, int n, int k) {
    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return false;
    }

    using matrix_type = sycl::ext::oneapi::experimental::matrix::matrix_type;
    using sycl::ext::oneapi::experimental::info::device::matrix_combinations;
    auto combos = dev.get_info<matrix_combinations>();
    for (const auto& c : combos) {
        bool m_ok = (static_cast<int>(c.msize) == 0) || (static_cast<int>(c.msize) == m);
        if (m_ok && static_cast<int>(c.nsize) == n &&
            static_cast<int>(c.ksize) == k &&
            c.atype == matrix_type::bf16 &&
            c.btype == matrix_type::bf16 &&
            c.ctype == matrix_type::fp32 &&
            c.dtype == matrix_type::fp32) {
            return true;
        }
    }
    return false;
}

bool supports_linear_delta_gqa_fastpath(int q_heads, int kv_heads,
                                        int head_k_dim, int head_v_dim,
                                        int kernel, int conv_rows,
                                        bool allow_grouped_fastpath = false) {
    return q_heads > 0 &&
           kv_heads > 0 &&
           (kv_heads % q_heads) == 0 &&
           (allow_grouped_fastpath || q_heads == kv_heads) &&
           head_k_dim == head_v_dim &&
           head_k_dim == 128 &&
           kernel == 4 &&
           conv_rows == 3;
}

bool supports_default_grouped_linear_delta_fastpath(int q_heads, int kv_heads,
                                                    int head_k_dim, int head_v_dim,
                                                    int kernel, int conv_rows) {
    return kv_heads > q_heads &&
           supports_linear_delta_gqa_fastpath(q_heads, kv_heads,
                                              head_k_dim, head_v_dim,
                                              kernel, conv_rows,
                                              true);
}

int linear_delta_gqa_group_size(int q_heads, int kv_heads) {
    if (q_heads <= 0 || kv_heads <= 0 || (kv_heads % q_heads) != 0) {
        return 0;
    }
    return kv_heads / q_heads;
}

int linear_delta_key_head_for_value_head(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return std::clamp(value_head, 0, std::max(0, q_heads - 1));
    }
    return std::min(value_head / group_size, q_heads - 1);
}

bool linear_delta_is_group_leader(int value_head, int q_heads, int kv_heads) {
    int group_size = linear_delta_gqa_group_size(q_heads, kv_heads);
    if (group_size <= 0) {
        return value_head < q_heads;
    }
    return (value_head % group_size) == 0;
}
}

float Qwen35HybridBnb4Backend::softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

float Qwen35HybridBnb4Backend::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void Qwen35HybridBnb4Backend::head_l2_norm_inplace(std::vector<float>& x,
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

void Qwen35HybridBnb4Backend::head_rms_norm_and_silu_gate(std::vector<float>& x,
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

void Qwen35HybridBnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
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

void Qwen35HybridBnb4Backend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(
        ctx,
        {(int64_t)max_attn_heads_, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);
    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3.5] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen35HybridBnb4Backend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
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

Qwen35HybridBnb4Backend::~Qwen35HybridBnb4Backend() {
    clear_mrope_positions();
}

bool Qwen35HybridBnb4Backend::load(Context& ctx,
                                   ModelWeights& weights,
                                   const ModelSpec& spec,
                                   int max_seq_len,
                                   std::string* error_message) {
    if (spec.family != ModelFamily::Qwen35Hybrid) {
        if (error_message) *error_message = "Qwen35HybridBnb4Backend: invalid model family";
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        if (error_message) *error_message = "Qwen35HybridBnb4Backend: model is not a bitsandbytes 4-bit checkpoint";
        return false;
    }
    if (!is_supported_qwen35_hybrid_text_spec(spec.qwen35_text)) {
        if (error_message) *error_message = "Qwen35HybridBnb4Backend: only the supported Qwen3.5 hybrid specs are enabled";
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
    if (!use_delta_linear_) {
        if (error_message) *error_message = "Qwen35HybridBnb4Backend: AILA_Q35_LINEAR_DELTA=1 is required";
        return false;
    }
    bool force_direct_tied_lm_head = aila::env::read_flag("AILA_Q35_DIRECT_TIED_LM_HEAD", false);
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool experimental_grouped_linear_gpu = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", false);
    bool force_host_grouped_linear = aila::env::read_flag("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", false);
    bool is_exact_q35_0p8b_spec = is_exact_qwen35_hybrid_0p8b_spec(cfg_);

    int linear_conv_rows = std::max(0, linear_conv_kernel_dim_ - 1);
    bool default_grouped_linear_fastpath = supports_default_grouped_linear_delta_fastpath(
        linear_q_heads_,
        linear_kv_heads_,
        linear_head_dim_,
        cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_,
        linear_conv_rows);
    bool linear_fastpath_eligible = supports_linear_delta_gqa_fastpath(
        linear_q_heads_,
        linear_kv_heads_,
        linear_head_dim_,
        cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_,
        linear_conv_rows,
        experimental_gqa_fastpath || default_grouped_linear_fastpath);
    bool use_grouped_linear_gpu_default = !force_host_grouped_linear && default_grouped_linear_fastpath;
    bool use_grouped_linear_gpu_experimental =
        !use_grouped_linear_gpu_default && !force_host_grouped_linear && experimental_grouped_linear_gpu;
    int linear_gqa_group_size = linear_delta_gqa_group_size(linear_q_heads_, linear_kv_heads_);

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
    auto ensure_f32_weight = [&](const std::string& name) -> Tensor* {
        Tensor& src = weights.get(name);
        if (src.dtype() == dnnl::memory::data_type::f32) {
            return &src;
        }
        if (src.dtype() != dnnl::memory::data_type::bf16) {
            if (error_message) {
                *error_message = "Qwen35HybridBnb4Backend: expected bf16/f32 tensor for " + name;
            }
            return nullptr;
        }
        size_t n = (size_t)src.numel();
        std::vector<bf16> host_bf16(n);
        std::vector<float> host_f32(n);
        ctx.memcpy_d2h(host_bf16.data(), src.data(), n * sizeof(bf16));
        for (size_t i = 0; i < n; ++i) {
            host_f32[i] = static_cast<float>(host_bf16[i]);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), dnnl::memory::data_type::f32);
        ctx.memcpy_h2d(dst.data(), host_f32.data(), n * sizeof(float));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };
    auto erase_language_weight = [&](const std::string& name) {
        if (name.rfind("model.language_model.", 0) == 0 && weights.has(name)) {
            weights.erase(name);
        }
    };
    auto erase_bnb_weight = [&](const std::string& name) {
        erase_language_weight(name);
        erase_language_weight(name + ".quant_state.bitsandbytes__nf4");
        erase_language_weight(name + ".absmax.bitsandbytes__nf4");
        erase_language_weight(name + ".nested_absmax.bitsandbytes__nf4");
        erase_language_weight(name + ".nested_quant_map.bitsandbytes__nf4");
        // NOTE: quant_map is NOT erased — fused linears still reference it.
    };
    auto init_quant_linear = [&](const std::string& name, Bnb4BitLinear& linear) -> bool {
        Bnb4BitWeightRef ref;
        std::string local_error;
        if (!LoadBnb4BitWeightRef(ctx, weights, name, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        if (!linear.init(ctx, ref, &local_error)) {
            if (error_message) *error_message = local_error;
            return false;
        }
        return true;
    };

    embed_weight_ = &weights.get("model.language_model.embed_tokens.weight");

    layers_.clear();
    layer_caches_.clear();
    fused_weights_.clear();
    layers_.resize(cfg_.num_hidden_layers);
    layer_caches_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 6 + 1);

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
        std::vector<std::string> weights_to_erase;

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
            if (!init_quant_linear(prefix + "linear_attn.out_proj.weight", layer.linear_o_proj)) {
                return false;
            }

            if (use_delta_linear_) {
                Bnb4BitLinear linear_qkv_proj;
                Bnb4BitLinear linear_z_proj;
                Bnb4BitLinear linear_a_proj;
                Bnb4BitLinear linear_b_proj;
                if (!init_quant_linear(prefix + "linear_attn.in_proj_qkv.weight", linear_qkv_proj)) {
                    return false;
                }
                if (!init_quant_linear(prefix + "linear_attn.in_proj_z.weight", linear_z_proj)) {
                    return false;
                }
                if (!init_quant_linear(prefix + "linear_attn.in_proj_a.weight", linear_a_proj)) {
                    return false;
                }
                if (!init_quant_linear(prefix + "linear_attn.in_proj_b.weight", linear_b_proj)) {
                    return false;
                }
                std::string fused_error;
                if (!layer.linear_all_proj.init_fused_rows(ctx,
                                                           linear_qkv_proj,
                                                           linear_z_proj,
                                                           linear_a_proj,
                                                           linear_b_proj,
                                                           &fused_error)) {
                    if (error_message) *error_message = fused_error;
                    return false;
                }
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_qkv.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_z.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_a.weight");
                weights_to_erase.push_back(prefix + "linear_attn.in_proj_b.weight");

                layer.linear_norm_weight = ensure_f32_weight(prefix + "linear_attn.norm.weight");
                layer.linear_A_log = ensure_f32_weight(prefix + "linear_attn.A_log");
                layer.linear_dt_bias = &weights.get(prefix + "linear_attn.dt_bias");
                layer.linear_conv1d_weight = &weights.get(prefix + "linear_attn.conv1d.weight");
                if (!layer.linear_norm_weight || !layer.linear_A_log) {
                    return false;
                }

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
            } else {
                if (error_message) *error_message = "Qwen35HybridBnb4Backend: legacy linear attention path is not supported";
                return false;
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
                cache.linear_conv_head = 0;
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
                cache.linear_conv_head = 0;
            }
        } else {
            if (!init_quant_linear(prefix + "self_attn.o_proj.weight", layer.o_proj)) {
                return false;
            }
            Bnb4BitLinear q_proj;
            Bnb4BitLinear k_proj;
            Bnb4BitLinear v_proj;
            if (!init_quant_linear(prefix + "self_attn.q_proj.weight", q_proj) ||
                !init_quant_linear(prefix + "self_attn.k_proj.weight", k_proj) ||
                !init_quant_linear(prefix + "self_attn.v_proj.weight", v_proj)) {
                return false;
            }
            std::string fused_error;
            if (!layer.qkv_proj.init_fused_rows(ctx, q_proj, k_proj, v_proj, &fused_error)) {
                if (error_message) *error_message = fused_error;
                return false;
            }
            weights_to_erase.push_back(prefix + "self_attn.q_proj.weight");
            weights_to_erase.push_back(prefix + "self_attn.k_proj.weight");
            weights_to_erase.push_back(prefix + "self_attn.v_proj.weight");
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
            cache.linear_conv_head = 0;
        }

        Bnb4BitLinear gate_proj;
        Bnb4BitLinear up_proj;
        if (!init_quant_linear(prefix + "mlp.gate_proj.weight", gate_proj) ||
            !init_quant_linear(prefix + "mlp.up_proj.weight", up_proj) ||
            !init_quant_linear(prefix + "mlp.down_proj.weight", layer.down_proj)) {
            return false;
        }
        std::string fused_gate_up_error;
        if (!layer.gate_up_proj.init_fused_rows(ctx, gate_proj, up_proj, &fused_gate_up_error)) {
            if (error_message) *error_message = fused_gate_up_error;
            return false;
        }
        weights_to_erase.push_back(prefix + "mlp.gate_proj.weight");
        weights_to_erase.push_back(prefix + "mlp.up_proj.weight");

        if (!weights_to_erase.empty()) {
            ctx.synchronize();
            for (const auto& name : weights_to_erase) {
                erase_bnb_weight(name);
            }
        }
    }

    final_norm_weight_ = plus_one_norm_weight("model.language_model.norm.weight");

    bool lm_head_loaded = false;
    if (weights.has("lm_head.weight") &&
        !weights.has("lm_head.weight.quant_state.bitsandbytes__nf4") &&
        weights.get("lm_head.weight").dtype() != dnnl::memory::data_type::u8) {
        Tensor* lm_w = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        lm_head_loaded = true;
        AILA_LOG_INFO("[Qwen3.5] lm_head loaded (standalone dense)");
    } else if (cfg_.tie_word_embeddings) {
        bool tied_lm_head_ready = false;
        bool prefer_preprocessed_tied_lm_head = is_exact_q35_0p8b_spec && !force_direct_tied_lm_head;
        if (prefer_preprocessed_tied_lm_head) {
            try {
                Tensor& src = *embed_weight_;
                Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
                ops::transpose(ctx, src, dst);
                ctx.synchronize();
                weights.put("model.language_model.lm_head.weight_preprocessed", std::move(dst));
                lm_head_.init(ctx, weights.get("model.language_model.lm_head.weight_preprocessed"),
                              hidden_size_, cfg_.vocab_size, true);
                tied_lm_head_ready = true;
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, preprocessed copy)");
            } catch (const std::exception& e) {
                AILA_LOG_WARN("[Qwen3.5] Failed to preprocess tied lm_head, falling back to direct embed weight: %s",
                              e.what());
            }
        }
        if (!tied_lm_head_ready) {
            lm_head_.init(ctx, *embed_weight_, hidden_size_, cfg_.vocab_size, false);
            lm_head_loaded = true;
            if (force_direct_tied_lm_head) {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight forced via AILA_Q35_DIRECT_TIED_LM_HEAD=1)");
            } else if (prefer_preprocessed_tied_lm_head) {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight fallback)");
            } else {
                AILA_LOG_INFO("[Qwen3.5] lm_head (tied, direct embed weight for hybrid spec)");
            }
        }
    }
    if (!lm_head_loaded) {
        // tie_word_embeddings=false with quantized lm_head — dequantize on GPU
        Bnb4BitWeightRef lm_ref;
        if (!LoadBnb4BitWeightRef(ctx, weights, "lm_head.weight", lm_ref, nullptr)) {
            throw std::runtime_error("Qwen3.5 BNB: tie_word_embeddings=false but failed to load lm_head.weight ref");
        }
        int64_t lm_in = lm_ref.logical_in_features();
        int64_t lm_out = lm_ref.logical_out_features();
        int lm_blocksize = lm_ref.quant_state.blocksize;

        Tensor lm_bf16 = Tensor::allocate(ctx, {lm_in, lm_out}, dnnl::memory::data_type::bf16);
        const uint8_t* p_ptr = static_cast<const uint8_t*>(lm_ref.packed_weight->data());
        const float* qm_ptr = static_cast<const float*>(lm_ref.quant_map->data());
        const float* am_ptr = static_cast<const float*>(lm_ref.absmax->data());
        bf16* out_ptr = static_cast<bf16*>(lm_bf16.data());
        int64_t total_vals = lm_in * lm_out;
        int64_t packed_bytes = lm_ref.packed_num_bytes();

        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(packed_bytes)), [=](sycl::id<1> idx) {
            int64_t bi = static_cast<int64_t>(idx[0]);
            uint8_t p = p_ptr[bi];
            uint8_t hi = static_cast<uint8_t>((p >> 4) & 0x0F);
            uint8_t lo = static_cast<uint8_t>(p & 0x0F);
            int64_t f0 = bi * 2;
            if (f0 < total_vals) {
                int64_t r = f0 / lm_in, c = f0 % lm_in;
                out_ptr[c * lm_out + r] = bf16(qm_ptr[hi] * am_ptr[f0 / lm_blocksize]);
            }
            int64_t f1 = f0 + 1;
            if (f1 < total_vals) {
                int64_t r = f1 / lm_in, c = f1 % lm_in;
                out_ptr[c * lm_out + r] = bf16(qm_ptr[lo] * am_ptr[f1 / lm_blocksize]);
            }
        });
        ctx.synchronize();

        lm_head_.init(ctx, lm_bf16, hidden_size_, cfg_.vocab_size, true);
        lm_head_loaded = true;
        AILA_LOG_INFO("[Qwen3.5] lm_head loaded (standalone, dequantized from NF4)");
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
    if (full_head_dim_ == 256) {
        constexpr int kDecodeExactTile = 128;
        constexpr int kDecodeExactPartialStride = 272;
        const int max_tiles = (max_seq_len_ + kDecodeExactTile - 1) / kDecodeExactTile;
        buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {(int64_t)max_attn_heads_, (int64_t)max_tiles, (int64_t)kDecodeExactPartialStride},
            dnnl::memory::data_type::f32);
    }

    AILA_LOG_INFO("[Qwen3.5 BnB4] Hybrid backend loaded: layers=%d hidden=%d vocab=%d",
                  cfg_.num_hidden_layers, hidden_size_, cfg_.vocab_size);
    AILA_LOG_INFO("[Qwen3.5 BnB4] Full attention: q_heads=%d kv_heads=%d head_dim=%d",
                  full_q_heads_, full_kv_heads_, full_head_dim_);
    AILA_LOG_INFO("[Qwen3.5 BnB4] Linear attention: q_heads=%d kv_heads=%d key_dim=%d value_dim=%d conv_kernel=%d exact_0p8b=%d",
                  linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
                  linear_conv_kernel_dim_, is_exact_q35_0p8b_spec ? 1 : 0);
    AILA_LOG_INFO("[Qwen3.5 BnB4] Linear fast path: eligible=%d gqa_group=%d head_dims_match=%d key_dim=%d value_dim=%d kernel=%d conv_rows=%d experimental_gqa=%d default_grouped=%d force_host=%d",
                  linear_fastpath_eligible ? 1 : 0,
                  linear_gqa_group_size,
                  (linear_head_dim_ == cfg_.linear_value_head_dim) ? 1 : 0,
                  linear_head_dim_,
                  cfg_.linear_value_head_dim,
                  linear_conv_kernel_dim_,
                  linear_conv_rows,
                  experimental_gqa_fastpath ? 1 : 0,
                  default_grouped_linear_fastpath ? 1 : 0,
                  force_host_grouped_linear ? 1 : 0);
    if (linear_kv_heads_ > linear_q_heads_) {
        const char* grouped_backend = "host-reference";
        if (use_grouped_linear_gpu_default) {
            grouped_backend = "gpu-fastpath-default";
        } else if (use_grouped_linear_gpu_experimental) {
            grouped_backend = "gpu-experimental";
        }
        AILA_LOG_INFO("[Qwen3.5 BnB4] Grouped linear delta backend: %s", grouped_backend);
        if (force_host_grouped_linear) {
            AILA_LOG_INFO("[Qwen3.5 BnB4] Grouped linear host fallback forced by AILA_Q35_FORCE_HOST_GROUPED_LINEAR=1");
        } else if (!default_grouped_linear_fastpath) {
            AILA_LOG_INFO("[Qwen3.5 BnB4] Set AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU=1 to try the generic grouped GPU path");
        }
    }
    AILA_LOG_INFO("[Qwen3.5 BnB4] Decode FFN custom path: disabled");
    AILA_LOG_INFO("[Qwen3.5 BnB4] Linear mode: delta-prefill-gpu-iter/decode-gpu-ring");
    AILA_LOG_INFO("[Qwen3.5 BnB4] Legacy linear-attn fallback is not supported in the BnB4 backend");
    return true;
}

void Qwen35HybridBnb4Backend::set_embedding_overrides(
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

void Qwen35HybridBnb4Backend::clear_embedding_overrides() {
    embed_override_positions_.clear();
    embed_override_values_.clear();
    embed_override_hidden_size_ = 0;
}

void Qwen35HybridBnb4Backend::set_mrope_positions(Context& ctx,
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

void Qwen35HybridBnb4Backend::clear_mrope_positions() {
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

void Qwen35HybridBnb4Backend::run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                                          Tensor& qkv_src, Tensor& z_src,
                                                          Tensor& a_src, Tensor& b_src) {
    run_linear_delta_decode_gpu(ctx, layer, cache, qkv_src, z_src, a_src, b_src, buf_.attn_out);
}

void Qwen35HybridBnb4Backend::run_linear_delta_decode_gpu(Context& ctx, Layer& layer, LayerCache& cache,
                                                          Tensor& qkv_src, Tensor& z_src,
                                                          Tensor& a_src, Tensor& b_src,
                                                          Tensor& out_dst) {
    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int q_dim = linear_q_dim_;
    const int kv_dim = linear_kv_dim_;
    const int num_heads = linear_kv_heads_;
    const int q_heads = std::max(1, linear_q_heads_);
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;
    const int conv_rows = std::max(0, kernel - 1);
    const int conv_head = (conv_rows > 0) ? (cache.linear_conv_head % conv_rows) : 0;
    const float eps = cfg_.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    constexpr int wg = 256;

    bf16* qkv_ptr = static_cast<bf16*>(qkv_src.data());
    bf16* z_ptr = static_cast<bf16*>(z_src.data());
    bf16* a_ptr = static_cast<bf16*>(a_src.data());
    bf16* b_ptr = static_cast<bf16*>(b_src.data());
    bf16* out_ptr = static_cast<bf16*>(out_dst.data());
    float* state_ptr = static_cast<float*>(cache.linear_state.data());
    float* conv_state_ptr = cache.linear_conv_state.valid()
        ? static_cast<float*>(cache.linear_conv_state.data())
        : nullptr;
    bf16* conv_w_ptr = static_cast<bf16*>(layer.linear_conv1d_weight->data());
    float* norm_w_ptr = static_cast<float*>(layer.linear_norm_weight->data());
    float* a_log_ptr = static_cast<float*>(layer.linear_A_log->data());
    bf16* dt_bias_ptr = static_cast<bf16*>(layer.linear_dt_bias->data());
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;
        const int row0 = conv_head;
        const int row1 = (conv_head + 1) % conv_rows;
        const int row2 = (conv_head + 2) % conv_rows;
        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base = hv * head_dim;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;
                    float v_reg = 0.0f;
                    float z_reg = 0.0f;

                    if (lid < head_dim) {
                        auto conv_channel = [&](int channel) {
                            const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                            const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                            const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                            bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                            float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                      conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                      conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                      static_cast<float>(qkv_ptr[channel]) * static_cast<float>(w[3]);
                            return silu_dev(v);
                        };

                        q_local[lid] = conv_channel(q_base + lid);
                        k_local[lid] = conv_channel(k_base + lid);
                        v_reg = conv_channel(v_base + lid);
                        z_reg = static_cast<float>(z_ptr[z_base + lid]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid < head_dim) {
                        size_t dst = static_cast<size_t>(conv_head) * qkv_dim;
                        if (write_shared_qk) {
                            conv_state_ptr[dst + q_base + lid] = static_cast<float>(qkv_ptr[q_base + lid]);
                            conv_state_ptr[dst + k_base + lid] = static_cast<float>(qkv_ptr[k_base + lid]);
                        }
                        conv_state_ptr[dst + v_base + lid] = static_cast<float>(qkv_ptr[v_base + lid]);
                    }

                    float q_sq = (lid < head_dim) ? (q_local[lid] * q_local[lid]) : 0.0f;
                    float k_sq = (lid < head_dim) ? (k_local[lid] * k_local[lid]) : 0.0f;
                    q_sq = sycl::reduce_over_group(item.get_group(), q_sq, sycl::plus<float>());
                    k_sq = sycl::reduce_over_group(item.get_group(), k_sq, sycl::plus<float>());
                    float q_scale = 1.0f / sycl::sqrt(q_sq + eps);
                    float k_scale = 1.0f / sycl::sqrt(k_sq + eps);

                    if (lid < head_dim) {
                        q_local[lid] *= q_scale;
                        k_local[lid] *= k_scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (lid == 0) {
                        float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(b_ptr[hv])));
                        float alpha = static_cast<float>(a_ptr[hv]);
                        float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                    * (-sycl::native::exp(a_log_ptr[hv]));
                        head_params[0] = beta;
                        head_params[1] = sycl::native::exp(g_pre);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float beta = head_params[0];
                    float decay = head_params[1];

                    if (lid < head_dim) {
                        const int dv = lid;
                        float acc = 0.0f;
                        for (int j = 0; j < head_dim; j += 8) {
                            const size_t b = static_cast<size_t>(j) * head_dim + dv;
                            float sv0 = S[b + 0 * head_dim] * decay;
                            float sv1 = S[b + 1 * head_dim] * decay;
                            float sv2 = S[b + 2 * head_dim] * decay;
                            float sv3 = S[b + 3 * head_dim] * decay;
                            float sv4 = S[b + 4 * head_dim] * decay;
                            float sv5 = S[b + 5 * head_dim] * decay;
                            float sv6 = S[b + 6 * head_dim] * decay;
                            float sv7 = S[b + 7 * head_dim] * decay;
                            S[b + 0 * head_dim] = sv0;
                            S[b + 1 * head_dim] = sv1;
                            S[b + 2 * head_dim] = sv2;
                            S[b + 3 * head_dim] = sv3;
                            S[b + 4 * head_dim] = sv4;
                            S[b + 5 * head_dim] = sv5;
                            S[b + 6 * head_dim] = sv6;
                            S[b + 7 * head_dim] = sv7;
                            acc += sv0 * k_local[j + 0] + sv1 * k_local[j + 1] +
                                   sv2 * k_local[j + 2] + sv3 * k_local[j + 3] +
                                   sv4 * k_local[j + 4] + sv5 * k_local[j + 5] +
                                   sv6 * k_local[j + 6] + sv7 * k_local[j + 7];
                        }

                        float dval = (v_reg - acc) * beta;
                        float out = 0.0f;
                        for (int j = 0; j < head_dim; j += 8) {
                            const size_t b = static_cast<size_t>(j) * head_dim + dv;
                            float sv0 = S[b + 0 * head_dim] + k_local[j + 0] * dval;
                            float sv1 = S[b + 1 * head_dim] + k_local[j + 1] * dval;
                            float sv2 = S[b + 2 * head_dim] + k_local[j + 2] * dval;
                            float sv3 = S[b + 3 * head_dim] + k_local[j + 3] * dval;
                            float sv4 = S[b + 4 * head_dim] + k_local[j + 4] * dval;
                            float sv5 = S[b + 5 * head_dim] + k_local[j + 5] * dval;
                            float sv6 = S[b + 6 * head_dim] + k_local[j + 6] * dval;
                            float sv7 = S[b + 7 * head_dim] + k_local[j + 7] * dval;
                            S[b + 0 * head_dim] = sv0;
                            S[b + 1 * head_dim] = sv1;
                            S[b + 2 * head_dim] = sv2;
                            S[b + 3 * head_dim] = sv3;
                            S[b + 4 * head_dim] = sv4;
                            S[b + 5 * head_dim] = sv5;
                            S[b + 6 * head_dim] = sv6;
                            S[b + 7 * head_dim] = sv7;
                            out += sv0 * q_local[j + 0] + sv1 * q_local[j + 1] +
                                   sv2 * q_local[j + 2] + sv3 * q_local[j + 3] +
                                   sv4 * q_local[j + 4] + sv5 * q_local[j + 5] +
                                   sv6 * q_local[j + 6] + sv7 * q_local[j + 7];
                        }
                        out_local[dv] = out * scale;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float out_sq = (lid < head_dim) ? (out_local[lid] * out_local[lid]) : 0.0f;
                    out_sq = sycl::reduce_over_group(item.get_group(), out_sq, sycl::plus<float>());
                    float out_scale = 1.0f / sycl::sqrt(out_sq / static_cast<float>(head_dim) + eps);

                    if (lid < head_dim) {
                        float n = out_local[lid] * out_scale * norm_w_ptr[lid];
                        out_ptr[hv * head_dim + lid] = bf16(n * silu_dev(z_reg));
                    }
                });
        });
        cache.device_state_dirty = true;
        if (conv_rows > 0) {
            cache.linear_conv_head = (conv_head + 1) % conv_rows;
        }
        return;
    }

    ctx.queue().submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> q_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> k_local(sycl::range<1>(q_dim), cgh);
        sycl::local_accessor<float, 1> v_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> z_local(sycl::range<1>(z_dim), cgh);
        sycl::local_accessor<float, 1> out_local(sycl::range<1>(kv_dim), cgh);
        sycl::local_accessor<float, 1> q_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> k_scale_local(sycl::range<1>(q_heads), cgh);
        sycl::local_accessor<float, 1> out_scale_local(sycl::range<1>(num_heads), cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> item) {
                auto softplus_dev = [](float x) {
                    if (x > 20.0f) return x;
                    if (x < -20.0f) return sycl::exp(x);
                    return sycl::log1p(sycl::exp(x));
                };
                auto silu_dev = [](float x) {
                    return x / (1.0f + sycl::exp(-x));
                };

                int lid = static_cast<int>(item.get_local_id(0));

                for (int i = lid; i < z_dim; i += wg) {
                    z_local[i] = static_cast<float>(z_ptr[i]);
                }

                for (int c = lid; c < qkv_dim; c += wg) {
                    float v = 0.0f;
                    bf16* w = conv_w_ptr + static_cast<size_t>(c) * kernel;
                    if (conv_rows > 0 && conv_state_ptr) {
                        for (int j = 0; j < kernel - 1; ++j) {
                            int row = (conv_head + j) % conv_rows;
                            v += conv_state_ptr[static_cast<size_t>(row) * qkv_dim + c]
                                * static_cast<float>(w[j]);
                        }
                    }
                    v += static_cast<float>(qkv_ptr[c]) * static_cast<float>(w[kernel - 1]);
                    float y = silu_dev(v);
                    if (c < q_dim) {
                        q_local[c] = y;
                    } else if (c < q_dim + q_dim) {
                        k_local[c - q_dim] = y;
                    } else {
                        v_local[c - q_dim - q_dim] = y;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (conv_rows > 0 && conv_state_ptr) {
                    size_t dst_base = static_cast<size_t>(conv_head) * qkv_dim;
                    for (int idx = lid; idx < qkv_dim; idx += wg) {
                        conv_state_ptr[dst_base + idx] = static_cast<float>(qkv_ptr[idx]);
                    }
                }

                for (int h = lid; h < q_heads; h += wg) {
                    float q_sum = 0.0f;
                    float k_sum = 0.0f;
                    int base = h * head_k_dim;
                    for (int d = 0; d < head_k_dim; ++d) {
                        float qv = q_local[base + d];
                        float kv = k_local[base + d];
                        q_sum += qv * qv;
                        k_sum += kv * kv;
                    }
                    q_scale_local[h] = 1.0f / sycl::sqrt(q_sum + eps);
                    k_scale_local[h] = 1.0f / sycl::sqrt(k_sum + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < q_dim; idx += wg) {
                    int h = idx / head_k_dim;
                    q_local[idx] *= q_scale_local[h];
                    k_local[idx] *= k_scale_local[h];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    float* S = state_ptr + static_cast<size_t>(hv) * head_k_dim * head_v_dim;

                    float beta = 1.0f / (1.0f + sycl::exp(-static_cast<float>(b_ptr[hv])));
                    float alpha = static_cast<float>(a_ptr[hv]);
                    float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                * (-sycl::exp(a_log_ptr[hv]));
                    float decay = sycl::exp(g_pre);

                    float acc = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] * decay;
                        S[off] = sv;
                        acc += sv * k_local[hk * head_k_dim + j];
                    }

                    float d = (v_local[idx] - acc) * beta;
                    float out = 0.0f;
                    for (int j = 0; j < head_k_dim; ++j) {
                        size_t off = static_cast<size_t>(j) * head_v_dim + dv;
                        float sv = S[off] + k_local[hk * head_k_dim + j] * d;
                        S[off] = sv;
                        out += sv * q_local[hk * head_k_dim + j];
                    }
                    out_local[idx] = out * scale;
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int h = lid; h < num_heads; h += wg) {
                    float sum_sq = 0.0f;
                    int base = h * head_v_dim;
                    for (int d = 0; d < head_v_dim; ++d) {
                        float v = out_local[base + d];
                        sum_sq += v * v;
                    }
                    out_scale_local[h] = 1.0f / sycl::sqrt(
                        sum_sq / static_cast<float>(head_v_dim) + eps);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int idx = lid; idx < kv_dim; idx += wg) {
                    int hv = idx / head_v_dim;
                    int dv = idx % head_v_dim;
                    float n = out_local[idx] * out_scale_local[hv] * norm_w_ptr[dv];
                    float zg = z_local[idx];
                    out_ptr[idx] = bf16(n * silu_dev(zg));
                }
            });
    });

    cache.device_state_dirty = true;
    if (conv_rows > 0) {
        cache.linear_conv_head = (conv_head + 1) % conv_rows;
    }
}

void Qwen35HybridBnb4Backend::run_linear_delta_prefill_gpu_batched(
    Context& ctx, Layer& layer, LayerCache& cache,
    Tensor& fused_all, Tensor& out_dst, int seq_len) {

    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int q_dim = linear_q_dim_;
    const int kv_dim = linear_kv_dim_;
    const int num_heads = linear_kv_heads_;
    const int q_heads = std::max(1, linear_q_heads_);
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;
    const int conv_rows = std::max(0, kernel - 1);
    int conv_head = (conv_rows > 0) ? (cache.linear_conv_head % conv_rows) : 0;
    const float eps = cfg_.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
    const int all_dim = linear_all_dim_;

    bf16* fused_ptr = static_cast<bf16*>(fused_all.data());
    bf16* out_ptr = static_cast<bf16*>(out_dst.data());
    float* state_ptr = static_cast<float*>(cache.linear_state.data());
    float* conv_state_ptr = cache.linear_conv_state.valid()
        ? static_cast<float*>(cache.linear_conv_state.data())
        : nullptr;
    bf16* conv_w_ptr = static_cast<bf16*>(layer.linear_conv1d_weight->data());
    float* norm_w_ptr = static_cast<float*>(layer.linear_norm_weight->data());
    float* a_log_ptr = static_cast<float*>(layer.linear_A_log->data());
    bf16* dt_bias_ptr = static_cast<bf16*>(layer.linear_dt_bias->data());
    static const bool s_gqa_fastpath_cached = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", false);
    bool experimental_gqa_fastpath = s_gqa_fastpath_cached;
    bool default_grouped_fastpath = supports_default_grouped_linear_delta_fastpath(
        q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows);
    bool allow_grouped_fastpath = experimental_gqa_fastpath || default_grouped_fastpath;

    if (supports_linear_delta_gqa_fastpath(q_heads, num_heads, head_k_dim, head_v_dim, kernel, conv_rows,
                                           allow_grouped_fastpath)) {
        constexpr int head_dim = 128;
        constexpr int head_wg = 128;

        ctx.queue().submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> q_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> k_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> out_local(sycl::range<1>(head_dim), cgh);
            sycl::local_accessor<float, 1> head_params(sycl::range<1>(2), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_heads * head_wg), sycl::range<1>(head_wg)),
                [=](sycl::nd_item<1> item) {
                    auto softplus_dev = [](float x) {
                        if (x > 20.0f) return x;
                        if (x < -20.0f) return sycl::native::exp(x);
                        return sycl::log1p(sycl::native::exp(x));
                    };
                    auto silu_dev = [](float x) {
                        return x / (1.0f + sycl::native::exp(-x));
                    };

                    int hv = static_cast<int>(item.get_group(0));
                    int hk = linear_delta_key_head_for_value_head(hv, q_heads, num_heads);
                    bool write_shared_qk = linear_delta_is_group_leader(hv, q_heads, num_heads);
                    int lid = static_cast<int>(item.get_local_id(0));
                    int q_base = hk * head_dim;
                    int k_base = q_dim + hk * head_dim;
                    int v_base = q_dim + q_dim + hv * head_dim;
                    int z_base_offset = qkv_dim + hv * head_dim;
                    int a_offset = qkv_dim + z_dim;
                    int b_offset = qkv_dim + z_dim + num_heads;
                    float* S = state_ptr + static_cast<size_t>(hv) * head_dim * head_dim;

                    int local_conv_head = conv_head;

                    for (int t = 0; t < seq_len; ++t) {
                        bf16* row = fused_ptr + static_cast<size_t>(t) * all_dim;
                        bf16* out_row = out_ptr + static_cast<size_t>(t) * kv_dim;

                        float v_reg = 0.0f;
                        float z_reg = 0.0f;

                        const int row0 = local_conv_head;
                        const int row1 = (local_conv_head + 1) % conv_rows;
                        const int row2 = (local_conv_head + 2) % conv_rows;

                        if (lid < head_dim) {
                            auto conv_channel = [&](int channel) {
                                const size_t state0 = static_cast<size_t>(row0) * qkv_dim + channel;
                                const size_t state1 = static_cast<size_t>(row1) * qkv_dim + channel;
                                const size_t state2 = static_cast<size_t>(row2) * qkv_dim + channel;
                                bf16* w = conv_w_ptr + static_cast<size_t>(channel) * kernel;
                                float v = conv_state_ptr[state0] * static_cast<float>(w[0]) +
                                          conv_state_ptr[state1] * static_cast<float>(w[1]) +
                                          conv_state_ptr[state2] * static_cast<float>(w[2]) +
                                          static_cast<float>(row[channel]) * static_cast<float>(w[3]);
                                return silu_dev(v);
                            };

                            q_local[lid] = conv_channel(q_base + lid);
                            k_local[lid] = conv_channel(k_base + lid);
                            v_reg = conv_channel(v_base + lid);
                            z_reg = static_cast<float>(row[z_base_offset + lid]);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if (lid < head_dim) {
                            size_t dst = static_cast<size_t>(local_conv_head) * qkv_dim;
                            if (write_shared_qk) {
                                conv_state_ptr[dst + q_base + lid] = static_cast<float>(row[q_base + lid]);
                                conv_state_ptr[dst + k_base + lid] = static_cast<float>(row[k_base + lid]);
                            }
                            conv_state_ptr[dst + v_base + lid] = static_cast<float>(row[v_base + lid]);
                        }

                        float q_sq = (lid < head_dim) ? (q_local[lid] * q_local[lid]) : 0.0f;
                        float k_sq = (lid < head_dim) ? (k_local[lid] * k_local[lid]) : 0.0f;
                        q_sq = sycl::reduce_over_group(item.get_group(), q_sq, sycl::plus<float>());
                        k_sq = sycl::reduce_over_group(item.get_group(), k_sq, sycl::plus<float>());
                        float q_scale = 1.0f / sycl::sqrt(q_sq + eps);
                        float k_scale = 1.0f / sycl::sqrt(k_sq + eps);

                        if (lid < head_dim) {
                            q_local[lid] *= q_scale;
                            k_local[lid] *= k_scale;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        if (lid == 0) {
                            float beta = 1.0f / (1.0f + sycl::native::exp(-static_cast<float>(row[b_offset + hv])));
                            float alpha = static_cast<float>(row[a_offset + hv]);
                            float g_pre = softplus_dev(alpha + static_cast<float>(dt_bias_ptr[hv]))
                                        * (-sycl::native::exp(a_log_ptr[hv]));
                            head_params[0] = beta;
                            head_params[1] = sycl::native::exp(g_pre);
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        float beta = head_params[0];
                        float decay = head_params[1];

                        if (lid < head_dim) {
                            const int dv = lid;
                            float acc = 0.0f;
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] * decay;
                                float sv1 = S[off1] * decay;
                                float sv2 = S[off2] * decay;
                                float sv3 = S[off3] * decay;
                                S[off0] = sv0;
                                S[off1] = sv1;
                                S[off2] = sv2;
                                S[off3] = sv3;
                                acc += sv0 * k_local[j + 0] +
                                       sv1 * k_local[j + 1] +
                                       sv2 * k_local[j + 2] +
                                       sv3 * k_local[j + 3];
                            }

                            float dval = (v_reg - acc) * beta;
                            float out = 0.0f;
                            for (int j = 0; j < head_dim; j += 4) {
                                const size_t off0 = static_cast<size_t>(j + 0) * head_dim + dv;
                                const size_t off1 = static_cast<size_t>(j + 1) * head_dim + dv;
                                const size_t off2 = static_cast<size_t>(j + 2) * head_dim + dv;
                                const size_t off3 = static_cast<size_t>(j + 3) * head_dim + dv;
                                float sv0 = S[off0] + k_local[j + 0] * dval;
                                float sv1 = S[off1] + k_local[j + 1] * dval;
                                float sv2 = S[off2] + k_local[j + 2] * dval;
                                float sv3 = S[off3] + k_local[j + 3] * dval;
                                S[off0] = sv0;
                                S[off1] = sv1;
                                S[off2] = sv2;
                                S[off3] = sv3;
                                out += sv0 * q_local[j + 0] +
                                       sv1 * q_local[j + 1] +
                                       sv2 * q_local[j + 2] +
                                       sv3 * q_local[j + 3];
                            }
                            out_local[dv] = out * scale;
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        float out_sq = (lid < head_dim) ? (out_local[lid] * out_local[lid]) : 0.0f;
                        out_sq = sycl::reduce_over_group(item.get_group(), out_sq, sycl::plus<float>());
                        float out_scale = 1.0f / sycl::sqrt(out_sq / static_cast<float>(head_dim) + eps);

                        if (lid < head_dim) {
                            float n = out_local[lid] * out_scale * norm_w_ptr[lid];
                            out_row[hv * head_dim + lid] = bf16(n * silu_dev(z_reg));
                        }
                        item.barrier(sycl::access::fence_space::local_space);

                        local_conv_head = (local_conv_head + 1) % conv_rows;
                    }
                });
        });

        cache.device_state_dirty = true;
        cache.linear_conv_head = (conv_head + seq_len) % conv_rows;
        return;
    }

    // Fallback: per-token iteration for non-standard shapes
    bf16* out_base = static_cast<bf16*>(out_dst.data());
    const int64_t fused_stride = fused_all.shape(1);
    const int64_t out_stride = out_dst.shape(1);
    for (int t = 0; t < seq_len; ++t) {
        bf16* row_ptr = fused_ptr + static_cast<size_t>(t) * fused_stride;
        Tensor qkv_view = Tensor::view(ctx, row_ptr, {1, (int64_t)qkv_dim});
        Tensor z_view = Tensor::view(ctx, row_ptr + qkv_dim, {1, (int64_t)z_dim});
        Tensor a_view = Tensor::view(ctx, row_ptr + qkv_dim + z_dim,
                                     {1, (int64_t)num_heads});
        Tensor b_view = Tensor::view(ctx, row_ptr + qkv_dim + z_dim + num_heads,
                                     {1, (int64_t)num_heads});
        Tensor out_view = Tensor::view(ctx,
                                       out_base + static_cast<size_t>(t) * out_stride,
                                       {1, (int64_t)kv_dim});
        run_linear_delta_decode_gpu(ctx, layer, cache,
                                    qkv_view, z_view, a_view, b_view, out_view);
    }
}

void Qwen35HybridBnb4Backend::debug_compare_linear_delta_decode(Context& ctx, int layer_idx,
                                                                Layer& layer, LayerCache& cache,
                                                                Tensor& qkv_src, Tensor& z_src,
                                                                Tensor& a_src, Tensor& b_src) {
    std::vector<float> orig_state(cache.host_linear_state.size(), 0.0f);
    std::vector<float> orig_conv(cache.host_linear_conv_state.size(), 0.0f);
    if (cache.linear_state.valid() && !orig_state.empty()) {
        ctx.memcpy_d2h(orig_state.data(), cache.linear_state.data(),
                       orig_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !orig_conv.empty()) {
        ctx.memcpy_d2h(orig_conv.data(), cache.linear_conv_state.data(),
                       orig_conv.size() * sizeof(float));
    }

    run_linear_delta_decode_gpu(ctx, layer, cache, qkv_src, z_src, a_src, b_src);

    std::vector<bf16> gpu_out((size_t)linear_kv_dim_);
    std::vector<float> gpu_state(orig_state.size(), 0.0f);
    std::vector<float> gpu_conv(orig_conv.size(), 0.0f);
    ctx.memcpy_d2h(gpu_out.data(), buf_.attn_out.data(), gpu_out.size() * sizeof(bf16));
    if (cache.linear_state.valid() && !gpu_state.empty()) {
        ctx.memcpy_d2h(gpu_state.data(), cache.linear_state.data(),
                       gpu_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !gpu_conv.empty()) {
        ctx.memcpy_d2h(gpu_conv.data(), cache.linear_conv_state.data(),
                       gpu_conv.size() * sizeof(float));
    }

    if (cache.linear_state.valid() && !orig_state.empty()) {
        ctx.memcpy_h2d(cache.linear_state.data(), orig_state.data(),
                       orig_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !orig_conv.empty()) {
        ctx.memcpy_h2d(cache.linear_conv_state.data(), orig_conv.data(),
                       orig_conv.size() * sizeof(float));
    }
    cache.host_linear_state = orig_state;
    cache.host_linear_conv_state = orig_conv;
    cache.device_state_dirty = false;

    run_linear_delta_host(ctx, layer, cache, qkv_src, z_src, a_src, b_src, 1);

    std::vector<bf16> host_out((size_t)linear_kv_dim_);
    ctx.memcpy_d2h(host_out.data(), buf_.attn_out.data(), host_out.size() * sizeof(bf16));

    auto max_abs_bf16 = [](const std::vector<bf16>& a, const std::vector<bf16>& b,
                           size_t* idx_out, float* a_val_out, float* b_val_out) {
        float best = 0.0f;
        size_t best_idx = 0;
        float best_a = 0.0f;
        float best_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float av = static_cast<float>(a[i]);
            float bv = static_cast<float>(b[i]);
            float diff = std::abs(av - bv);
            if (diff > best) {
                best = diff;
                best_idx = i;
                best_a = av;
                best_b = bv;
            }
        }
        if (idx_out) *idx_out = best_idx;
        if (a_val_out) *a_val_out = best_a;
        if (b_val_out) *b_val_out = best_b;
        return best;
    };
    auto max_abs_f32 = [](const std::vector<float>& a, const std::vector<float>& b,
                          size_t* idx_out, float* a_val_out, float* b_val_out) {
        float best = 0.0f;
        size_t best_idx = 0;
        float best_a = 0.0f;
        float best_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = std::abs(a[i] - b[i]);
            if (diff > best) {
                best = diff;
                best_idx = i;
                best_a = a[i];
                best_b = b[i];
            }
        }
        if (idx_out) *idx_out = best_idx;
        if (a_val_out) *a_val_out = best_a;
        if (b_val_out) *b_val_out = best_b;
        return best;
    };

    size_t out_idx = 0, state_idx = 0, conv_idx = 0;
    float gpu_out_val = 0.0f, host_out_val = 0.0f;
    float gpu_state_val = 0.0f, host_state_val = 0.0f;
    float gpu_conv_val = 0.0f, host_conv_val = 0.0f;
    float out_diff = max_abs_bf16(gpu_out, host_out, &out_idx, &gpu_out_val, &host_out_val);
    float state_diff = gpu_state.empty() ? 0.0f
        : max_abs_f32(gpu_state, cache.host_linear_state,
                      &state_idx, &gpu_state_val, &host_state_val);
    float conv_diff = gpu_conv.empty() ? 0.0f
        : max_abs_f32(gpu_conv, cache.host_linear_conv_state,
                      &conv_idx, &gpu_conv_val, &host_conv_val);

    AILA_LOG_INFO(
        "[Q35LinearCompare] layer=%d out_max=%.6g idx=%zu gpu=%.6g host=%.6g "
        "state_max=%.6g idx=%zu gpu=%.6g host=%.6g conv_max=%.6g idx=%zu gpu=%.6g host=%.6g",
        layer_idx, out_diff, out_idx, gpu_out_val, host_out_val,
        state_diff, state_idx, gpu_state_val, host_state_val,
        conv_diff, conv_idx, gpu_conv_val, host_conv_val);
}

void Qwen35HybridBnb4Backend::run_linear_delta_host(Context& ctx, Layer& layer, LayerCache& cache,
                                                    Tensor& qkv_src, Tensor& z_src,
                                                    Tensor& a_src, Tensor& b_src,
                                                    int seq_len) {
    const int qkv_dim = linear_qkv_dim_;
    const int z_dim = linear_z_dim_;
    const int num_heads = linear_kv_heads_;
    const int head_k_dim = linear_head_dim_;
    const int head_v_dim = cfg_.linear_value_head_dim;
    const int kernel = linear_conv_kernel_dim_;

    if (cache.device_state_dirty) {
        if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
            ctx.memcpy_d2h(cache.host_linear_state.data(), cache.linear_state.data(),
                           cache.host_linear_state.size() * sizeof(float));
        }
        if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
            ctx.memcpy_d2h(cache.host_linear_conv_state.data(), cache.linear_conv_state.data(),
                           cache.host_linear_conv_state.size() * sizeof(float));
        }
        cache.device_state_dirty = false;
    }

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
            int hk = linear_delta_key_head_for_value_head(hv, linear_q_heads_, num_heads);
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

    if (cache.linear_state.valid() && !cache.host_linear_state.empty()) {
        ctx.memcpy_h2d(cache.linear_state.data(), cache.host_linear_state.data(),
                       cache.host_linear_state.size() * sizeof(float));
    }
    if (cache.linear_conv_state.valid() && !cache.host_linear_conv_state.empty()) {
        ctx.memcpy_h2d(cache.linear_conv_state.data(), cache.host_linear_conv_state.data(),
                       cache.host_linear_conv_state.size() * sizeof(float));
    }
    cache.device_state_dirty = false;
}

Tensor& Qwen35HybridBnb4Backend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("Qwen35HybridBnb4Backend::forward: seq_len must be positive");
    }
    static const bool s_profile_decode = aila::env::read_flag("AILA_PROFILE_Q35_DECODE", false);
    static const bool s_profile_prefill = aila::env::read_flag("AILA_PROFILE_Q35_PREFILL", false);
    static const int s_decode_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q35_DECODE_EVERY", 32));
    static const int s_prefill_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q35_PREFILL_EVERY", 1));
    static const bool s_profile_host_only = aila::env::read_flag("AILA_PROFILE_Q35_HOST_ONLY", false);
    bool profile_decode = (seq_len == 1) && s_profile_decode;
    bool profile_prefill = (seq_len > 1) && s_profile_prefill;
    int profile_every = profile_decode ? s_decode_every : s_prefill_every;
    bool profile_enabled = profile_decode || profile_prefill;
    std::array<double, static_cast<size_t>(ProfileStage::Count)> stage_ms{};
    bool profile_host_only = profile_enabled && s_profile_host_only;
    auto time_stage = [&](ProfileStage stage, auto&& fn) {
        if (!profile_enabled) {
            fn();
            return;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        if (!profile_host_only) ctx.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        stage_ms[static_cast<size_t>(stage)] +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    };
    int start_pos = current_len_;
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen35HybridBnb4Backend::forward: context window exceeded");
    }

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, cached_len);
        }
    }

    time_stage(ProfileStage::EmbedNorm, [&] {
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
    });

    int full_rotary_dim = std::max(2, (int)std::floor(full_head_dim_ * cfg_.rope.partial_rotary_factor));
    full_rotary_dim = std::min(full_head_dim_, full_rotary_dim);
    if (full_rotary_dim & 1) --full_rotary_dim;
    if (full_rotary_dim <= 0) full_rotary_dim = std::min(2, full_head_dim_);
    static const bool s_debug_layer_stats = aila::env::read_flag("AILA_DEBUG_Q35_LAYER_STATS", false);
    static const int s_debug_layer_detail = aila::env::read_int_raw("AILA_DEBUG_Q35_LAYER_DETAIL", -1);
    static const bool s_debug_linear_compare = aila::env::read_flag("AILA_DEBUG_Q35_LINEAR_COMPARE", false);
    static const int s_debug_linear_compare_layer = aila::env::read_int_raw("AILA_DEBUG_Q35_LINEAR_COMPARE_LAYER", -1);
    bool debug_layer_stats = s_debug_layer_stats;
    int debug_layer_detail_idx = s_debug_layer_detail;
    bool debug_linear_compare = s_debug_linear_compare;
    int debug_linear_compare_layer = s_debug_linear_compare_layer;
    bool grouped_linear_heads = linear_kv_heads_ > linear_q_heads_;
    int linear_conv_rows = std::max(0, linear_conv_kernel_dim_ - 1);
    bool default_grouped_linear_fastpath = supports_default_grouped_linear_delta_fastpath(
        linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
        linear_conv_kernel_dim_, linear_conv_rows);
    static const bool s_experimental_grouped_linear_gpu = aila::env::read_flag("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", false);
    static const bool s_force_host_grouped_linear = aila::env::read_flag("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", false);
    bool experimental_grouped_linear_gpu = s_experimental_grouped_linear_gpu;
    bool force_host_grouped_linear = s_force_host_grouped_linear;
    bool use_grouped_linear_gpu = grouped_linear_heads && !force_host_grouped_linear &&
                                  (default_grouped_linear_fastpath || experimental_grouped_linear_gpu);
    bool use_host_grouped_linear = grouped_linear_heads && !use_grouped_linear_gpu;
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

    static const bool s_fuse_residual = aila::env::read_flag("AILA_FUSE_RESIDUAL_ADD", false);
    bf16* hidden_ptr_for_residual = nullptr;
    if (seq_len == 1 && s_fuse_residual) {
        hidden_ptr_for_residual = static_cast<bf16*>(buf_.hidden.data());
    }

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
                    time_stage(ProfileStage::LinearProj, [&] {
                        layer.linear_all_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.linear_all, seq_len);
                    });
                    time_stage(ProfileStage::LinearDelta, [&] {
                        bf16* fused_ptr = static_cast<bf16*>(buf_.linear_all.data());
                        Tensor qkv_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)linear_qkv_dim_});
                        Tensor z_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_, {1, (int64_t)linear_z_dim_});
                        Tensor a_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_ + linear_z_dim_,
                                                     {1, (int64_t)linear_kv_heads_});
                        Tensor b_view = Tensor::view(ctx, fused_ptr + linear_qkv_dim_ + linear_z_dim_ + linear_kv_heads_,
                                                     {1, (int64_t)linear_kv_heads_});
                        if (use_host_grouped_linear) {
                            run_linear_delta_host(ctx, layer, cache, qkv_view, z_view, a_view, b_view, 1);
                            return;
                        }
                        bool compare_this_layer = debug_linear_compare &&
                                                  (debug_linear_compare_layer < 0 || debug_linear_compare_layer == i);
                        if (compare_this_layer) {
                            debug_compare_linear_delta_decode(ctx, i, layer, cache, qkv_view, z_view, a_view, b_view);
                        } else {
                            run_linear_delta_decode_gpu(ctx, layer, cache, qkv_view, z_view, a_view, b_view);
                        }
                    });
                } else {
                    time_stage(ProfileStage::LinearProj, [&] {
                        layer.linear_all_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.linear_all, seq_len);
                    });
                    time_stage(ProfileStage::LinearDelta, [&] {
                        if (use_host_grouped_linear) {
                            bf16* fused_ptr = static_cast<bf16*>(buf_.linear_all.data());
                            bf16* qkv_ptr = static_cast<bf16*>(buf_.qkv.data());
                            bf16* z_ptr = static_cast<bf16*>(buf_.z.data());
                            bf16* a_ptr = static_cast<bf16*>(buf_.a.data());
                            bf16* b_ptr = static_cast<bf16*>(buf_.b.data());
                            const int row_width = linear_all_dim_;
                            const int qkv_width = linear_qkv_dim_;
                            const int z_width = linear_z_dim_;
                            const int ab_width = linear_kv_heads_;
                            ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(seq_len) * static_cast<size_t>(row_width)),
                                [=](sycl::id<1> idx) {
                                    size_t linear_idx = idx[0];
                                    int row = static_cast<int>(linear_idx / static_cast<size_t>(row_width));
                                    int col = static_cast<int>(linear_idx % static_cast<size_t>(row_width));
                                    bf16 v = fused_ptr[linear_idx];
                                    if (col < qkv_width) {
                                        qkv_ptr[static_cast<size_t>(row) * qkv_width + col] = v;
                                    } else if (col < qkv_width + z_width) {
                                        z_ptr[static_cast<size_t>(row) * z_width + (col - qkv_width)] = v;
                                    } else if (col < qkv_width + z_width + ab_width) {
                                        a_ptr[static_cast<size_t>(row) * ab_width + (col - qkv_width - z_width)] = v;
                                    } else {
                                        b_ptr[static_cast<size_t>(row) * ab_width + (col - qkv_width - z_width - ab_width)] = v;
                                    }
                                });
                            run_linear_delta_host(ctx, layer, cache,
                                                  buf_.qkv, buf_.z, buf_.a, buf_.b,
                                                  seq_len);
                            return;
                        }
                        run_linear_delta_prefill_gpu_batched(ctx, layer, cache,
                                                             buf_.linear_all, buf_.attn_out, seq_len);
                    });
                }
            } else {
                time_stage(ProfileStage::LinearProj, [&] {
                    layer.linear_qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v,
                                   seq_len, linear_q_dim_, linear_q_dim_);
                });

                time_stage(ProfileStage::QkNormRope, [&] {
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
                });

                time_stage(ProfileStage::KvCache, [&] {
                    ops::copy_to_cache(ctx, buf_.k, cache.k, seq_len, start_pos,
                                       linear_q_heads_, linear_head_dim_, max_seq_len_);
                    ops::copy_to_cache(ctx, buf_.v, cache.v, seq_len, start_pos,
                                       linear_kv_heads_, cfg_.linear_value_head_dim, max_seq_len_);
                });

                time_stage(ProfileStage::Attention, [&] {
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
                });

                time_stage(ProfileStage::AttnGate, [&] {
                    layer.linear_z_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.z, seq_len);
                    ops::sigmoid_mul(ctx, buf_.attn_out, buf_.z, buf_.attn_out, seq_len * linear_kv_dim_);
                });
            }
            time_stage(ProfileStage::LinearOProj, [&] {
                layer.linear_o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.gate, seq_len, hidden_ptr_for_residual);
            });
        } else {
            Tensor q_decode_view;
            Tensor q_gate_decode_view;
            Tensor k_decode_view;
            Tensor v_decode_view;
            Tensor* q_for_attn = &buf_.q;
            Tensor* k_for_attn = &buf_.k;
            Tensor* v_for_attn = &buf_.v;

            if (seq_len == 1) {
                time_stage(ProfileStage::FullQkvProj, [&] {
                    layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.full_qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    bf16* fused_ptr = static_cast<bf16*>(buf_.full_qkv.data());
                    if (cfg_.attn_output_gate) {
                        q_gate_decode_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)full_q_proj_dim_});
                        q_for_attn = &buf_.q;
                    } else {
                        q_decode_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)full_q_dim_});
                        q_for_attn = &q_decode_view;
                    }
                    k_decode_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_, {1, (int64_t)full_kv_dim_});
                    v_decode_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_ + full_kv_dim_, {1, (int64_t)full_kv_dim_});
                    k_for_attn = &k_decode_view;
                    v_for_attn = &v_decode_view;
                });
            } else {
                time_stage(ProfileStage::FullQkvProj, [&] {
                    layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.full_qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    ops::split_qkv(ctx, buf_.full_qkv, buf_.qkv, buf_.k, buf_.v,
                                   seq_len, full_q_proj_dim_, full_kv_dim_);
                    if (cfg_.attn_output_gate) {
                        ops::split_q_gate(ctx, buf_.qkv, buf_.q, buf_.z,
                                          seq_len, full_q_heads_, full_head_dim_);
                    } else {
                        ops::copy_tensor(ctx, buf_.qkv, buf_.q, seq_len * full_q_dim_);
                    }
                });
            }
            if (debug_this_layer && seq_len > 0 && cfg_.attn_output_gate) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_gate_z", i);
                log_row_stats(tag, buf_.z, dbg_row, full_q_dim_);
            }

            time_stage(ProfileStage::QkNormRope, [&] {
                if (seq_len == 1) {
                    if (cfg_.attn_output_gate) {
                        ops::decode_prepare_qgkv_packed_partial(ctx, q_gate_decode_view, *k_for_attn, *v_for_attn,
                                                                buf_.q, buf_.z,
                                                                *layer.q_norm_weight, *layer.k_norm_weight,
                                                                cache.k, cache.v, start_pos,
                                                                full_q_heads_, full_kv_heads_, full_head_dim_,
                                                                cfg_.rms_norm_eps, full_rotary_dim,
                                                                cfg_.rope.rope_theta,
                                                                cfg_.rope.mrope_interleaved,
                                                                mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
                                                                mrope_prompt_len_, mrope_text_pos_delta_,
                                                                cfg_.rope.mrope_section[0],
                                                                cfg_.rope.mrope_section[1],
                                                                cfg_.rope.mrope_section[2]);
                    } else {
                        ops::decode_prepare_qkv_partial(ctx, *q_for_attn, *k_for_attn, *v_for_attn,
                                                        *layer.q_norm_weight, *layer.k_norm_weight,
                                                        cache.k, cache.v, start_pos,
                                                        full_q_heads_, full_kv_heads_, full_head_dim_,
                                                        cfg_.rms_norm_eps, full_rotary_dim,
                                                        cfg_.rope.rope_theta,
                                                        cfg_.rope.mrope_interleaved,
                                                        mrope_pos_t_, mrope_pos_h_, mrope_pos_w_,
                                                        mrope_prompt_len_, mrope_text_pos_delta_,
                                                        cfg_.rope.mrope_section[0],
                                                        cfg_.rope.mrope_section[1],
                                                        cfg_.rope.mrope_section[2]);
                    }
                } else {
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
                }
            });
            if (debug_this_layer && seq_len > 0) {
                char tag_q[64];
                char tag_k[64];
                char tag_v[64];
                std::snprintf(tag_q, sizeof(tag_q), "layer%d_full_q_after_rope", i);
                std::snprintf(tag_k, sizeof(tag_k), "layer%d_full_k_after_rope", i);
                std::snprintf(tag_v, sizeof(tag_v), "layer%d_full_v", i);
                log_row_stats(tag_q, *q_for_attn, dbg_row, full_q_dim_);
                log_row_stats(tag_k, *k_for_attn, dbg_row, full_kv_dim_);
                log_row_stats(tag_v, *v_for_attn, dbg_row, full_kv_dim_);
            }

            time_stage(ProfileStage::KvCache, [&] {
                if (seq_len > 1) {
                    ops::copy_to_cache(ctx, *k_for_attn, cache.k, seq_len, start_pos,
                                       full_kv_heads_, full_head_dim_, max_seq_len_);
                    ops::copy_to_cache(ctx, *v_for_attn, cache.v, seq_len, start_pos,
                                       full_kv_heads_, full_head_dim_, max_seq_len_);
                }
            });

            time_stage(ProfileStage::Attention, [&] {
                if (seq_len == 1) {
                    ops::attention_decode(ctx, *q_for_attn, cache.k, cache.v, buf_.attn_out, buf_.decode_scores,
                                          full_q_heads_, full_kv_heads_, full_head_dim_, cached_len,
                                          &buf_.decode_attn_partials);
                } else if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores,
                                           seq_len, full_q_heads_, full_kv_heads_, full_head_dim_);
                } else {
                    ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores,
                                                  seq_len, start_pos, full_q_heads_, full_kv_heads_,
                                                  full_head_dim_, max_seq_len_);
                }
            });
            if (debug_this_layer && seq_len > 0) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_pre_gate", i);
                log_row_stats(tag, buf_.attn_out, dbg_row, full_q_dim_);
            }

            if (cfg_.attn_output_gate) {
                time_stage(ProfileStage::AttnGate, [&] {
                    ops::sigmoid_mul(ctx, buf_.attn_out, buf_.z, buf_.attn_out, seq_len * full_q_dim_);
                });
                if (debug_this_layer && seq_len > 0) {
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_post_gate", i);
                    log_row_stats(tag, buf_.attn_out, dbg_row, full_q_dim_);
                }
            }
            time_stage(ProfileStage::FullOProj, [&] {
                layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.gate, seq_len, hidden_ptr_for_residual);
            });
        }
        if (debug_this_layer && seq_len > 0) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "layer%d_token_mixer_out", i);
            log_row_stats(tag, buf_.gate, dbg_row, hidden_size_);
        }

        time_stage(ProfileStage::PostAttnNorm, [&] {
            if (hidden_ptr_for_residual) {
                ops::rms_norm(ctx, buf_.gate, *layer.post_attn_ln_weight,
                              cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
            } else {
                ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate, *layer.post_attn_ln_weight,
                                        cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
            }
        });
        if (debug_this_layer && seq_len > 0) {
            char tag1[64];
            char tag2[64];
            std::snprintf(tag1, sizeof(tag1), "layer%d_after_attn_residual", i);
            std::snprintf(tag2, sizeof(tag2), "layer%d_post_attn_ln_out", i);
            log_row_stats(tag1, buf_.hidden, dbg_row, hidden_size_);
            log_row_stats(tag2, buf_.normed, dbg_row, hidden_size_);
        }

        if (seq_len == 1) {
            // Fused gate+up GEMV + SiLU in one kernel (no SLM, vec8+FMA).
            // Falls back to separate path if packed weights are unavailable.
            bool fused_ok = false;
            time_stage(ProfileStage::FfnProj, [&] {
                fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                    ctx, layer.gate_up_proj, buf_.normed, buf_.gate, ff_dim_);
                if (!fused_ok) {
                    layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, seq_len);
                }
            });
            if (fused_ok) {
                time_stage(ProfileStage::FfnAct, [&] { });
            } else {
                time_stage(ProfileStage::FfnAct, [&] {
                    ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, ff_dim_);
                });
            }
            time_stage(ProfileStage::DownProj, [&] {
                layer.down_proj.forward(ctx, linear_scratch_, buf_.gate, buf_.up, seq_len, hidden_ptr_for_residual);
            });
        } else {
            time_stage(ProfileStage::FfnProj, [&] {
                layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::split_gate_up(ctx, buf_.gate_up, buf_.gate, buf_.up, seq_len, ff_dim_);
                ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * ff_dim_);
            });
            time_stage(ProfileStage::DownProj, [&] {
                layer.down_proj.forward(ctx, linear_scratch_, buf_.gate, buf_.up, seq_len, hidden_ptr_for_residual);
            });
        }

        time_stage(ProfileStage::PostMlpNorm, [&] {
            if (hidden_ptr_for_residual) {
                if (i < cfg_.num_hidden_layers - 1) {
                    ops::rms_norm(ctx, buf_.up, *layers_[i + 1].input_ln_weight,
                                  cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
                } else {
                    ops::rms_norm(ctx, buf_.up, *final_norm_weight_,
                                  cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
                }
            } else {
                if (i < cfg_.num_hidden_layers - 1) {
                    ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up, *layers_[i + 1].input_ln_weight,
                                            cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
                } else {
                    ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up, *final_norm_weight_,
                                            cfg_.rms_norm_eps, buf_.normed, seq_len, hidden_size_);
                }
            }
        });
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

    time_stage(ProfileStage::LmHead, [&] {
        if (seq_len > 1) {
            bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * hidden_size_;
            Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)hidden_size_});
            lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
        } else {
            lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
        }
    });

    if (profile_enabled) {
        static StageProfileTotals decode_totals;
        static StageProfileTotals prefill_totals;
        StageProfileTotals& totals = profile_decode ? decode_totals : prefill_totals;
        for (size_t i = 0; i < stage_ms.size(); ++i) {
            totals.stage_ms[i] += stage_ms[i];
        }
        totals.calls += 1;
        totals.tokens += profile_decode ? 1 : seq_len;

        if (totals.calls >= profile_every) {
            auto avg = [&](ProfileStage stage) {
                return totals.stage_ms[static_cast<size_t>(stage)] /
                       static_cast<double>(std::max(1, totals.calls));
            };
            auto per_token = [&](ProfileStage stage) {
                return totals.stage_ms[static_cast<size_t>(stage)] /
                       static_cast<double>(std::max(1, totals.tokens));
            };
            double total_ms = 0.0;
            for (double v : totals.stage_ms) total_ms += v;
            if (profile_decode) {
                AILA_LOG_INFO(
                    "[Q35DecodeProfile] tokens=%d total=%.3f embed=%.3f linear_proj=%.3f linear_delta=%.3f linear_o=%.3f "
                    "full_qkv=%.3f full_split=%.3f qk_rope=%.3f kv_copy=%.3f attn=%.3f attn_gate=%.3f full_o=%.3f "
                    "post_attn=%.3f ffn_proj=%.3f ffn_act=%.3f down=%.3f post_mlp=%.3f lm_head=%.3f",
                    totals.tokens,
                    total_ms / static_cast<double>(std::max(1, totals.tokens)),
                    avg(ProfileStage::EmbedNorm),
                    avg(ProfileStage::LinearProj),
                    avg(ProfileStage::LinearDelta),
                    avg(ProfileStage::LinearOProj),
                    avg(ProfileStage::FullQkvProj),
                    avg(ProfileStage::FullSplit),
                    avg(ProfileStage::QkNormRope),
                    avg(ProfileStage::KvCache),
                    avg(ProfileStage::Attention),
                    avg(ProfileStage::AttnGate),
                    avg(ProfileStage::FullOProj),
                    avg(ProfileStage::PostAttnNorm),
                    avg(ProfileStage::FfnProj),
                    avg(ProfileStage::FfnAct),
                    avg(ProfileStage::DownProj),
                    avg(ProfileStage::PostMlpNorm),
                    avg(ProfileStage::LmHead));
            } else {
                AILA_LOG_INFO(
                    "[Q35PrefillProfile] calls=%d tokens=%d total=%.3f total_tok=%.6f "
                    "embed=%.6f linear_proj=%.6f linear_delta=%.6f linear_o=%.6f "
                    "full_qkv=%.6f full_split=%.6f qk_rope=%.6f kv_copy=%.6f attn=%.6f attn_gate=%.6f full_o=%.6f "
                    "post_attn=%.6f ffn_proj=%.6f ffn_act=%.6f down=%.6f post_mlp=%.6f lm_head=%.6f",
                    totals.calls,
                    totals.tokens,
                    total_ms / static_cast<double>(std::max(1, totals.calls)),
                    total_ms / static_cast<double>(std::max(1, totals.tokens)),
                    per_token(ProfileStage::EmbedNorm),
                    per_token(ProfileStage::LinearProj),
                    per_token(ProfileStage::LinearDelta),
                    per_token(ProfileStage::LinearOProj),
                    per_token(ProfileStage::FullQkvProj),
                    per_token(ProfileStage::FullSplit),
                    per_token(ProfileStage::QkNormRope),
                    per_token(ProfileStage::KvCache),
                    per_token(ProfileStage::Attention),
                    per_token(ProfileStage::AttnGate),
                    per_token(ProfileStage::FullOProj),
                    per_token(ProfileStage::PostAttnNorm),
                    per_token(ProfileStage::FfnProj),
                    per_token(ProfileStage::FfnAct),
                    per_token(ProfileStage::DownProj),
                    per_token(ProfileStage::PostMlpNorm),
                    per_token(ProfileStage::LmHead));
            }
            totals.reset();
        }
    }

    current_len_ += seq_len;
    return buf_.logits;
}

void Qwen35HybridBnb4Backend::reset() {
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
        cache.device_state_dirty = false;
        cache.linear_conv_head = 0;
    }
}

bool Qwen35HybridBnb4Backend::truncate_kv_cache(int new_len) {
    if (new_len >= current_len_) return true;
    if (new_len == 0) {
        reset();
        return false;
    }
    // Truncate attention KV caches to new_len (just move the position
    // cursor — GQA layers use current_len_ as the write head).
    current_len_ = new_len;
    // DeltaNet recurrent state cannot be positionally truncated — it is a
    // single accumulated matrix.  Zero it out and let the caller replay
    // all prompt tokens through a full prefill to rebuild it.
    if (use_delta_linear_) {
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
            cache.device_state_dirty = false;
            cache.linear_conv_head = 0;
        }
    }
    return false;
}
