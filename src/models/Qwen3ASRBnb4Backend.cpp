#include "Qwen3ASRBnb4Backend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <array>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
int round_up_seq(int v, int granularity) {
    return ((v + granularity - 1) / granularity) * granularity;
}

enum class ProfileStage : int {
    EmbedNorm = 0,
    FullQkvProj,
    FullSplit,
    QkNormRope,
    KvCache,
    Attention,
    OProj,
    PostAttnNorm,
    FfnProj,
    FfnAct,
    DownProj,
    PostMlpNorm,
    LmHead,
    Count
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
}

Qwen3ASRBnb4Backend::~Qwen3ASRBnb4Backend() = default;

bool Qwen3ASRBnb4Backend::apply_lora(Context& ctx, const aila::lora::LoraAdapter& adapter,
                                    std::string* error_message) {
    int H = hidden_size_;
    int QD = q_dim_;
    int KVD = kv_dim_;
    int FF = ff_dim_;
    float scaling = adapter.config.scaling;

    auto upload_f32_to_bf16 = [&](const std::vector<float>& f32_data,
                                   int rows, int cols) -> Tensor {
        size_t n = static_cast<size_t>(rows) * cols;
        std::vector<bf16> bf16_data(n);
        for (size_t i = 0; i < n; ++i) {
            bf16_data[i] = bf16(f32_data[i]);
        }
        Tensor t = Tensor::allocate(ctx, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                    dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(t.data(), bf16_data.data(), n * sizeof(bf16));
        return t;
    };

    int num_layers = cfg_.num_hidden_layers;
    std::vector<std::vector<LoraAttachment>> qkv_attachments(num_layers);
    std::vector<std::vector<LoraAttachment>> gate_up_attachments(num_layers);

    for (const auto& pair : adapter.pairs) {
        const std::string prefix = "thinker.model.layers.";
        size_t pos = pair.weight_name.find(prefix);
        if (pos != 0) continue;

        size_t num_start = prefix.size();
        size_t num_end = pair.weight_name.find('.', num_start);
        if (num_end == std::string::npos) continue;
        int layer_idx = std::stoi(pair.weight_name.substr(num_start, num_end - num_start));
        if (layer_idx < 0 || layer_idx >= num_layers) continue;

        size_t last_dot = pair.weight_name.rfind('.');
        size_t prev_dot = pair.weight_name.rfind('.', last_dot - 1);
        std::string target = pair.weight_name.substr(prev_dot + 1, last_dot - prev_dot - 1);

        LoraAttachment att;
        att.lora_a = upload_f32_to_bf16(pair.lora_a, pair.r, pair.in_features);
        att.lora_b = upload_f32_to_bf16(pair.lora_b, pair.out_features, pair.r);
        att.scaling = scaling;

        if (target == "q_proj") {
            att.output_offset = 0;
            att.output_rows = QD;
            qkv_attachments[layer_idx].push_back(std::move(att));
        } else if (target == "k_proj") {
            att.output_offset = QD;
            att.output_rows = KVD;
            qkv_attachments[layer_idx].push_back(std::move(att));
        } else if (target == "v_proj") {
            att.output_offset = QD + KVD;
            att.output_rows = KVD;
            qkv_attachments[layer_idx].push_back(std::move(att));
        } else if (target == "o_proj") {
            att.output_offset = 0;
            att.output_rows = H;
            std::vector<LoraAttachment> tmp;
            tmp.push_back(std::move(att));
            layers_[layer_idx].o_proj.set_lora(std::move(tmp));
        } else if (target == "gate_proj") {
            att.output_offset = 0;
            att.output_rows = FF;
            gate_up_attachments[layer_idx].push_back(std::move(att));
        } else if (target == "up_proj") {
            att.output_offset = FF;
            att.output_rows = FF;
            gate_up_attachments[layer_idx].push_back(std::move(att));
        } else if (target == "down_proj") {
            att.output_offset = 0;
            att.output_rows = H;
            std::vector<LoraAttachment> tmp;
            tmp.push_back(std::move(att));
            layers_[layer_idx].down_proj.set_lora(std::move(tmp));
        }
    }

    for (int i = 0; i < num_layers; ++i) {
        if (!qkv_attachments[i].empty()) {
            layers_[i].qkv_proj.set_lora(std::move(qkv_attachments[i]));
        }
        if (!gate_up_attachments[i].empty()) {
            layers_[i].gate_up_proj.set_lora(std::move(gate_up_attachments[i]));
        }
    }

    AILA_LOG_INFO("[Qwen3ASRBnb4] LoRA adapter applied: r=%d alpha=%d scaling=%.2f",
                  adapter.config.r, adapter.config.lora_alpha, adapter.config.scaling);
    if (error_message) *error_message = "";
    return true;
}

void Qwen3ASRBnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;

    int H = hidden_size_;
    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));

    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.qkv      = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)fused_qkv_dim_});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.gate_up  = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * ff_dim_)});
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)ff_dim_});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3ASRBnb4] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen3ASRBnb4Backend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);

    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3ASRBnb4] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen3ASRBnb4Backend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;

    int new_seq_cap = std::max(seq_len, incr_prefill_seq_cap_);
    int new_total_cap = std::max(total_len, incr_prefill_total_cap_);
    new_seq_cap = round_up_seq(new_seq_cap, 16);
    new_total_cap = round_up_seq(new_total_cap, 64);

    buf_.incr_scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_seq_cap, (int64_t)new_total_cap},
        dnnl::memory::data_type::f32);

    incr_prefill_seq_cap_ = new_seq_cap;
    incr_prefill_total_cap_ = new_total_cap;
    AILA_LOG_INFO("[Qwen3ASRBnb4] Incremental prefill score buffer resized: seq_cap=%d, total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

void Qwen3ASRBnb4Backend::upload_mrope_positions(Context& ctx) {
    int n = static_cast<int>(mrope_pos_t_.size());
    if (n == 0) return;
    if (n > mrope_pos_capacity_) {
        mrope_pos_t_dev_ = Tensor::allocate(ctx, {n}, dnnl::memory::data_type::s32);
        mrope_pos_h_dev_ = Tensor::allocate(ctx, {n}, dnnl::memory::data_type::s32);
        mrope_pos_w_dev_ = Tensor::allocate(ctx, {n}, dnnl::memory::data_type::s32);
        mrope_pos_capacity_ = n;
    }
    ctx.memcpy_h2d(mrope_pos_t_dev_.data(), mrope_pos_t_.data(), n * sizeof(int));
    ctx.memcpy_h2d(mrope_pos_h_dev_.data(), mrope_pos_h_.data(), n * sizeof(int));
    ctx.memcpy_h2d(mrope_pos_w_dev_.data(), mrope_pos_w_.data(), n * sizeof(int));
}

void Qwen3ASRBnb4Backend::set_embedding_overrides(const std::vector<int>& positions,
                                                 const std::vector<bf16>& embeddings,
                                                 int hidden_size) {
    has_embedding_overrides_ = true;
    override_positions_ = positions;
    override_embeddings_ = embeddings;
    override_hidden_size_ = hidden_size;
}

void Qwen3ASRBnb4Backend::clear_embedding_overrides() {
    has_embedding_overrides_ = false;
    override_positions_.clear();
    override_embeddings_.clear();
}

void Qwen3ASRBnb4Backend::set_mrope_positions(Context& ctx,
                                             const std::vector<int>& pos_t,
                                             const std::vector<int>& pos_h,
                                             const std::vector<int>& pos_w,
                                             int text_pos_delta) {
    has_mrope_positions_ = true;
    mrope_pos_t_ = pos_t;
    mrope_pos_h_ = pos_h;
    mrope_pos_w_ = pos_w;
    mrope_text_pos_delta_ = text_pos_delta;
    upload_mrope_positions(ctx);
}

void Qwen3ASRBnb4Backend::clear_mrope_positions() {
    has_mrope_positions_ = false;
}

bool Qwen3ASRBnb4Backend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                              int max_seq_len, std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3ASR && spec.family != ModelFamily::Qwen3ForceAligner) {
        if (error_message) *error_message = "Qwen3ASRBnb4Backend: invalid model family";
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        if (error_message) *error_message = "Qwen3ASRBnb4Backend: model is not a bitsandbytes 4-bit checkpoint";
        return false;
    }

    cfg_ = spec.qwen3;
    rope_ = cfg_.rope;
    max_seq_len_ = max_seq_len;

    hidden_size_ = cfg_.hidden_size;
    q_dim_ = cfg_.num_attention_heads * cfg_.head_dim;
    kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
    fused_qkv_dim_ = q_dim_ + 2 * kv_dim_;
    ff_dim_ = cfg_.intermediate_size;

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

    auto erase_weight = [&](const std::string& name) {
        if (weights.has(name)) {
            weights.erase(name);
        }
    };
    auto erase_bnb_weight = [&](const std::string& name) {
        erase_weight(name);
        erase_weight(name + ".quant_state.bitsandbytes__nf4");
        erase_weight(name + ".absmax.bitsandbytes__nf4");
        erase_weight(name + ".nested_absmax.bitsandbytes__nf4");
        erase_weight(name + ".nested_quant_map.bitsandbytes__nf4");
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

    embed_weight_ = &weights.get("thinker.model.embed_tokens.weight");
    AILA_LOG_INFO("[Qwen3ASRBnb4] embed_tokens loaded");

    layers_.clear();
    fused_weights_.clear();
    layers_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 2 + 1);

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "thinker.model.layers." + std::to_string(i) + ".";
        std::vector<std::string> weights_to_erase;

        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

        if (!init_quant_linear(prefix + "self_attn.o_proj.weight", layer.o_proj)) return false;

        Bnb4BitLinear q_proj, k_proj, v_proj;
        if (!init_quant_linear(prefix + "self_attn.q_proj.weight", q_proj) ||
            !init_quant_linear(prefix + "self_attn.k_proj.weight", k_proj) ||
            !init_quant_linear(prefix + "self_attn.v_proj.weight", v_proj)) {
            return false;
        }
        std::string fused_qkv_error;
        if (!layer.qkv_proj.init_fused_rows(ctx, q_proj, k_proj, v_proj, &fused_qkv_error)) {
            if (error_message) *error_message = fused_qkv_error;
            return false;
        }
        weights_to_erase.push_back(prefix + "self_attn.q_proj.weight");
        weights_to_erase.push_back(prefix + "self_attn.k_proj.weight");
        weights_to_erase.push_back(prefix + "self_attn.v_proj.weight");

        layer.q_norm_weight = &weights.get(prefix + "self_attn.q_norm.weight");
        layer.k_norm_weight = &weights.get(prefix + "self_attn.k_norm.weight");

        Bnb4BitLinear gate_proj, up_proj;
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
    AILA_LOG_INFO("[Qwen3ASRBnb4] %d transformer layers loaded", cfg_.num_hidden_layers);

    final_norm_weight_ = &weights.get("thinker.model.norm.weight");

    kv_cache_.init(ctx, cfg_, max_seq_len, "AILA_ASR_KV_QUANT");

    if (weights.has("thinker.lm_head.weight") &&
        !weights.has("thinker.lm_head.weight.quant_state.bitsandbytes__nf4") &&
        weights.get("thinker.lm_head.weight").dtype() != dnnl::memory::data_type::u8) {
        Tensor* lm_w = transpose_weight("thinker.lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3ASRBnb4] lm_head loaded (dense)");
    } else {
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.put("thinker.lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("thinker.lm_head.weight_preprocessed"), hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3ASRBnb4] lm_head tied");
    }

    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    incr_prefill_seq_cap_ = incr_prefill_total_cap_ = 0;
    ensure_runtime_buffers(ctx, 1);

    buf_.logits = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
    buf_.decode_scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)max_seq_len},
        dnnl::memory::data_type::f32);
    buf_.rope_freq = Tensor::allocate(ctx,
        {(int64_t)(cfg_.head_dim / 2)},
        dnnl::memory::data_type::f32);
    {
        std::vector<float> rope_freq_host(static_cast<size_t>(cfg_.head_dim / 2));
        for (int d = 0; d < cfg_.head_dim / 2; ++d) {
            rope_freq_host[static_cast<size_t>(d)] =
                1.0f / std::pow(cfg_.rope_theta, (2.0f * d) / static_cast<float>(cfg_.head_dim));
        }
        ctx.memcpy_h2d(buf_.rope_freq.data(), rope_freq_host.data(),
                       rope_freq_host.size() * sizeof(float));
    }

    if (cfg_.head_dim == 256 || cfg_.head_dim == 128) {
        constexpr int kDecodeExactTile = 128;
        const int partial_stride = (cfg_.head_dim == 256) ? 272 : 144;
        const int max_tiles = (max_seq_len + kDecodeExactTile - 1) / kDecodeExactTile;
        buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {(int64_t)cfg_.num_attention_heads, (int64_t)max_tiles, (int64_t)partial_stride},
            dnnl::memory::data_type::f32);
    }
    override_buf_ = Tensor();

    AILA_LOG_INFO("[Qwen3ASRBnb4] Model fully loaded and initialized");
    return true;
}

Tensor& Qwen3ASRBnb4Backend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("Qwen3ASRBnb4Backend::forward: seq_len must be positive");
    }

    static const bool s_profile_decode = aila::env::read_flag("AILA_PROFILE_Q3_DECODE", false);
    static const bool s_profile_prefill = aila::env::read_flag("AILA_PROFILE_Q3_PREFILL", false);
    static const int s_decode_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q3_DECODE_EVERY", 32));
    static const int s_prefill_every = std::max(1, aila::env::read_int_raw("AILA_PROFILE_Q3_PREFILL_EVERY", 1));
    static const bool s_profile_host_only = aila::env::read_flag("AILA_PROFILE_Q3_HOST_ONLY", false);
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

    int H = hidden_size_;
    int QD = q_dim_;
    int KVD = kv_dim_;
    int FF = ff_dim_;
    int start_pos = current_len_;
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen3ASRBnb4Backend::forward: context window exceeded");
    }

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, start_pos + seq_len);
        }
    }

    static const bool s_fuse_residual = aila::env::read_flag("AILA_FUSE_RESIDUAL_ADD", false);
    bf16* hidden_ptr_for_residual = nullptr;
    if (seq_len == 1 && s_fuse_residual) {
        hidden_ptr_for_residual = static_cast<bf16*>(buf_.hidden.data());
    }

    time_stage(ProfileStage::EmbedNorm, [&] {
        // 1. Embedding lookup
        ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

        // 2. Apply audio embedding overrides
        if (has_embedding_overrides_ && !override_positions_.empty()) {
            size_t n_override = override_positions_.size();
            if (!override_buf_.valid() || override_buf_.numel() < static_cast<int64_t>(n_override * H)) {
                override_buf_ = Tensor::allocate(ctx, {static_cast<int64_t>(n_override), H});
            }
            ctx.memcpy_h2d(override_buf_.data(), override_embeddings_.data(),
                           n_override * H * sizeof(bf16));

            if (!override_pos_buf_.valid() || override_pos_buf_.numel() < static_cast<int64_t>(n_override)) {
                override_pos_buf_ = Tensor::allocate(ctx, {static_cast<int64_t>(n_override)},
                                                      dnnl::memory::data_type::s32);
            }
            ctx.memcpy_h2d(override_pos_buf_.data(), override_positions_.data(),
                           n_override * sizeof(int));

            bf16* hidden_ptr = buf_.hidden.data_as<bf16>();
            bf16* over_ptr = override_buf_.data_as<bf16>();
            int* pos_ptr = override_pos_buf_.data_as<int>();

            ctx.queue().parallel_for(sycl::range<1>(n_override), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                int pos = pos_ptr[i];
                for (int j = 0; j < H; ++j) {
                    hidden_ptr[pos * H + j] = over_ptr[i * H + j];
                }
            });
            has_embedding_overrides_ = false;
        }

        // 3. Initial norm
        ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                      cfg_.rms_norm_eps, buf_.normed, seq_len, H);
    });

    // MRoPE dimension
    int rotary_dim = cfg_.head_dim;
    if (rotary_dim & 1) --rotary_dim;

    const int* pos_t = has_mrope_positions_ && mrope_pos_t_dev_.valid()
                       ? mrope_pos_t_dev_.data_as<int>() : nullptr;
    const int* pos_h = has_mrope_positions_ && mrope_pos_h_dev_.valid()
                       ? mrope_pos_h_dev_.data_as<int>() : nullptr;
    const int* pos_w = has_mrope_positions_ && mrope_pos_w_dev_.valid()
                       ? mrope_pos_w_dev_.data_as<int>() : nullptr;

    // 4. Iterate layers
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        Tensor q_decode_view;
        Tensor* q_for_attn = &buf_.q;

        if (seq_len == 1) {
            time_stage(ProfileStage::FullQkvProj, [&] {
                layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.qkv, 1);
            });
            bf16* qkv_ptr = static_cast<bf16*>(buf_.qkv.data());
            q_decode_view = Tensor::view(ctx, qkv_ptr, {1, (int64_t)QD});
            Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, (int64_t)KVD});
            Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, (int64_t)KVD});
            q_for_attn = &q_decode_view;

            time_stage(ProfileStage::QkNormRope, [&] {
                ops::decode_prepare_qkv_partial(ctx,
                    q_decode_view, k_view, v_view,
                    *layer.q_norm_weight, *layer.k_norm_weight,
                    kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                    start_pos,
                    cfg_.num_attention_heads, cfg_.num_key_value_heads, cfg_.head_dim,
                    cfg_.rms_norm_eps, rotary_dim, cfg_.rope_theta,
                    rope_.mrope_interleaved,
                    pos_t, pos_h, pos_w,
                    static_cast<int>(mrope_pos_t_.size()),
                    mrope_text_pos_delta_,
                    rope_.mrope_section[0], rope_.mrope_section[1], rope_.mrope_section[2]);
            });
        } else {
            time_stage(ProfileStage::FullQkvProj, [&] {
                layer.qkv_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.qkv, seq_len);
            });
            time_stage(ProfileStage::FullSplit, [&] {
                ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v, seq_len, QD, KVD);
            });

            time_stage(ProfileStage::QkNormRope, [&] {
                ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                                   cfg_.rms_norm_eps, seq_len,
                                   cfg_.num_attention_heads, cfg_.head_dim);
                ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight,
                                   cfg_.rms_norm_eps, seq_len,
                                   cfg_.num_key_value_heads, cfg_.head_dim);

                ops::apply_rope_partial(ctx, buf_.q, buf_.k, seq_len, start_pos,
                                        cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                        cfg_.head_dim, rotary_dim, cfg_.rope_theta,
                                        rope_.mrope_interleaved,
                                        pos_t, pos_h, pos_w,
                                        static_cast<int>(mrope_pos_t_.size()),
                                        mrope_text_pos_delta_,
                                        rope_.mrope_section[0], rope_.mrope_section[1],
                                        rope_.mrope_section[2]);
            });

            time_stage(ProfileStage::KvCache, [&] {
                ops::copy_to_cache(ctx, buf_.k, kv_cache_.k_cache(i), seq_len, start_pos,
                                   cfg_.num_key_value_heads, cfg_.head_dim, kv_cache_.max_length());
                ops::copy_to_cache(ctx, buf_.v, kv_cache_.v_cache(i), seq_len, start_pos,
                                   cfg_.num_key_value_heads, cfg_.head_dim, kv_cache_.max_length());
            });
        }

        // Attention
        time_stage(ProfileStage::Attention, [&] {
            if (seq_len == 1) {
                ops::attention_decode(ctx, *q_for_attn, kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                      buf_.attn_out, buf_.decode_scores,
                                      cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                      cfg_.head_dim, cached_len,
                                      &buf_.decode_attn_partials);
            } else if (start_pos == 0) {
                ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v,
                                       buf_.attn_out, buf_.scores, seq_len,
                                       cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                       cfg_.head_dim);
            } else {
                ops::attention_prefill_cached(ctx, buf_.q,
                                              kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                              buf_.attn_out, buf_.incr_scores,
                                              seq_len, start_pos,
                                              cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                              cfg_.head_dim, kv_cache_.max_length());
            }
        });

        // O projection
        time_stage(ProfileStage::OProj, [&] {
            layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.gate, seq_len, hidden_ptr_for_residual);
        });

        // Residual + post-attn norm
        time_stage(ProfileStage::PostAttnNorm, [&] {
            if (hidden_ptr_for_residual) {
                ops::rms_norm(ctx, buf_.gate, *layer.post_attn_ln_weight,
                              cfg_.rms_norm_eps, buf_.normed, seq_len, H);
            } else {
                ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                        *layer.post_attn_ln_weight, cfg_.rms_norm_eps,
                                        buf_.normed, seq_len, H);
            }
        });

        // FFN
        if (seq_len == 1) {
            bool fused_ok = false;
            time_stage(ProfileStage::FfnProj, [&] {
                fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                    ctx, layer.gate_up_proj, buf_.normed, buf_.gate, FF);
            });
            if (!fused_ok) {
                time_stage(ProfileStage::FfnProj, [&] {
                    layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, 1);
                });
                time_stage(ProfileStage::FfnAct, [&] {
                    ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, FF);
                });
            }
        } else {
            time_stage(ProfileStage::FfnProj, [&] {
                layer.gate_up_proj.forward(ctx, linear_scratch_, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::split_gate_up(ctx, buf_.gate_up, buf_.gate, buf_.up, seq_len, FF);
                ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
            });
        }

        // Down projection
        time_stage(ProfileStage::DownProj, [&] {
            layer.down_proj.forward(ctx, linear_scratch_, buf_.gate, buf_.up, seq_len, hidden_ptr_for_residual);
        });

        // Residual + next-norm
        time_stage(ProfileStage::PostMlpNorm, [&] {
            if (hidden_ptr_for_residual) {
                if (i < cfg_.num_hidden_layers - 1) {
                    ops::rms_norm(ctx, buf_.up, *layers_[i + 1].input_ln_weight,
                                  cfg_.rms_norm_eps, buf_.normed, seq_len, H);
                } else {
                    ops::rms_norm(ctx, buf_.up, *final_norm_weight_,
                                  cfg_.rms_norm_eps, buf_.normed, seq_len, H);
                }
            } else {
                if (i < cfg_.num_hidden_layers - 1) {
                    ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                            *layers_[i + 1].input_ln_weight, cfg_.rms_norm_eps,
                                            buf_.normed, seq_len, H);
                } else {
                    ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                            *final_norm_weight_, cfg_.rms_norm_eps,
                                            buf_.normed, seq_len, H);
                }
            }
        });
    }

    // LM Head
    time_stage(ProfileStage::LmHead, [&] {
        if (seq_len > 1) {
            bf16* last_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
            Tensor last = Tensor::view(ctx, last_ptr, {1, (int64_t)H});
            lm_head_.forward(ctx, last, buf_.logits, 1);
        } else {
            lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
        }
    });

    kv_cache_.advance(seq_len);
    current_len_ += seq_len;

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
            double total_ms = 0.0;
            for (double v : totals.stage_ms) total_ms += v;
            const char* tag = profile_decode ? "[Q3ASRDecodeProfile]" : "[Q3ASRPrefillProfile]";
            AILA_LOG_INFO(
                "%s tokens=%d total=%.3f embed=%.3f full_qkv=%.3f full_split=%.3f qk_rope=%.3f "
                "kv_copy=%.3f attn=%.3f o_proj=%.3f post_attn=%.3f ffn_proj=%.3f ffn_act=%.3f "
                "down=%.3f post_mlp=%.3f lm_head=%.3f",
                tag,
                totals.tokens,
                total_ms / static_cast<double>(std::max(1, totals.tokens)),
                avg(ProfileStage::EmbedNorm),
                avg(ProfileStage::FullQkvProj),
                avg(ProfileStage::FullSplit),
                avg(ProfileStage::QkNormRope),
                avg(ProfileStage::KvCache),
                avg(ProfileStage::Attention),
                avg(ProfileStage::OProj),
                avg(ProfileStage::PostAttnNorm),
                avg(ProfileStage::FfnProj),
                avg(ProfileStage::FfnAct),
                avg(ProfileStage::DownProj),
                avg(ProfileStage::PostMlpNorm),
                avg(ProfileStage::LmHead));
            totals.reset();
        }
    }

    return buf_.logits;
}

void Qwen3ASRBnb4Backend::reset() {
    kv_cache_.reset();
    current_len_ = 0;
    has_embedding_overrides_ = false;
}

bool Qwen3ASRBnb4Backend::truncate_kv_cache(int new_len) {
    kv_cache_.truncate(new_len);
    current_len_ = new_len;
    return true;
}
