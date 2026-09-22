#include "Qwen3Bnb4Backend.hpp"
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

struct StageProfiler {
    Context& context;
    std::array<double, static_cast<size_t>(ProfileStage::Count)>& stageMs;
    bool enabled;
    bool hostOnly;

    template <typename Fn>
    void operator()(ProfileStage stage, Fn&& fn) const {
        if (!enabled) {
            fn();
            return;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        if (!hostOnly) context.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        stageMs[static_cast<size_t>(stage)] +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};
}

struct Qwen3Bnb4Backend::LayerExecution {
    Context& context;
    Buffers& buffers;
    Bnb4BitLinearScratch& linearScratch;
    ExecutionPhase phase;
    int startPosition;
    int tokenCount;
    int cachedLength;
    Tensor& keyCache;
    Tensor& valueCache;
    int cacheCapacity;
    bf16* residualHidden;
    const StageProfiler& profiler;
};

Qwen3Bnb4Backend::Session::Session(Context& ctx, const Qwen3Config& config,
                                  int capacity,
                                  std::shared_ptr<const SessionOwner> owner)
    : context_(&ctx), owner_(std::move(owner)) {
    cache_.init(ctx, config, capacity);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        cacheBytes_ += cache_.k_cache(i).size_bytes();
        cacheBytes_ += cache_.v_cache(i).size_bytes();
    }
}

Qwen3Bnb4Backend::~Qwen3Bnb4Backend() = default;

std::unique_ptr<Qwen3Bnb4Backend::Session>
Qwen3Bnb4Backend::CreateSessionImpl(Context& ctx) {
    if (!sessionOwner_) {
        throw std::logic_error("Qwen3Bnb4Backend::CreateSession: model is not loaded");
    }
    if (&ctx != context_) {
        throw std::invalid_argument("Qwen3Bnb4Backend::CreateSession: context does not match the loaded model");
    }
    return std::unique_ptr<Session>(new Session(ctx, cfg_, max_seq_len_, sessionOwner_));
}

std::unique_ptr<Qwen3Bnb4Backend::Session>
Qwen3Bnb4Backend::CreateSession(Context& ctx) {
    // Existing FP8 decode kernels are not valid for this shared-session path:
    // the JM reader assumes BF16 and the fused writer omits half of head128.
    if (aila::env::read_flag("AILA_KV_QUANT", false)) {
        throw std::invalid_argument(
            "Qwen3 shared sessions require BF16 KV; set AILA_KV_QUANT=0");
    }
    auto session = CreateSessionImpl(ctx);
    // Once published, existing cache state must not be mixed with replacement
    // weights or another adapter, even if every explicit Session is released.
    explicitSessionsCreated_ = true;
    return session;
}

void Qwen3Bnb4Backend::ValidateSession(const Session& session) const {
    if (!sessionOwner_) {
        throw std::logic_error("Qwen3Bnb4Backend: model is not loaded");
    }
    if (session.owner_ != sessionOwner_ || session.context_ != context_) {
        throw std::invalid_argument("Qwen3Bnb4Backend: session does not belong to this loaded model");
    }
}

bool Qwen3Bnb4Backend::apply_lora(Context& ctx, const aila::lora::LoraAdapter& adapter,
                                  std::string* error_message) {
    if (explicitSessionsCreated_) {
        if (error_message) {
            *error_message = "Qwen3Bnb4Backend: cannot change LoRA after explicit sessions have been created";
        }
        return false;
    }
    if (!sessionOwner_ || &ctx != context_) {
        if (error_message) {
            *error_message = "Qwen3Bnb4Backend: LoRA requires a loaded model and its original context";
        }
        return false;
    }

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

    // Collect attachments per layer, per fused projection
    int num_layers = cfg_.num_hidden_layers;
    std::vector<std::vector<LoraAttachment>> qkv_attachments(num_layers);
    std::vector<std::vector<LoraAttachment>> gate_up_attachments(num_layers);

    for (const auto& pair : adapter.pairs) {
        // Parse "model.layers.N.self_attn.TARGET.weight" or "model.layers.N.mlp.TARGET.weight"
        const std::string prefix = "model.layers.";
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

    // Set fused attachments on qkv_proj and gate_up_proj
    for (int i = 0; i < num_layers; ++i) {
        if (!qkv_attachments[i].empty()) {
            layers_[i].qkv_proj.set_lora(std::move(qkv_attachments[i]));
        }
        if (!gate_up_attachments[i].empty()) {
            layers_[i].gate_up_proj.set_lora(std::move(gate_up_attachments[i]));
        }
    }

    AILA_LOG_INFO("[Qwen3Bnb4] LoRA adapter applied: r=%d alpha=%d scaling=%.2f",
                  adapter.config.r, adapter.config.lora_alpha, adapter.config.scaling);
    if (error_message) *error_message = "";
    return true;
}

void Qwen3Bnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;

    // Another Session may have just submitted work using these allocations.
    // Tensor replacement frees USM immediately, so growth must drain first.
    if (buf_.hidden.valid()) ctx.synchronize();

    int H = hidden_size_;
    // Decode batches contain independent rows: B may exceed a very small
    // per-Session context capacity. Existing single-sequence calls have
    // already validated seq_len <= max_seq_len_, so their capacity is intact.
    int new_cap = std::max(seq_len, std::min(max_seq_len_, round_up_seq(seq_len, 64)));

    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.qkv      = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)fused_qkv_dim_});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)kv_dim_});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)q_dim_});
    buf_.gate_up  = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * ff_dim_)});
    // These buffers also hold attention/down projections of hidden width.
    // Their producers and consumers use the logical projection row stride.
    const int projection_capacity = std::max(ff_dim_, H);
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)projection_capacity});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)projection_capacity});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen3Bnb4Backend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    if (buf_.scores.valid()) ctx.synchronize();

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);

    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen3Bnb4Backend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;

    if (buf_.incr_scores.valid()) ctx.synchronize();

    int new_seq_cap = std::max(seq_len, incr_prefill_seq_cap_);
    int new_total_cap = std::max(total_len, incr_prefill_total_cap_);
    new_seq_cap = round_up_seq(new_seq_cap, 16);
    new_total_cap = round_up_seq(new_total_cap, 64);

    buf_.incr_scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_seq_cap, (int64_t)new_total_cap},
        dnnl::memory::data_type::f32);

    incr_prefill_seq_cap_ = new_seq_cap;
    incr_prefill_total_cap_ = new_total_cap;
    AILA_LOG_INFO("[Qwen3Bnb4] Incremental prefill score buffer resized: seq_cap=%d, total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

bool Qwen3Bnb4Backend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                             int max_seq_len, std::string* error_message) {
    if (explicitSessionsCreated_) {
        if (error_message) {
            *error_message = "Qwen3Bnb4Backend: cannot reload a model after explicit sessions have been created";
        }
        return false;
    }
    if (spec.family != ModelFamily::Qwen3Dense) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: invalid model family";
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: model is not a bitsandbytes 4-bit checkpoint";
        return false;
    }
    if (max_seq_len <= 0) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: max_seq_len must be positive";
        return false;
    }
    if (spec.qwen3.num_hidden_layers <= 0) {
        if (error_message) *error_message = "Qwen3Bnb4Backend: model must contain at least one decoder layer";
        return false;
    }

    // Reload is still supported for the legacy facade before any explicit
    // Session is published. Discard every binding and allocation from the old
    // execution slot only after its queue has finished.
    if (context_) context_->synchronize();
    defaultSession_.reset();
    sessionOwner_.reset();
    decoderPlan_.reset();
    layers_.clear();
    fused_weights_.clear();
    linear_scratch_ = Bnb4BitLinearScratch();
    lm_head_ = Linear();
    decodeBatchLogitsView_ = Tensor();
    buf_ = Buffers();
    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    incr_prefill_seq_cap_ = 0;
    incr_prefill_total_cap_ = 0;
    embed_weight_ = nullptr;
    final_norm_weight_ = nullptr;
    context_ = &ctx;

    cfg_ = spec.qwen3;
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

    // --- Embedding ---
    embed_weight_ = &weights.get("model.embed_tokens.weight");
    AILA_LOG_INFO("[Qwen3Bnb4] embed_tokens loaded");

    // --- Layers ---
    layers_.resize(cfg_.num_hidden_layers);
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 2 + 1);

    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        std::vector<std::string> weights_to_erase;

        // Qwen3 uses direct RMSNorm weights (not plus-one like Qwen3.5).
        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

        // Fuse Q/K/V into a single Bnb4BitLinear
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

        // Fuse gate/up and init down
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

        // Erase individual BNB4 tensors after fusion (keep quant_map — fused linears still reference it).
        if (!weights_to_erase.empty()) {
            ctx.synchronize();
            for (const auto& name : weights_to_erase) {
                erase_bnb_weight(name);
            }
        }
    }
    AILA_LOG_INFO("[Qwen3Bnb4] %d transformer layers loaded", cfg_.num_hidden_layers);

    // --- Final norm ---
    final_norm_weight_ = &weights.get("model.norm.weight");

    // --- LM Head (always dense, even in quantized mode) ---
    if (weights.has("lm_head.weight") &&
        !weights.has("lm_head.weight.quant_state.bitsandbytes__nf4") &&
        weights.get("lm_head.weight").dtype() != dnnl::memory::data_type::u8) {
        Tensor* lm_w = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lm_w, hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Bnb4] lm_head loaded (standalone dense)");
    } else {
        // Tied: transpose copy of embed_weight_
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.put("lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("lm_head.weight_preprocessed"), hidden_size_, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Bnb4] lm_head (tied, preprocessed copy)");
    }

    // --- Runtime buffers (lazy grow) ---
    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    ensure_runtime_buffers(ctx, 1);

    // --- Persistent buffers ---
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
        // Per-sequence partials block: the decode batch bound is four rows, so
        // up to four independent sequences can merge in one launch and each row
        // reads only its own statistics.
        buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {4, (int64_t)cfg_.num_attention_heads, (int64_t)max_tiles,
             (int64_t)partial_stride},
            dnnl::memory::data_type::f32);
    }

    BuildDecoderPlan();
    sessionOwner_ = std::make_shared<const SessionOwner>();
    AILA_LOG_INFO("[Qwen3Bnb4] Model fully loaded and initialized");
    return true;
}

void Qwen3Bnb4Backend::BuildDecoderPlan() {
    auto plan = std::make_unique<DecoderPlan>();
    plan->initialNormWeight = layers_.front().input_ln_weight;
    plan->layers.reserve(layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        // Preserve the existing cross-layer residual + input-norm fusion.
        // layers_ is never resized after binding its fused linear weights.
        Tensor* postMlpNorm = i + 1 < layers_.size()
            ? layers_[i + 1].input_ln_weight : final_norm_weight_;
        plan->layers.push_back({&layers_[i], postMlpNorm, static_cast<int>(i)});
    }
    decoderPlan_ = std::move(plan);
}

void Qwen3Bnb4Backend::ExecuteLayer(const LayerPlan& plan,
                                    const LayerExecution& execution) {
    auto& ctx = execution.context;
    auto& buf = execution.buffers;
    auto& linearScratch = execution.linearScratch;
    auto& layer = *plan.layer;
    const auto& time_stage = execution.profiler;
    const int seq_len = execution.tokenCount;
    const int start_pos = execution.startPosition;
    const bool decode = execution.phase == ExecutionPhase::Decode;
    bf16* hidden_ptr_for_residual = execution.residualHidden;
    const int H = hidden_size_;
    const int QD = q_dim_;
    const int KVD = kv_dim_;
    const int FF = ff_dim_;
    Tensor q_decode_view;
    Tensor* q_for_attn = &buf.q;

    // Entry: hidden carries the residual stream; normed is this layer's
    // normalized input. Exit: normed is prepared for the next layer / LM head.
    if (decode) {
        // Decode path: fused QKV projection via Bnb4BitLinear.
        time_stage(ProfileStage::FullQkvProj, [&] {
            layer.qkv_proj.forward(ctx, linearScratch, buf.normed, buf.qkv, 1);
        });
        bf16* qkv_ptr = static_cast<bf16*>(buf.qkv.data());
        q_decode_view = Tensor::view(ctx, qkv_ptr, {1, (int64_t)QD});
        Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, (int64_t)KVD});
        Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, (int64_t)KVD});
        q_for_attn = &q_decode_view;

        time_stage(ProfileStage::QkNormRope, [&] {
            ops::decode_prepare_qkv(ctx, q_decode_view, k_view, v_view, buf.rope_freq,
                                    *layer.q_norm_weight, *layer.k_norm_weight,
                                    execution.keyCache, execution.valueCache,
                                    start_pos,
                                    cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                    cfg_.head_dim, cfg_.rms_norm_eps, cfg_.rope_theta);
        });
    } else {
        time_stage(ProfileStage::FullQkvProj, [&] {
            layer.qkv_proj.forward(ctx, linearScratch, buf.normed, buf.qkv, seq_len);
        });
        time_stage(ProfileStage::FullSplit, [&] {
            ops::split_qkv(ctx, buf.qkv, buf.q, buf.k, buf.v, seq_len, QD, KVD);
        });

        time_stage(ProfileStage::QkNormRope, [&] {
            ops::head_rms_norm(ctx, buf.q, *layer.q_norm_weight,
                               cfg_.rms_norm_eps, seq_len,
                               cfg_.num_attention_heads, cfg_.head_dim);
            ops::head_rms_norm(ctx, buf.k, *layer.k_norm_weight,
                               cfg_.rms_norm_eps, seq_len,
                               cfg_.num_key_value_heads, cfg_.head_dim);

            ops::apply_rope(ctx, buf.q, buf.k, seq_len, start_pos,
                            cfg_.num_attention_heads, cfg_.num_key_value_heads,
                            cfg_.head_dim, cfg_.rope_theta);
        });

        time_stage(ProfileStage::KvCache, [&] {
            ops::copy_to_cache(ctx, buf.k, execution.keyCache,
                               seq_len, start_pos,
                               cfg_.num_key_value_heads, cfg_.head_dim,
                               execution.cacheCapacity);
            ops::copy_to_cache(ctx, buf.v, execution.valueCache,
                               seq_len, start_pos,
                               cfg_.num_key_value_heads, cfg_.head_dim,
                               execution.cacheCapacity);
        });
    }

    time_stage(ProfileStage::Attention, [&] {
        if (decode) {
            ops::attention_decode(ctx, *q_for_attn, execution.keyCache, execution.valueCache,
                                  buf.attn_out, buf.decode_scores,
                                  cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                  cfg_.head_dim, execution.cachedLength,
                                  &buf.decode_attn_partials);
        } else if (start_pos == 0) {
            ops::attention_prefill(ctx, buf.q, buf.k, buf.v,
                                   buf.attn_out, buf.scores,
                                   seq_len,
                                   cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                   cfg_.head_dim);
        } else {
            ops::attention_prefill_cached(ctx, buf.q,
                                          execution.keyCache, execution.valueCache,
                                          buf.attn_out, buf.incr_scores,
                                          seq_len, start_pos,
                                          cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                          cfg_.head_dim, execution.cacheCapacity);
        }
    });

    time_stage(ProfileStage::OProj, [&] {
        layer.o_proj.forward(ctx, linearScratch, buf.attn_out, buf.gate, seq_len, hidden_ptr_for_residual);
    });

    time_stage(ProfileStage::PostAttnNorm, [&] {
        if (hidden_ptr_for_residual) {
            ops::rms_norm(ctx, buf.gate, *layer.post_attn_ln_weight,
                          cfg_.rms_norm_eps, buf.normed, seq_len, H);
        } else {
            ops::fused_add_rms_norm(ctx, buf.hidden, buf.gate,
                                    *layer.post_attn_ln_weight, cfg_.rms_norm_eps,
                                    buf.normed, seq_len, H);
        }
    });

    if (decode) {
        bool fused_ok = false;
        time_stage(ProfileStage::FfnProj, [&] {
            fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.gate_up_proj, buf.normed, buf.gate, FF);
        });
        if (!fused_ok) {
            time_stage(ProfileStage::FfnProj, [&] {
                layer.gate_up_proj.forward(ctx, linearScratch, buf.normed, buf.gate_up, 1);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::fused_gate_up_swiglu(ctx, buf.gate_up, buf.gate, FF);
            });
        }
    } else {
        time_stage(ProfileStage::FfnProj, [&] {
            layer.gate_up_proj.forward(ctx, linearScratch, buf.normed, buf.gate_up, seq_len);
        });
        time_stage(ProfileStage::FfnAct, [&] {
            ops::split_gate_up(ctx, buf.gate_up, buf.gate, buf.up, seq_len, FF);
            ops::swiglu(ctx, buf.gate, buf.up, buf.gate, seq_len * FF);
        });
    }

    time_stage(ProfileStage::DownProj, [&] {
        layer.down_proj.forward(ctx, linearScratch, buf.gate, buf.up, seq_len, hidden_ptr_for_residual);
    });

    time_stage(ProfileStage::PostMlpNorm, [&] {
        if (hidden_ptr_for_residual) {
            ops::rms_norm(ctx, buf.up, *plan.postMlpNormWeight,
                          cfg_.rms_norm_eps, buf.normed, seq_len, H);
        } else {
            ops::fused_add_rms_norm(ctx, buf.hidden, buf.up,
                                    *plan.postMlpNormWeight, cfg_.rms_norm_eps,
                                    buf.normed, seq_len, H);
        }
    });
}

Tensor& Qwen3Bnb4Backend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0 || !token_ids_device) {
        throw std::invalid_argument("Qwen3Bnb4Backend::forward: token IDs must be non-null and seq_len positive");
    }
    if (!defaultSession_) defaultSession_ = CreateSessionImpl(ctx);
    return Forward(ctx, *defaultSession_, token_ids_device, seq_len);
}

bool Qwen3Bnb4Backend::SupportsDecodeBatch() const noexcept {
    static const bool profile = aila::env::read_flag("AILA_PROFILE_Q3_DECODE", false);
    static const bool fusedResidual = aila::env::read_flag("AILA_FUSE_RESIDUAL_ADD", false);
    return decoderPlan_ && !profile && !fusedResidual;
}

Tensor& Qwen3Bnb4Backend::ForwardDecodeBatch(
    Context& ctx, const std::vector<Session*>& sessions, const int* deviceTokenIds) {
    if (sessions.empty() || sessions.size() > 4) {
        throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: batch size must be 1..4");
    }
    if (!deviceTokenIds) {
        throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: token IDs must be non-null");
    }
    if (&ctx != context_) {
        throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: context does not match the loaded model");
    }

    const int batchSize = static_cast<int>(sessions.size());
    std::array<int, 4> positions{};
    for (int row = 0; row < batchSize; ++row) {
        if (!sessions[row]) {
            throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: Session must be non-null");
        }
        ValidateSession(*sessions[row]);
        for (int earlier = 0; earlier < row; ++earlier) {
            if (sessions[earlier] == sessions[row]) {
                throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: duplicate Session");
            }
        }
        positions[row] = sessions[row]->CurrentLength();
        if (positions[row] >= sessions[row]->Capacity()) {
            throw std::invalid_argument("Qwen3Bnb4Backend::ForwardDecodeBatch: context window exceeded");
        }
    }
    if (batchSize == 1) {
        return Forward(ctx, *sessions.front(), deviceTokenIds, 1);
    }
    if (!SupportsDecodeBatch()) {
        throw std::invalid_argument("Qwen3 decode profiling/residual configuration requires the original single-sequence path");
    }

    // Reserve the row workspace before touching any Session's KV. These
    // tensors contain dense logical rows even when their allocation's second
    // dimension is larger than the width of the current projection.
    ensure_runtime_buffers(ctx, batchSize);
    if (!buf_.decode_batch_logits.valid()) {
        buf_.decode_batch_logits = Tensor::allocate(
            ctx, {4, static_cast<int64_t>(cfg_.vocab_size)});
    }
    decodeBatchLogitsView_ = Tensor::view(
        ctx, buf_.decode_batch_logits.data(),
        {batchSize, static_cast<int64_t>(cfg_.vocab_size)}, buf_.decode_batch_logits.dtype());

    auto row_view = [&](Tensor& tensor, int row, int width) {
        auto* pointer = tensor.data_as<bf16>() + static_cast<size_t>(row) * width;
        return Tensor::view(ctx, pointer, {1, static_cast<int64_t>(width)}, tensor.dtype());
    };
    auto normalize_rows = [&](Tensor& input, Tensor& weight, Tensor& output) {
        for (int r = 0; r < batchSize; ++r) {
            auto source = row_view(input, r, hidden_size_);
            auto target = row_view(output, r, hidden_size_);
            ops::rms_norm(ctx, source, weight, cfg_.rms_norm_eps, target, 1, hidden_size_);
        }
    };
    auto add_normalize_rows = [&](Tensor& input, Tensor& residual, Tensor& weight, Tensor& output) {
        for (int r = 0; r < batchSize; ++r) {
            auto source = row_view(input, r, hidden_size_);
            auto added = row_view(residual, r, hidden_size_);
            auto target = row_view(output, r, hidden_size_);
            ops::fused_add_rms_norm(ctx, source, added, weight, cfg_.rms_norm_eps, target, 1, hidden_size_);
        }
    };

    ops::embedding_lookup(ctx, *embed_weight_, deviceTokenIds, batchSize, buf_.hidden, hidden_size_);
    normalize_rows(buf_.hidden, *decoderPlan_->initialNormWeight, buf_.normed);

    for (const auto& plan : decoderPlan_->layers) {
        auto& layer = *plan.layer;
        layer.qkv_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.normed, buf_.qkv, batchSize);
        ops::DecodeRowGeometry attentionRows[4];
        int maxAttentionLen = 0;
        for (int row = 0; row < batchSize; ++row) {
            auto& cache = sessions[row]->cache_;
            auto* qkv = buf_.qkv.data_as<bf16>() + static_cast<size_t>(row) * fused_qkv_dim_;
            auto q = Tensor::view(ctx, qkv, {1, static_cast<int64_t>(q_dim_)});
            auto k = Tensor::view(ctx, qkv + q_dim_, {1, static_cast<int64_t>(kv_dim_)});
            auto v = Tensor::view(ctx, qkv + q_dim_ + kv_dim_, {1, static_cast<int64_t>(kv_dim_)});
            auto& keyCache = cache.k_cache(plan.cacheIndex);
            auto& valueCache = cache.v_cache(plan.cacheIndex);
            ops::decode_prepare_qkv(ctx, q, k, v, buf_.rope_freq,
                                    *layer.q_norm_weight, *layer.k_norm_weight,
                                    keyCache, valueCache, positions[row],
                                    cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                    cfg_.head_dim, cfg_.rms_norm_eps, cfg_.rope_theta);
            attentionRows[row].k_cache = &keyCache;
            attentionRows[row].v_cache = &valueCache;
            attentionRows[row].cached_len = positions[row] + 1;
            maxAttentionLen = std::max(maxAttentionLen, positions[row] + 1);
        }
        // One multi-row launch for every independent sequence. Each row keeps
        // the single-row kernel body, so a row's output stays bitwise identical
        // to the per-row attention_decode it replaces; a head_dim 128 row below
        // the exact-merge threshold still needs the joint-matrix kernel and
        // falls back per row inside attention_decode_rows.
        ops::attention_decode_rows(ctx, buf_.qkv, fused_qkv_dim_, buf_.attn_out,
                                   buf_.decode_scores, attentionRows, batchSize,
                                   cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                   cfg_.head_dim, maxAttentionLen,
                                   &buf_.decode_attn_partials);

        // Residual fusion is a single-row GEMV option. Keep the existing
        // explicit residual/norm semantics for multi-row projections.
        layer.o_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.attn_out, buf_.gate, batchSize);
        add_normalize_rows(buf_.hidden, buf_.gate, *layer.post_attn_ln_weight, buf_.normed);
        // Preserve decode's gate/up weight and activation rounding boundaries.
        // The fused kernel accumulates unrounded dequantized weights and applies
        // SwiGLU before BF16 output; a multi-row projection changes both steps.
        for (int row = 0; row < batchSize; ++row) {
            auto normalized = row_view(buf_.normed, row, hidden_size_);
            auto gate = row_view(buf_.gate, row, ff_dim_);
            if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                    ctx, layer.gate_up_proj, normalized, gate, ff_dim_)) {
                auto gateUp = row_view(buf_.gate_up, row, 2 * ff_dim_);
                layer.gate_up_proj.forward(ctx, linear_scratch_, normalized, gateUp, 1);
                ops::fused_gate_up_swiglu(ctx, gateUp, gate, ff_dim_);
            }
        }
        layer.down_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.gate, buf_.up, batchSize);
        add_normalize_rows(buf_.hidden, buf_.up, *plan.postMlpNormWeight, buf_.normed);
    }

    // Keep the single-row LM head's reduction and vocabulary layout. Sampling
    // is owned by each Request and consumes its row before this slot is reused.
    for (int row = 0; row < batchSize; ++row) {
        auto normalized = row_view(buf_.normed, row, hidden_size_);
        auto logits = row_view(buf_.decode_batch_logits, row, cfg_.vocab_size);
        lm_head_.forward(ctx, normalized, logits, 1);
    }
    // Publish logical lengths only after every row's complete forward has
    // been submitted. The caller drains/handles asynchronous device failures.
    for (auto* session : sessions) session->cache_.advance(1);
    return decodeBatchLogitsView_;
}

Tensor& Qwen3Bnb4Backend::Forward(Context& ctx, Session& session,
                                const int* token_ids_device, int seq_len) {
    ValidateSession(session);
    if (&ctx != context_) {
        throw std::invalid_argument("Qwen3Bnb4Backend::Forward: context does not match the loaded model");
    }
    if (seq_len <= 0) {
        throw std::invalid_argument("Qwen3Bnb4Backend::Forward: seq_len must be positive");
    }
    if (!token_ids_device) {
        throw std::invalid_argument("Qwen3Bnb4Backend::Forward: token IDs must be non-null");
    }

    auto& kvCache = session.cache_;
    const int start_pos = kvCache.current_length();
    // Subtract before adding to reject oversized counts without signed overflow.
    if (seq_len > session.Capacity() - start_pos) {
        throw std::invalid_argument("Qwen3Bnb4Backend::Forward: context window exceeded (start_pos=" +
                                    std::to_string(start_pos) + ", seq_len=" +
                                    std::to_string(seq_len) + ", capacity=" +
                                    std::to_string(session.Capacity()) + ")");
    }
    const int cached_len = start_pos + seq_len;

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
    const StageProfiler time_stage{ctx, stage_ms, profile_enabled, profile_host_only};

    int H = hidden_size_;
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

        // 2. Initial norm for the first layer
        ops::rms_norm(ctx, buf_.hidden, *decoderPlan_->initialNormWeight,
                      cfg_.rms_norm_eps, buf_.normed, seq_len, H);
    });

    // A single-token prompt retains the existing decode kernel fast path.
    const auto phase = seq_len == 1 ? ExecutionPhase::Decode : ExecutionPhase::Prefill;
    for (const auto& layerPlan : decoderPlan_->layers) {
        const LayerExecution execution{
            ctx, buf_, linear_scratch_, phase, start_pos, seq_len, cached_len,
            kvCache.k_cache(layerPlan.cacheIndex), kvCache.v_cache(layerPlan.cacheIndex),
            kvCache.max_length(), hidden_ptr_for_residual, time_stage
        };
        ExecuteLayer(layerPlan, execution);
    }

    time_stage(ProfileStage::LmHead, [&] {
        // LM Head: logits for the last token only
        if (seq_len > 1) {
            bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
            Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)H});
            lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
        } else {
            lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
        }
    });

    kvCache.advance(seq_len);

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
            const char* tag = profile_decode ? "[Q3DecodeProfile]" : "[Q3PrefillProfile]";
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

void Qwen3Bnb4Backend::reset() {
    if (defaultSession_) Reset(*defaultSession_);
}

bool Qwen3Bnb4Backend::truncate_kv_cache(int new_len) {
    if (new_len < 0) return false;
    // Keep the legacy no-op behavior when asked to truncate beyond the end.
    if (defaultSession_ && new_len < defaultSession_->CurrentLength()) {
        Truncate(*defaultSession_, new_len);
    }
    return true;
}

void Qwen3Bnb4Backend::Reset(Session& session) {
    ValidateSession(session);
    session.cache_.reset();
}

int Qwen3Bnb4Backend::Truncate(Session& session, int newLength) {
    ValidateSession(session);
    if (newLength < 0 || newLength > session.CurrentLength()) {
        throw std::invalid_argument("Qwen3Bnb4Backend::Truncate: length must be between zero and the current length");
    }
    session.cache_.truncate(newLength);
    return session.CurrentLength();
}
