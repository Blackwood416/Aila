#include "Qwen3DenseBackend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
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

void Qwen3DenseBackend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;

    int H = cfg_.hidden_size;
    int QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim;
    int FF = cfg_.intermediate_size;
    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));

    buf_.hidden   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.normed   = Tensor::allocate(ctx, {(int64_t)new_cap, H});
    buf_.qkv      = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(QD + 2 * KVD)});
    buf_.q        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)QD});
    buf_.k        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)KVD});
    buf_.v        = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)KVD});
    buf_.attn_out = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)QD});
    buf_.gate_up  = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)(2 * FF)});
    buf_.gate     = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)FF});
    buf_.up       = Tensor::allocate(ctx, {(int64_t)new_cap, (int64_t)FF});

    runtime_seq_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Dense] Runtime buffers resized: seq_cap=%d", runtime_seq_capacity_);
}

void Qwen3DenseBackend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;

    int new_cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(ctx,
        {(int64_t)cfg_.num_attention_heads, (int64_t)new_cap, (int64_t)new_cap},
        dnnl::memory::data_type::f32);

    prefill_scores_capacity_ = new_cap;
    AILA_LOG_INFO("[Qwen3Dense] Prefill score buffer resized: seq_cap=%d", prefill_scores_capacity_);
}

void Qwen3DenseBackend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
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
    AILA_LOG_INFO("[Qwen3Dense] Incremental prefill score buffer resized: seq_cap=%d, total_cap=%d",
                  incr_prefill_seq_cap_, incr_prefill_total_cap_);
}

bool Qwen3DenseBackend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                              int max_seq_len, std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3Dense) {
        if (error_message) *error_message = "Qwen3DenseBackend: invalid model family";
        return false;
    }

    cfg_ = spec.qwen3;
    max_seq_len_ = max_seq_len;

    int H = cfg_.hidden_size;
    int QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim;
    int FF = cfg_.intermediate_size;
    fused_weights_.clear();
    fused_weights_.reserve(static_cast<size_t>(cfg_.num_hidden_layers) * 2);

    auto transpose_weight = [&](const std::string& name) {
        Tensor& src = weights.get(name);
        int64_t out_f = src.shape(0);
        int64_t in_f = src.shape(1);

        Tensor dst = Tensor::allocate(ctx, {in_f, out_f}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();

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
    AILA_LOG_INFO("[Qwen3Dense] embed_tokens loaded");

    // --- Layers ---
    layers_.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.input_ln_weight = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln_weight = &weights.get(prefix + "post_attention_layernorm.weight");

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
    AILA_LOG_INFO("[Qwen3Dense] %d transformer layers loaded", cfg_.num_hidden_layers);

    // --- Final norm ---
    final_norm_weight_ = &weights.get("model.norm.weight");

    // --- LM Head ---
    if (weights.has("lm_head.weight")) {
        Tensor* lw = transpose_weight("lm_head.weight");
        lm_head_.init(ctx, *lw, H, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Dense] lm_head loaded (standalone)");
    } else {
        Tensor& src = *embed_weight_;
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        weights.put("lm_head.weight_preprocessed", std::move(dst));
        lm_head_.init(ctx, weights.get("lm_head.weight_preprocessed"), H, cfg_.vocab_size, true);
        AILA_LOG_INFO("[Qwen3Dense] lm_head (tied, preprocessed copy)");
    }

    // --- KV Cache ---
    kv_cache_.init(ctx, cfg_, max_seq_len);

    // --- Runtime buffers (lazy grow) ---
    runtime_seq_capacity_ = 0;
    prefill_scores_capacity_ = 0;
    ensure_runtime_buffers(ctx, 1);

    // --- Persistent buffers ---
    buf_.logits   = Tensor::allocate(ctx, {1, (int64_t)cfg_.vocab_size});
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

    AILA_LOG_INFO("[Qwen3Dense] Model fully loaded and initialized");
    return true;
}

Tensor& Qwen3DenseBackend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("Qwen3DenseBackend::forward: seq_len must be positive");
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

    int H = cfg_.hidden_size;
    int QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim;
    int FF = cfg_.intermediate_size;
    int start_pos = kv_cache_.current_length();
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen3DenseBackend::forward: context window exceeded (cached_len=" +
                                 std::to_string(cached_len) + ", max_seq_len=" +
                                 std::to_string(max_seq_len_) + ")");
    }

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) {
            ensure_prefill_scores(ctx, seq_len);
        } else {
            ensure_incr_prefill_scores(ctx, seq_len, start_pos + seq_len);
        }
    }

    time_stage(ProfileStage::EmbedNorm, [&] {
        // 1. Embedding lookup
        ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

        // 2. Initial norm for the first layer
        ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                      cfg_.rms_norm_eps, buf_.normed, seq_len, H);
    });

    // 3. Iterate through transformer layers
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& layer = layers_[i];
        Tensor q_decode_view;
        Tensor* q_for_attn = &buf_.q;

        if (seq_len == 1) {
            // Decode path: fused QKV projection
            time_stage(ProfileStage::FullQkvProj, [&] {
                layer.qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
            });
            bf16* qkv_ptr = static_cast<bf16*>(buf_.qkv.data());
            q_decode_view = Tensor::view(ctx, qkv_ptr, {1, (int64_t)QD});
            Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, (int64_t)KVD});
            Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, (int64_t)KVD});
            q_for_attn = &q_decode_view;

            time_stage(ProfileStage::QkNormRope, [&] {
                ops::decode_prepare_qkv(ctx, q_decode_view, k_view, v_view, buf_.rope_freq,
                                        *layer.q_norm_weight, *layer.k_norm_weight,
                                        kv_cache_.k_cache(i), kv_cache_.v_cache(i),
                                        start_pos,
                                        cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                        cfg_.head_dim, cfg_.rms_norm_eps, cfg_.rope_theta);
            });
        } else {
            // Prefill path: fused QKV projection
            time_stage(ProfileStage::FullQkvProj, [&] {
                layer.qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
            });
            time_stage(ProfileStage::FullSplit, [&] {
                ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v, seq_len, QD, KVD);
            });

            time_stage(ProfileStage::QkNormRope, [&] {
                // Qwen3 QK-norm: per-head RMSNorm on Q and K
                ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight,
                                   cfg_.rms_norm_eps, seq_len,
                                   cfg_.num_attention_heads, cfg_.head_dim);
                ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight,
                                   cfg_.rms_norm_eps, seq_len,
                                   cfg_.num_key_value_heads, cfg_.head_dim);

                // Apply RoPE to Q and K
                ops::apply_rope(ctx, buf_.q, buf_.k, seq_len, start_pos,
                                cfg_.num_attention_heads, cfg_.num_key_value_heads,
                                cfg_.head_dim, cfg_.rope_theta);
            });

            time_stage(ProfileStage::KvCache, [&] {
                // Write K, V to cache
                ops::copy_to_cache(ctx, buf_.k, kv_cache_.k_cache(i),
                                   seq_len, start_pos,
                                   cfg_.num_key_value_heads, cfg_.head_dim,
                                   kv_cache_.max_length());
                ops::copy_to_cache(ctx, buf_.v, kv_cache_.v_cache(i),
                                   seq_len, start_pos,
                                   cfg_.num_key_value_heads, cfg_.head_dim,
                                   kv_cache_.max_length());
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
                                       buf_.attn_out, buf_.scores,
                                       seq_len,
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
            layer.o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);
        });

        // Fused: residual + norm
        time_stage(ProfileStage::PostAttnNorm, [&] {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                    *layer.post_attn_ln_weight, cfg_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        });

        if (seq_len == 1) {
            // Decode: fused gate/up + SwiGLU
            time_stage(ProfileStage::FfnProj, [&] {
                layer.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, FF);
            });
        } else {
            // Prefill: fused gate/up projection
            time_stage(ProfileStage::FfnProj, [&] {
                layer.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::split_gate_up(ctx, buf_.gate_up, buf_.gate, buf_.up, seq_len, FF);
                ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
            });
        }

        // Down projection
        time_stage(ProfileStage::DownProj, [&] {
            layer.down_proj.forward(ctx, buf_.gate, buf_.up, seq_len);
        });

        // Residual + norm for next layer (or final)
        time_stage(ProfileStage::PostMlpNorm, [&] {
            if (i < cfg_.num_hidden_layers - 1) {
                ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                        *layers_[i + 1].input_ln_weight, cfg_.rms_norm_eps,
                                        buf_.normed, seq_len, H);
            } else {
                ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.up,
                                        *final_norm_weight_, cfg_.rms_norm_eps,
                                        buf_.normed, seq_len, H);
            }
        });
    }

    // LM Head: logits for the last token
    time_stage(ProfileStage::LmHead, [&] {
        if (seq_len > 1) {
            bf16* last_token_ptr = static_cast<bf16*>(buf_.normed.data()) + (seq_len - 1) * H;
            Tensor last_hidden = Tensor::view(ctx, last_token_ptr, {1, (int64_t)H});
            lm_head_.forward(ctx, last_hidden, buf_.logits, 1);
        } else {
            lm_head_.forward(ctx, buf_.normed, buf_.logits, 1);
        }
    });

    kv_cache_.advance(seq_len);

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

void Qwen3DenseBackend::reset() {
    kv_cache_.reset();
}

bool Qwen3DenseBackend::truncate_kv_cache(int new_len) {
    kv_cache_.truncate(new_len);
    return true;
}
