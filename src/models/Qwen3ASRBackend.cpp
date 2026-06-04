#include "Qwen3ASRBackend.hpp"
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
int round_up_seq(int v, int g) { return ((v + g - 1) / g) * g; }

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

// ---- Buffer management ----

void Qwen3ASRBackend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;
    int H = cfg_.hidden_size, QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim, FF = cfg_.intermediate_size;
    int cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.hidden   = Tensor::allocate(ctx, {cap, H});
    buf_.normed   = Tensor::allocate(ctx, {cap, H});
    buf_.qkv      = Tensor::allocate(ctx, {cap, QD + 2 * KVD});
    buf_.q        = Tensor::allocate(ctx, {cap, QD});
    buf_.k        = Tensor::allocate(ctx, {cap, KVD});
    buf_.v        = Tensor::allocate(ctx, {cap, KVD});
    buf_.attn_out = Tensor::allocate(ctx, {cap, QD});
    buf_.gate_up  = Tensor::allocate(ctx, {cap, 2 * FF});
    buf_.gate     = Tensor::allocate(ctx, {cap, FF});
    buf_.up       = Tensor::allocate(ctx, {cap, FF});
    runtime_seq_capacity_ = cap;
}

void Qwen3ASRBackend::ensure_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= prefill_scores_capacity_) return;
    int cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    buf_.scores = Tensor::allocate(ctx,
        {cfg_.num_attention_heads, cap, cap}, dnnl::memory::data_type::f32);
    prefill_scores_capacity_ = cap;
}

void Qwen3ASRBackend::ensure_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= incr_prefill_seq_cap_ && total_len <= incr_prefill_total_cap_) return;
    int sc = round_up_seq(std::max(seq_len, incr_prefill_seq_cap_), 16);
    int tc = round_up_seq(std::max(total_len, incr_prefill_total_cap_), 64);
    buf_.incr_scores = Tensor::allocate(ctx,
        {cfg_.num_attention_heads, sc, tc}, dnnl::memory::data_type::f32);
    incr_prefill_seq_cap_ = sc; incr_prefill_total_cap_ = tc;
}

// ---- MRoPE upload ----

void Qwen3ASRBackend::upload_mrope_positions(Context& ctx) {
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

// ---- Embedding overrides ----

void Qwen3ASRBackend::set_embedding_overrides(const std::vector<int>& positions,
                                               const std::vector<bf16>& embeddings,
                                               int hidden_size) {
    has_embedding_overrides_ = true;
    override_positions_ = positions;
    override_embeddings_ = embeddings;
    override_hidden_size_ = hidden_size;
}

void Qwen3ASRBackend::clear_embedding_overrides() {
    has_embedding_overrides_ = false;
    override_positions_.clear();
    override_embeddings_.clear();
}

void Qwen3ASRBackend::set_mrope_positions(Context& ctx,
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

void Qwen3ASRBackend::clear_mrope_positions() {
    has_mrope_positions_ = false;
}

// ---- load ----

bool Qwen3ASRBackend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                             int max_seq_len, std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3ASR && spec.family != ModelFamily::Qwen3ForceAligner) {
        if (error_message) *error_message = "Qwen3ASRBackend: invalid model family";
        return false;
    }
    cfg_ = spec.qwen3;
    rope_ = cfg_.rope;
    max_seq_len_ = max_seq_len;
    int H = cfg_.hidden_size, QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim, FF = cfg_.intermediate_size;

    fused_weights_.clear();
    fused_weights_.reserve(cfg_.num_hidden_layers * 2);

    auto transpose_weight = [&](const std::string& name) {
        Tensor& src = weights.get(name);
        Tensor dst = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, dst);
        ctx.synchronize();
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    auto fuse_three_cols = [&](Tensor& a, Tensor& b, Tensor& c) {
        int64_t rows = a.shape(0), ac = a.shape(1), bc = b.shape(1), cc = c.shape(1);
        int64_t tc = ac + bc + cc;
        Tensor out = Tensor::allocate(ctx, {rows, tc}, a.dtype());
        bf16 *ap = a.data_as<bf16>(), *bp = b.data_as<bf16>(), *cp = c.data_as<bf16>();
        bf16 *op = out.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(rows * tc), [=](sycl::id<1> id) {
            int64_t i = id[0], r = i / tc, ci = i % tc;
            int64_t oi = r * tc + ci;
            if (ci < ac) op[oi] = ap[r * ac + ci];
            else if (ci < ac + bc) op[oi] = bp[r * bc + ci - ac];
            else op[oi] = cp[r * cc + ci - ac - bc];
        });
        return out;
    };

    auto fuse_two_cols = [&](Tensor& a, Tensor& b) {
        int64_t rows = a.shape(0), ac = a.shape(1), bc = b.shape(1), tc = ac + bc;
        Tensor out = Tensor::allocate(ctx, {rows, tc}, a.dtype());
        bf16 *ap = a.data_as<bf16>(), *bp = b.data_as<bf16>(), *op = out.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(rows * tc), [=](sycl::id<1> id) {
            int64_t i = id[0], r = i / tc, ci = i % tc;
            int64_t oi = r * tc + ci;
            if (ci < ac) op[oi] = ap[r * ac + ci];
            else op[oi] = bp[r * bc + ci - ac];
        });
        return out;
    };

    // Embedding
    embed_weight_ = &weights.get("thinker.model.embed_tokens.weight");

    // Layers
    layers_.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& L = layers_[i];
        std::string p = "thinker.model.layers." + std::to_string(i) + ".";

        L.input_ln_weight = &weights.get(p + "input_layernorm.weight");
        L.post_attn_ln_weight = &weights.get(p + "post_attention_layernorm.weight");

        Tensor* qw = transpose_weight(p + "self_attn.q_proj.weight");
        Tensor* kw = transpose_weight(p + "self_attn.k_proj.weight");
        Tensor* vw = transpose_weight(p + "self_attn.v_proj.weight");
        L.q_proj.init(ctx, *qw, H, QD, true);
        L.k_proj.init(ctx, *kw, H, KVD, true);
        L.v_proj.init(ctx, *vw, H, KVD, true);
        fused_weights_.push_back(fuse_three_cols(*qw, *kw, *vw));
        L.qkv_proj.init(ctx, fused_weights_.back(), H, QD + 2 * KVD, true);

        Tensor* ow = transpose_weight(p + "self_attn.o_proj.weight");
        L.o_proj.init(ctx, *ow, QD, H, true);

        L.q_norm_weight = &weights.get(p + "self_attn.q_norm.weight");
        L.k_norm_weight = &weights.get(p + "self_attn.k_norm.weight");

        Tensor* gw = transpose_weight(p + "mlp.gate_proj.weight");
        Tensor* uw = transpose_weight(p + "mlp.up_proj.weight");
        L.gate_proj.init(ctx, *gw, H, FF, true);
        L.up_proj.init(ctx, *uw, H, FF, true);
        fused_weights_.push_back(fuse_two_cols(*gw, *uw));
        L.gate_up_proj.init(ctx, fused_weights_.back(), H, 2 * FF, true);

        Tensor* dw = transpose_weight(p + "mlp.down_proj.weight");
        L.down_proj.init(ctx, *dw, FF, H, true);
    }

    // Final norm + LM head
    final_norm_weight_ = &weights.get("thinker.model.norm.weight");
    Tensor* lw = transpose_weight("thinker.lm_head.weight");
    lm_head_.init(ctx, *lw, H, cfg_.vocab_size, true);

    // KV Cache
    kv_cache_.init(ctx, cfg_, max_seq_len);

    runtime_seq_capacity_ = prefill_scores_capacity_ = 0;
    incr_prefill_seq_cap_ = incr_prefill_total_cap_ = 0;
    ensure_runtime_buffers(ctx, 1);

    buf_.logits = Tensor::allocate(ctx, {1, cfg_.vocab_size});
    buf_.decode_scores = Tensor::allocate(ctx,
        {cfg_.num_attention_heads, max_seq_len}, dnnl::memory::data_type::f32);
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

    return true;
}

// ---- forward ----

Tensor& Qwen3ASRBackend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    if (seq_len <= 0)
        throw std::runtime_error("Qwen3ASRBackend::forward: seq_len must be positive");

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

    int H = cfg_.hidden_size, QD = cfg_.num_attention_heads * cfg_.head_dim;
    int KVD = cfg_.num_key_value_heads * cfg_.head_dim, FF = cfg_.intermediate_size;
    int start_pos = current_len_;
    int cached_len = start_pos + seq_len;
    if (cached_len > max_seq_len_)
        throw std::runtime_error("Context window exceeded");

    ensure_runtime_buffers(ctx, seq_len);
    if (seq_len > 1) {
        if (start_pos == 0) ensure_prefill_scores(ctx, seq_len);
        else ensure_incr_prefill_scores(ctx, seq_len, start_pos + seq_len);
    }

    time_stage(ProfileStage::EmbedNorm, [&] {
        // Embedding lookup
        ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len, buf_.hidden, H);

        // Apply audio embedding overrides (scatter audio features into <|audio_pad|> positions)
        if (has_embedding_overrides_ && !override_positions_.empty()) {
            size_t n_override = override_positions_.size();
            if (!override_buf_.valid() || override_buf_.numel() < static_cast<int64_t>(n_override * H)) {
                override_buf_ = Tensor::allocate(ctx, {static_cast<int64_t>(n_override), H});
            }
            ctx.memcpy_h2d(override_buf_.data(), override_embeddings_.data(),
                           n_override * H * sizeof(bf16));
            // Upload positions to persistent device buffer
            if (!override_pos_buf_.valid() || override_pos_buf_.numel() < static_cast<int64_t>(n_override)) {
                override_pos_buf_ = Tensor::allocate(ctx, {static_cast<int64_t>(n_override)},
                                                      dnnl::memory::data_type::s32);
            }
            ctx.memcpy_h2d(override_pos_buf_.data(), override_positions_.data(),
                           n_override * sizeof(int));

            // Scatter: for each override position, copy one row from override_buf_ into hidden
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
            // clear after applying
            has_embedding_overrides_ = false;
        }

        // Initial norm
        ops::rms_norm(ctx, buf_.hidden, *layers_[0].input_ln_weight,
                      cfg_.rms_norm_eps, buf_.normed, seq_len, H);
    });

    // MRoPE params — full rotary for Qwen3
    int rotary_dim = cfg_.head_dim;  // 128
    if (rotary_dim & 1) --rotary_dim;

    const int* pos_t = has_mrope_positions_ && mrope_pos_t_dev_.valid()
                       ? mrope_pos_t_dev_.data_as<int>() : nullptr;
    const int* pos_h = has_mrope_positions_ && mrope_pos_h_dev_.valid()
                       ? mrope_pos_h_dev_.data_as<int>() : nullptr;
    const int* pos_w = has_mrope_positions_ && mrope_pos_w_dev_.valid()
                       ? mrope_pos_w_dev_.data_as<int>() : nullptr;

    for (int i = 0; i < cfg_.num_hidden_layers; i++) {
        auto& L = layers_[i];
        Tensor q_decode_view;
        Tensor* q_for_attn = &buf_.q;

        if (seq_len == 1) {
            // Decode: fused QKV + QK-norm + MRoPE + cache write
            time_stage(ProfileStage::FullQkvProj, [&] {
                L.qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
            });
            bf16* qkv_ptr = buf_.qkv.data_as<bf16>();
            q_decode_view = Tensor::view(ctx, qkv_ptr, {1, QD});
            Tensor k_view = Tensor::view(ctx, qkv_ptr + QD, {1, KVD});
            Tensor v_view = Tensor::view(ctx, qkv_ptr + QD + KVD, {1, KVD});
            q_for_attn = &q_decode_view;

            time_stage(ProfileStage::QkNormRope, [&] {
                ops::decode_prepare_qkv_partial(ctx,
                    q_decode_view, k_view, v_view,
                    *L.q_norm_weight, *L.k_norm_weight,
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
            // Prefill
            time_stage(ProfileStage::FullQkvProj, [&] {
                L.qkv_proj.forward(ctx, buf_.normed, buf_.qkv, seq_len);
            });
            time_stage(ProfileStage::FullSplit, [&] {
                ops::split_qkv(ctx, buf_.qkv, buf_.q, buf_.k, buf_.v, seq_len, QD, KVD);
            });

            time_stage(ProfileStage::QkNormRope, [&] {
                ops::head_rms_norm(ctx, buf_.q, *L.q_norm_weight,
                                   cfg_.rms_norm_eps, seq_len,
                                   cfg_.num_attention_heads, cfg_.head_dim);
                ops::head_rms_norm(ctx, buf_.k, *L.k_norm_weight,
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
                ops::attention_decode(ctx, *q_for_attn,
                                      kv_cache_.k_cache(i), kv_cache_.v_cache(i),
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

        time_stage(ProfileStage::OProj, [&] {
            L.o_proj.forward(ctx, buf_.attn_out, buf_.gate, seq_len);
        });

        time_stage(ProfileStage::PostAttnNorm, [&] {
            ops::fused_add_rms_norm(ctx, buf_.hidden, buf_.gate,
                                    *L.post_attn_ln_weight, cfg_.rms_norm_eps,
                                    buf_.normed, seq_len, H);
        });

        if (seq_len == 1) {
            time_stage(ProfileStage::FfnProj, [&] {
                L.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::fused_gate_up_swiglu(ctx, buf_.gate_up, buf_.gate, FF);
            });
        } else {
            time_stage(ProfileStage::FfnProj, [&] {
                L.gate_up_proj.forward(ctx, buf_.normed, buf_.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::split_gate_up(ctx, buf_.gate_up, buf_.gate, buf_.up, seq_len, FF);
                ops::swiglu(ctx, buf_.gate, buf_.up, buf_.gate, seq_len * FF);
            });
        }

        time_stage(ProfileStage::DownProj, [&] {
            L.down_proj.forward(ctx, buf_.gate, buf_.up, seq_len);
        });

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

    time_stage(ProfileStage::LmHead, [&] {
        if (seq_len > 1) {
            bf16* last_ptr = buf_.normed.data_as<bf16>() + (seq_len - 1) * H;
            Tensor last = Tensor::view(ctx, last_ptr, {1, H});
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

void Qwen3ASRBackend::reset() {
    kv_cache_.reset();
    current_len_ = 0;
    has_embedding_overrides_ = false;
}

bool Qwen3ASRBackend::truncate_kv_cache(int new_len) {
    kv_cache_.truncate(new_len);
    current_len_ = new_len;
    return true;
}
