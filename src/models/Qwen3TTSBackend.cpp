#include "Qwen3TTSBackend.hpp"
#include "../ops/ConvOps.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <array>
#include <iostream>
#include <fstream>
#include <utility>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
int round_up_seq(int v, int g) { return ((v + g - 1) / g) * g; }
constexpr int kMimiSamplesPerFrame = 1920;

class ScopedAllocationProfile {
public:
    ScopedAllocationProfile(Context& ctx, bool enabled, const char* label)
        : ctx_(ctx), enabled_(enabled), label_(label) {
        if (enabled_) {
            ctx_.reset_allocation_stats();
            ctx_.set_allocation_stats_enabled(true);
        }
    }

    ScopedAllocationProfile(const ScopedAllocationProfile&) = delete;
    ScopedAllocationProfile& operator=(const ScopedAllocationProfile&) = delete;

    ~ScopedAllocationProfile() {
        if (!enabled_) {
            return;
        }
        const auto stats = ctx_.allocation_stats();
        ctx_.set_allocation_stats_enabled(false);
        const double alloc_mb = static_cast<double>(stats.allocated_bytes) / (1024.0 * 1024.0);
        const double freed_mb = static_cast<double>(stats.freed_bytes) / (1024.0 * 1024.0);
        AILA_LOG_INFO("[TTS-Profile]   %s allocs: %lld alloc / %lld free, %.2f / %.2f MB, alloc %.3f ms (max %.3f), free %.3f ms (max %.3f)",
                      label_,
                      static_cast<long long>(stats.alloc_count),
                      static_cast<long long>(stats.free_count),
                      alloc_mb,
                      freed_mb,
                      stats.alloc_ms_total,
                      stats.alloc_ms_max,
                      stats.free_ms_total,
                      stats.free_ms_max);
    }

private:
    Context& ctx_;
    bool enabled_ = false;
    const char* label_ = "";
};

void print_gpu_tensor(Context& ctx, const std::string& name, Tensor& tensor, int offset = 0) {
    std::cout << name << " = [";
    if (tensor.dtype() == dnnl::memory::data_type::bf16) {
        bf16 cpu_data[5];
        ctx.queue().memcpy(cpu_data, tensor.data_as<bf16>() + offset, 5 * sizeof(bf16)).wait();
        for (int i = 0; i < 5; ++i) {
            std::cout << static_cast<float>(cpu_data[i]) << (i < 4 ? ", " : "");
        }
    } else if (tensor.dtype() == dnnl::memory::data_type::f32) {
        float cpu_data[5];
        ctx.queue().memcpy(cpu_data, tensor.data_as<float>() + offset, 5 * sizeof(float)).wait();
        for (int i = 0; i < 5; ++i) {
            std::cout << cpu_data[i] << (i < 4 ? ", " : "");
        }
    } else {
        std::cout << "unsupported dtype";
    }
    std::cout << "]" << std::endl;
}

void print_gpu_tensor_tokens(Context& ctx, const std::string& name, Tensor& tensor, int H) {
    std::vector<int> tokens = {0, 1, 2, 8};
    for (int t : tokens) {
        int64_t total_elements = tensor.shape(0);
        if (tensor.ndim() > 1) {
            total_elements = 1;
            for (int i = 0; i < tensor.ndim(); ++i) {
                total_elements *= tensor.shape(i);
            }
        }
        if (t * H >= total_elements) continue;
        std::cout << name << "[0, " << t << ", :5] = [";
        if (tensor.dtype() == dnnl::memory::data_type::bf16) {
            bf16 cpu_data[5];
            ctx.queue().memcpy(cpu_data, tensor.data_as<bf16>() + t * H, 5 * sizeof(bf16)).wait();
            for (int i = 0; i < 5; ++i) {
                std::cout << static_cast<float>(cpu_data[i]) << (i < 4 ? ", " : "");
            }
        } else if (tensor.dtype() == dnnl::memory::data_type::f32) {
            float cpu_data[5];
            ctx.queue().memcpy(cpu_data, tensor.data_as<float>() + t * H, 5 * sizeof(float)).wait();
            for (int i = 0; i < 5; ++i) {
                std::cout << cpu_data[i] << (i < 4 ? ", " : "");
            }
        }
        std::cout << "]" << std::endl;
    }
}
}

// ---- Buffer management for Talker ----
void Qwen3TTSBackend::ensure_talker_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= talker_runtime_seq_capacity_) return;
    int H = talker_cfg_.hidden_size;
    int QD = talker_cfg_.num_attention_heads * talker_cfg_.head_dim;
    int KVD = talker_cfg_.num_key_value_heads * talker_cfg_.head_dim;
    int FF = talker_cfg_.intermediate_size;
    int cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    
    t_buf_.hidden   = Tensor::allocate(ctx, {cap, H});
    t_buf_.normed   = Tensor::allocate(ctx, {cap, H});
    t_buf_.qkv      = Tensor::allocate(ctx, {cap, QD + 2 * KVD});
    t_buf_.q        = Tensor::allocate(ctx, {cap, QD});
    t_buf_.k        = Tensor::allocate(ctx, {cap, KVD});
    t_buf_.v        = Tensor::allocate(ctx, {cap, KVD});
    t_buf_.attn_out = Tensor::allocate(ctx, {cap, QD});
    t_buf_.gate_up  = Tensor::allocate(ctx, {cap, 2 * FF});
    t_buf_.gate     = Tensor::allocate(ctx, {cap, FF});
    t_buf_.up       = Tensor::allocate(ctx, {cap, FF});
    talker_runtime_seq_capacity_ = cap;
}

void Qwen3TTSBackend::ensure_talker_prefill_scores(Context& ctx, int seq_len) {
    if (seq_len <= talker_prefill_scores_capacity_) return;
    int cap = std::min(max_seq_len_, round_up_seq(seq_len, 64));
    t_buf_.scores = Tensor::allocate(ctx,
        {talker_cfg_.num_attention_heads, cap, cap}, dnnl::memory::data_type::f32);
    talker_prefill_scores_capacity_ = cap;
}

void Qwen3TTSBackend::ensure_talker_incr_prefill_scores(Context& ctx, int seq_len, int total_len) {
    if (seq_len <= talker_incr_prefill_seq_cap_ && total_len <= talker_incr_prefill_total_cap_) return;
    int sc = round_up_seq(std::max(seq_len, talker_incr_prefill_seq_cap_), 16);
    int tc = round_up_seq(std::max(total_len, talker_incr_prefill_total_cap_), 64);
    t_buf_.incr_scores = Tensor::allocate(ctx,
        {talker_cfg_.num_attention_heads, sc, tc}, dnnl::memory::data_type::f32);
    talker_incr_prefill_seq_cap_ = sc; talker_incr_prefill_total_cap_ = tc;
}

// ---- load ----
bool Qwen3TTSBackend::load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
                           int max_seq_len, std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3TTS) {
        if (error_message) *error_message = "Qwen3TTSBackend: invalid model family";
        return false;
    }
    
    talker_cfg_ = spec.qwen3;
    predictor_cfg_ = spec.code_predictor;
    max_seq_len_ = max_seq_len;
    tts_model_type_ = talker_cfg_.tts_model_type;
    AILA_LOG_INFO("[TTS] Model type: %s",
        tts_model_type_ == Qwen3TTSModelType::Base ? "Base" :
        tts_model_type_ == Qwen3TTSModelType::CustomVoice ? "CustomVoice" : "VoiceDesign");

    int H_talker = talker_cfg_.hidden_size;
    int QD_talker = talker_cfg_.num_attention_heads * talker_cfg_.head_dim;
    int KVD_talker = talker_cfg_.num_key_value_heads * talker_cfg_.head_dim;
    int FF_talker = talker_cfg_.intermediate_size;

    talker_fused_weights_.clear();
    talker_fused_weights_.reserve(talker_cfg_.num_hidden_layers * 2);

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

    // ============================================================
    // Load Talker weights
    // ============================================================
    talker_codec_embed_weight_ = &weights.get("talker.model.codec_embedding.weight");
    talker_text_embed_weight_ = &weights.get("talker.model.text_embedding.weight");

    // Text projection (ResizeMLP)
    Tensor* tp_fc1_w = transpose_weight("talker.text_projection.linear_fc1.weight");
    talker_text_proj_fc1_bias_ = &weights.get("talker.text_projection.linear_fc1.bias");
    talker_text_proj_fc1_.init(ctx, *tp_fc1_w, talker_cfg_.text_hidden_size, talker_cfg_.text_hidden_size, true);

    Tensor* tp_fc2_w = transpose_weight("talker.text_projection.linear_fc2.weight");
    talker_text_proj_fc2_bias_ = &weights.get("talker.text_projection.linear_fc2.bias");
    talker_text_proj_fc2_.init(ctx, *tp_fc2_w, talker_cfg_.text_hidden_size, talker_cfg_.hidden_size, true);

    // Talker layers
    talker_layers_.resize(talker_cfg_.num_hidden_layers);
    for (int i = 0; i < talker_cfg_.num_hidden_layers; i++) {
        auto& L = talker_layers_[i];
        std::string p = "talker.model.layers." + std::to_string(i) + ".";

        L.input_ln_weight = &weights.get(p + "input_layernorm.weight");
        L.post_attn_ln_weight = &weights.get(p + "post_attention_layernorm.weight");

        Tensor* qw = transpose_weight(p + "self_attn.q_proj.weight");
        Tensor* kw = transpose_weight(p + "self_attn.k_proj.weight");
        Tensor* vw = transpose_weight(p + "self_attn.v_proj.weight");
        talker_fused_weights_.push_back(fuse_three_cols(*qw, *kw, *vw));
        L.qkv_proj.init(ctx, talker_fused_weights_.back(), H_talker, QD_talker + 2 * KVD_talker, true);

        Tensor* ow = transpose_weight(p + "self_attn.o_proj.weight");
        L.o_proj.init(ctx, *ow, QD_talker, H_talker, true);

        L.q_norm_weight = &weights.get(p + "self_attn.q_norm.weight");
        L.k_norm_weight = &weights.get(p + "self_attn.k_norm.weight");

        Tensor* gw = transpose_weight(p + "mlp.gate_proj.weight");
        Tensor* uw = transpose_weight(p + "mlp.up_proj.weight");
        talker_fused_weights_.push_back(fuse_two_cols(*gw, *uw));
        L.gate_up_proj.init(ctx, talker_fused_weights_.back(), H_talker, 2 * FF_talker, true);

        Tensor* dw = transpose_weight(p + "mlp.down_proj.weight");
        L.down_proj.init(ctx, *dw, FF_talker, H_talker, true);
    }

    talker_final_norm_weight_ = &weights.get("talker.model.norm.weight");
    Tensor* lw = transpose_weight("talker.codec_head.weight");
    talker_codec_head_.init(ctx, *lw, H_talker, talker_cfg_.vocab_size, true);

    talker_kv_cache_.init(ctx, talker_cfg_, max_seq_len_, "AILA_TTS_KV_QUANT");

    // ============================================================
    // Load Code Predictor weights
    // ============================================================
    int H_pred = predictor_cfg_.hidden_size;
    int QD_pred = predictor_cfg_.num_attention_heads * predictor_cfg_.head_dim;
    int KVD_pred = predictor_cfg_.num_key_value_heads * predictor_cfg_.head_dim;
    int FF_pred = predictor_cfg_.intermediate_size;

    predictor_fused_weights_.clear();
    predictor_fused_weights_.reserve(predictor_cfg_.num_hidden_layers * 2);

    has_predictor_projection_ = (H_talker != H_pred);
    if (has_predictor_projection_) {
        Tensor* proj_w = transpose_weight("talker.code_predictor.small_to_mtp_projection.weight");
        predictor_projection_bias_ = &weights.get("talker.code_predictor.small_to_mtp_projection.bias");
        predictor_projection_linear_.init(ctx, *proj_w, H_talker, H_pred, true);
    }

    // Embeddings
    predictor_embed_weights_.resize(predictor_cfg_.num_code_groups - 1);
    for (int i = 0; i < predictor_cfg_.num_code_groups - 1; i++) {
        std::string name = "talker.code_predictor.model.codec_embedding." + std::to_string(i) + ".weight";
        predictor_embed_weights_[i] = &weights.get(name);
    }

    // Layers
    predictor_layers_.resize(predictor_cfg_.num_hidden_layers);
    for (int i = 0; i < predictor_cfg_.num_hidden_layers; i++) {
        auto& L = predictor_layers_[i];
        std::string p = "talker.code_predictor.model.layers." + std::to_string(i) + ".";

        L.input_ln_weight = &weights.get(p + "input_layernorm.weight");
        L.post_attn_ln_weight = &weights.get(p + "post_attention_layernorm.weight");

        Tensor* qw = transpose_weight(p + "self_attn.q_proj.weight");
        Tensor* kw = transpose_weight(p + "self_attn.k_proj.weight");
        Tensor* vw = transpose_weight(p + "self_attn.v_proj.weight");
        predictor_fused_weights_.push_back(fuse_three_cols(*qw, *kw, *vw));
        L.qkv_proj.init(ctx, predictor_fused_weights_.back(), H_pred, QD_pred + 2 * KVD_pred, true);

        Tensor* ow = transpose_weight(p + "self_attn.o_proj.weight");
        L.o_proj.init(ctx, *ow, QD_pred, H_pred, true);

        L.q_norm_weight = &weights.get(p + "self_attn.q_norm.weight");
        L.k_norm_weight = &weights.get(p + "self_attn.k_norm.weight");

        Tensor* gw = transpose_weight(p + "mlp.gate_proj.weight");
        Tensor* uw = transpose_weight(p + "mlp.up_proj.weight");
        predictor_fused_weights_.push_back(fuse_two_cols(*gw, *uw));
        L.gate_up_proj.init(ctx, predictor_fused_weights_.back(), H_pred, 2 * FF_pred, true);

        Tensor* dw = transpose_weight(p + "mlp.down_proj.weight");
        L.down_proj.init(ctx, *dw, FF_pred, H_pred, true);
    }

    predictor_final_norm_weight_ = &weights.get("talker.code_predictor.model.norm.weight");

    // Output heads
    predictor_lm_heads_.resize(predictor_cfg_.num_code_groups - 1);
    for (int i = 0; i < predictor_cfg_.num_code_groups - 1; i++) {
        std::string name = "talker.code_predictor.lm_head." + std::to_string(i) + ".weight";
        Tensor* lhw = transpose_weight(name);
        predictor_lm_heads_[i].init(ctx, *lhw, H_pred, predictor_cfg_.vocab_size, true);
    }

    predictor_kv_cache_.init(ctx, predictor_cfg_, 17, "AILA_TTS_KV_QUANT");

    // ============================================================
    // Allocate runtime buffers
    // ============================================================
    talker_runtime_seq_capacity_ = talker_prefill_scores_capacity_ = 0;
    talker_incr_prefill_seq_cap_ = talker_incr_prefill_total_cap_ = 0;
    ensure_talker_runtime_buffers(ctx, 1);

    t_buf_.logits = Tensor::allocate(ctx, {1, talker_cfg_.vocab_size});
    t_buf_.decode_scores = Tensor::allocate(ctx,
        {talker_cfg_.num_attention_heads, max_seq_len_}, dnnl::memory::data_type::f32);
    if (talker_cfg_.head_dim == 256 || talker_cfg_.head_dim == 128) {
        constexpr int kDecodeExactTile = 128;
        const int partial_stride = (talker_cfg_.head_dim == 256) ? 272 : 144;
        const int max_tiles = (max_seq_len_ + kDecodeExactTile - 1) / kDecodeExactTile;
        t_buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {(int64_t)talker_cfg_.num_attention_heads, (int64_t)max_tiles, (int64_t)partial_stride},
            dnnl::memory::data_type::f32);
    }

    // Allocate predictor runtime buffers once
    p_buf_.hidden   = Tensor::allocate(ctx, {17, H_pred});
    p_buf_.normed   = Tensor::allocate(ctx, {17, H_pred});
    p_buf_.qkv      = Tensor::allocate(ctx, {17, QD_pred + 2 * KVD_pred});
    p_buf_.q        = Tensor::allocate(ctx, {17, QD_pred});
    p_buf_.k        = Tensor::allocate(ctx, {17, KVD_pred});
    p_buf_.v        = Tensor::allocate(ctx, {17, KVD_pred});
    p_buf_.attn_out = Tensor::allocate(ctx, {17, QD_pred});
    p_buf_.gate_up  = Tensor::allocate(ctx, {17, 2 * FF_pred});
    p_buf_.gate     = Tensor::allocate(ctx, {17, FF_pred});
    p_buf_.up       = Tensor::allocate(ctx, {17, FF_pred});
    p_buf_.logits   = Tensor::allocate(ctx, {1, predictor_cfg_.vocab_size});
    p_buf_.pred_input_proj = Tensor::allocate(ctx, {2, H_pred});
    
    p_buf_.scores = Tensor::allocate(ctx, {predictor_cfg_.num_attention_heads, 17, 17}, dnnl::memory::data_type::f32);
    p_buf_.decode_scores = Tensor::allocate(ctx, {predictor_cfg_.num_attention_heads, 17}, dnnl::memory::data_type::f32);
    if (predictor_cfg_.head_dim == 128) {
        p_buf_.decode_attn_partials = Tensor::allocate(
            ctx,
            {(int64_t)predictor_cfg_.num_attention_heads, 1, 144},
            dnnl::memory::data_type::f32);
    }

    // Warmup: run one dummy talker layer to trigger oneDNN JIT compilation.
    // Without this, the first real prefill pays ~5s of JIT overhead.
    {
        int H = talker_cfg_.hidden_size;
        int QD = talker_cfg_.num_attention_heads * talker_cfg_.head_dim;
        int KVD = talker_cfg_.num_key_value_heads * talker_cfg_.head_dim;
        int FF = talker_cfg_.intermediate_size;

        AILA_LOG_INFO("[TTS] Running talker warmup (JIT compilation)...");
        ensure_talker_runtime_buffers(ctx, 1);

        // Dummy input
        Tensor warmup_in = Tensor::allocate(ctx, {1, H});
        ctx.queue().memset(warmup_in.data(), 0, H * sizeof(bf16)).wait();
        ctx.queue().memcpy(t_buf_.hidden.data(), warmup_in.data(), H * sizeof(bf16)).wait();

        // RMS norm + QKV projection (triggers GEMM for [1,H]×[H,QD+2KVD])
        ops::rms_norm(ctx, t_buf_.hidden, *talker_layers_[0].input_ln_weight,
                      talker_cfg_.rms_norm_eps, t_buf_.normed, 1, H);
        talker_layers_[0].qkv_proj.forward(ctx, t_buf_.normed, t_buf_.qkv, 1);
        talker_layers_[0].o_proj.forward(ctx, t_buf_.qkv, t_buf_.attn_out, 1);

        // FFN (triggers GEMM for [1,H]×[H,2FF] and [1,FF]×[FF,H])
        ops::rms_norm(ctx, t_buf_.hidden, *talker_layers_[0].post_attn_ln_weight,
                      talker_cfg_.rms_norm_eps, t_buf_.normed, 1, H);
        talker_layers_[0].gate_up_proj.forward(ctx, t_buf_.normed, t_buf_.gate_up, 1);
        talker_layers_[0].down_proj.forward(ctx, t_buf_.gate_up, t_buf_.attn_out, 1);

        // Codec head (triggers GEMM for [1,H]×[H,vocab])
        talker_codec_head_.forward(ctx, t_buf_.normed, t_buf_.logits, 1);

        ctx.synchronize();
        reset();
        AILA_LOG_INFO("[TTS] Talker warmup complete");

        // Extend warmup: text_projection fc1/fc2 use different dimensions
        // (2048→2048, 2048→1024) than the talker layers. Pre-compile them at
        // prefill batch size to avoid JIT during the first real prefill.
        {
            int wb = 4;
            Tensor wt = Tensor::allocate(ctx, {wb, 2048});
            ctx.queue().memset(wt.data(), 0, wb * 2048 * sizeof(bf16));
            Tensor wf1 = Tensor::allocate(ctx, {wb, 2048});
            talker_text_proj_fc1_.forward_bias(ctx, wt, *talker_text_proj_fc1_bias_, wf1, wb);
            Tensor ws = Tensor::allocate(ctx, {wb, 2048});
            ops::sigmoid_mul(ctx, wf1, wf1, ws, wb * 2048);
            Tensor wf2 = Tensor::allocate(ctx, {wb, H_talker});
            talker_text_proj_fc2_.forward_bias(ctx, ws, *talker_text_proj_fc2_bias_, wf2, wb);
            ctx.synchronize();
            AILA_LOG_INFO("[TTS] Text projection warmup complete (batch=%d)", wb);
        }
    }

    // Pre-compute fixed embeddings (bos/eos/pad + codec_pad/codec_bos)
    // These are constants used in every synthesize_codes call — computing them
    // once eliminates ~250ms of redundant ResizeMLP forward passes per synthesis.
    {
        auto compute_tts_embed = [&](int token_id) -> Tensor {
            Tensor t_id = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
            ctx.memcpy_h2d(t_id.data(), &token_id, sizeof(int));
            Tensor emb = Tensor::allocate(ctx, {1, 2048});
            ops::embedding_lookup(ctx, *talker_text_embed_weight_, t_id.data_as<int>(), 1, emb, 2048);
            Tensor f1 = Tensor::allocate(ctx, {1, 2048});
            talker_text_proj_fc1_.forward_bias(ctx, emb, *talker_text_proj_fc1_bias_, f1, 1);
            Tensor s1 = Tensor::allocate(ctx, {1, 2048});
            ops::sigmoid_mul(ctx, f1, f1, s1, 2048);
            Tensor out = Tensor::allocate(ctx, {1, H_talker});
            talker_text_proj_fc2_.forward_bias(ctx, s1, *talker_text_proj_fc2_bias_, out, 1);
            return out;
        };
        precomputed_tts_bos_ = compute_tts_embed(talker_cfg_.tts_bos_token_id);
        precomputed_tts_eos_ = compute_tts_embed(talker_cfg_.tts_eos_token_id);
        precomputed_tts_pad_ = compute_tts_embed(talker_cfg_.tts_pad_token_id);

        auto compute_codec_embed = [&](int codec_id) -> Tensor {
            Tensor c_id = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
            ctx.memcpy_h2d(c_id.data(), &codec_id, sizeof(int));
            Tensor out = Tensor::allocate(ctx, {1, H_talker});
            ops::embedding_lookup(ctx, *talker_codec_embed_weight_, c_id.data_as<int>(), 1, out, H_talker);
            return out;
        };
        precomputed_codec_pad_ = compute_codec_embed(talker_cfg_.codec_pad_id);
        precomputed_codec_bos_ = compute_codec_embed(talker_cfg_.codec_bos_id);

        ctx.synchronize();
        AILA_LOG_INFO("[TTS] Pre-computed fixed embeddings");
    }

    // Warm the real codec-generation path after fixed embeddings are ready.
    // The earlier warmup covers representative GEMMs, but the first product
    // utterance also needs predictor decode/talker decode kernels.
    {
        AILA_LOG_INFO("[TTS] Running codec decode warmup...");
        GenerationConfig warmup_gen{};
        warmup_gen.max_new_tokens = 2;
        warmup_gen.temperature = 0.0f;
        warmup_gen.top_k = 1;
        warmup_gen.top_p = 1.0f;
        warmup_gen.do_sample = false;
        warmup_gen.repetition_penalty = 1.1f;

        std::vector<int> warmup_text = {151644, 77091, 198, 0, 151645};
        std::vector<int32_t> warmup_codes;
        int warmup_frames = 0;
        if (!synthesize_codes(ctx, warmup_text, {}, 0, {}, 0, warmup_gen,
                              warmup_codes, warmup_frames,
                              []() { return false; }, {}, 0)) {
            AILA_LOG_WARN("[TTS] Codec decode warmup failed; first synthesis may pay JIT cost");
        } else {
            AILA_LOG_INFO("[TTS] Codec decode warmup complete (frames=%d)", warmup_frames);
        }
        reset();
    }

    return true;
}

void Qwen3TTSBackend::reset() {
    talker_kv_cache_.reset();
    predictor_kv_cache_.reset();
    current_talker_len_ = 0;
}

bool Qwen3TTSBackend::truncate_kv_cache(int new_len) {
    if (new_len < current_talker_len_) {
        talker_kv_cache_.truncate(new_len);
        current_talker_len_ = new_len;
        return true;
    }
    return false;
}

// 我们继承 IModelBackend 必须实现的 forward
// 此 forward 实现 Talker 的单步前向以匹配已有的推理流调试。
Tensor& Qwen3TTSBackend::forward(Context& ctx, const int* token_ids_device, int seq_len) {
    throw std::runtime_error("Qwen3TTSBackend::forward is not used. Call synthesize_codes instead.");
}

// ============================================================
// Qwen3-TTS 专有的自回归合成核心 (synthesize_codes)
// ============================================================
bool Qwen3TTSBackend::synthesize_codes(Context& ctx,
                                       const std::vector<int>& text_tokens,
                                       const std::vector<float>& speaker_embedding,
                                       int speaker_id,
                                       const std::vector<int>& instruct_tokens,
                                       int language_id,
                                       const GenerationConfig& gen_config,
                                       std::vector<int32_t>& out_codes,
                                       int& out_n_frames,
                                       std::function<bool()> should_cancel,
                                       CodeFrameCallback frame_callback,
                                       int frame_callback_batch_frames) {
    out_codes.clear();
    out_n_frames = 0;
    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };
    if (cancelled()) {
        return false;
    }

    static const bool tts_profile = aila::env::read_flag("AILA_TTS_PROFILE", false);
    static const bool tts_debug = aila::env::read_flag("AILA_TTS_DEBUG", false);

    int L = static_cast<int>(text_tokens.size());
    if (L <= 0) return false;

    int H_talker = talker_cfg_.hidden_size;
    int QD_talker = talker_cfg_.num_attention_heads * talker_cfg_.head_dim;
    int KVD_talker = talker_cfg_.num_key_value_heads * talker_cfg_.head_dim;
    int FF_talker = talker_cfg_.intermediate_size;

    int H_pred = predictor_cfg_.hidden_size;
    int QD_pred = predictor_cfg_.num_attention_heads * predictor_cfg_.head_dim;
    int KVD_pred = predictor_cfg_.num_key_value_heads * predictor_cfg_.head_dim;
    int FF_pred = predictor_cfg_.intermediate_size;

    // ------------------------------------------------------------------------
    // 1. Prefill 阶段: 文本投影与前置计算
    // ------------------------------------------------------------------------
    auto t_total_start = std::chrono::high_resolution_clock::now();
    auto t_prefill_start = t_total_start;
    reset();

    // 临时分配用于文本投影的 GPU 张量
    Tensor text_ids_dev = Tensor::allocate(ctx, {L}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(text_ids_dev.data(), text_tokens.data(), L * sizeof(int));

    Tensor text_emb = Tensor::allocate(ctx, {L, 2048});
    ops::embedding_lookup(ctx, *talker_text_embed_weight_, text_ids_dev.data_as<int>(), L, text_emb, 2048);
    if (tts_debug) print_gpu_tensor(ctx, "text_emb[0, 0, :5]", text_emb, 0);

    // 对 text_emb 过 ResizeMLP (text_projection)
    Tensor fc1_out = Tensor::allocate(ctx, {L, 2048});
    talker_text_proj_fc1_.forward_bias(ctx, text_emb, *talker_text_proj_fc1_bias_, fc1_out, L);
    if (tts_debug) print_gpu_tensor(ctx, "fc1_out[0, 0, :5]", fc1_out, 0);

    // SiLU(x) = x * sigmoid(x)
    Tensor silu_out = Tensor::allocate(ctx, {L, 2048});
    ops::sigmoid_mul(ctx, fc1_out, fc1_out, silu_out, L * 2048);
    if (tts_debug) print_gpu_tensor(ctx, "silu_out[0, 0, :5]", silu_out, 0);

    Tensor projected_text = Tensor::allocate(ctx, {L, H_talker});
    talker_text_proj_fc2_.forward_bias(ctx, silu_out, *talker_text_proj_fc2_bias_, projected_text, L);
    if (tts_debug) print_gpu_tensor(ctx, "projected_text[0, 0, :5]", projected_text, 0);

    // Use pre-computed embeddings (computed once during load)
    Tensor& tts_bos_embed = precomputed_tts_bos_;
    Tensor& tts_eos_embed = precomputed_tts_eos_;
    Tensor& tts_pad_embed = precomputed_tts_pad_;
    if (tts_debug) print_gpu_tensor(ctx, "tts_pad_embed[0, 0, :5]", tts_pad_embed, 0);

    // Codec embedding lookup helper (still needed for dynamic spk_id in CustomVoice)
    auto get_talker_codec_embed = [&](int codec_id) {
        Tensor c_id = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(c_id.data(), &codec_id, sizeof(int));
        Tensor out = Tensor::allocate(ctx, {1, H_talker});
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, c_id.data_as<int>(), 1, out, H_talker);
        return out;
    };

    Tensor& embed_codec_pad = precomputed_codec_pad_;
    Tensor& embed_codec_bos = precomputed_codec_bos_;

    // 正文提取：找到 <|im_end|> (151645) 的位置作为正文结束边界
    // BPE tokenizer 可能将 <|im_end|> 与相邻字符合并，不能用固定 L-5 计算
    // 前 3 个 tokens 是 role 前缀：`<|im_start|>assistant\n`
    int text_body_start = 3;
    int text_body_end = L; // default: all remaining tokens
    // 从后往前查找 <|im_end|> token (151645)
    for (int i = L - 1; i >= text_body_start; --i) {
        if (text_tokens[i] == 151645) { // <|im_end|>
            text_body_end = i;
            break;
        }
    }
    // 如果没找到 <|im_end|>，可能是 BPE 把 "<|im_end|>" 拆开了
    // 回退到查找 <|im_start|> (151644) 作为边界（正文之后的第一个 im_start）
    if (text_body_end == L) {
        for (int i = text_body_start + 1; i < L; ++i) {
            if (text_tokens[i] == 151644) { // <|im_start|>
                text_body_end = i; // 正文在 im_start 之前结束
                break;
            }
        }
    }

    // ==========================================
    // Instruct text prefix (for VoiceDesign and CustomVoice with style override)
    // ==========================================
    bool has_instruct = !instruct_tokens.empty();
    int instruct_offset = 0;
    Tensor instruct_embeds;
    if (has_instruct) {
        int IL = static_cast<int>(instruct_tokens.size());
        // Tokenize instruct text
        Tensor instruct_ids_dev = Tensor::allocate(ctx, {IL}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(instruct_ids_dev.data(), instruct_tokens.data(), IL * sizeof(int));

        // text_embedding lookup
        Tensor instruct_text_emb = Tensor::allocate(ctx, {IL, 2048});
        ops::embedding_lookup(ctx, *talker_text_embed_weight_, instruct_ids_dev.data_as<int>(), IL, instruct_text_emb, 2048);

        // Pass through text_projection (ResizeMLP)
        Tensor inst_fc1 = Tensor::allocate(ctx, {IL, 2048});
        talker_text_proj_fc1_.forward_bias(ctx, instruct_text_emb, *talker_text_proj_fc1_bias_, inst_fc1, IL);
        Tensor inst_silu = Tensor::allocate(ctx, {IL, 2048});
        ops::sigmoid_mul(ctx, inst_fc1, inst_fc1, inst_silu, IL * 2048);

        instruct_embeds = Tensor::allocate(ctx, {IL, H_talker});
        talker_text_proj_fc2_.forward_bias(ctx, inst_silu, *talker_text_proj_fc2_bias_, instruct_embeds, IL);

        instruct_offset = IL;
    }

    // ==========================================
    // Build codec prefill token list based on language
    // language_id == 0 means "auto" → use nothink mode
    // otherwise: think → think_bos → language → think_eos
    // ==========================================
    std::vector<int> codec_prefill_ids;
    if (language_id == 0) {
        codec_prefill_ids = {
            talker_cfg_.codec_nothink_id,
            talker_cfg_.codec_think_bos_id,
            talker_cfg_.codec_think_eos_id
        };
    } else {
        codec_prefill_ids = {
            talker_cfg_.codec_think_id,
            talker_cfg_.codec_think_bos_id,
            language_id,
            talker_cfg_.codec_think_eos_id
        };
    }
    int codec_prefill_count = static_cast<int>(codec_prefill_ids.size());

    // ==========================================
    // Determine prefill layout based on model type
    // Base layout:   [role(3)] [codec_prefill] [spk_embed(1)] [pad+bos(1)] [text+bos(1)]
    // CustomVoice:   [role(3)] [codec_prefill] [spk_codec(1)] [pad+bos(1)] [text+bos(1)]
    // VoiceDesign:   [role(3)] [codec_prefill] [pad+bos(1)] [text+bos(1)]  (no spk slot)
    // ==========================================
    int prefill_base_len = 3 + codec_prefill_count + 2; // role + codec_prefill + pad_bos + text_bos
    int has_spk = 0;
    int spk_idx = 0;

    if (tts_model_type_ == Qwen3TTSModelType::Base) {
        has_spk = (speaker_embedding.size() == static_cast<size_t>(H_talker)) ? 1 : 0;
    } else if (tts_model_type_ == Qwen3TTSModelType::CustomVoice && speaker_id > 0) {
        has_spk = 1;
    }
    // VoiceDesign: always has_spk = 0

    if (has_spk) spk_idx = 3 + codec_prefill_count;
    int prefill_len = prefill_base_len + has_spk;
    int first_text_idx = prefill_len - 1;
    int pad_bos_idx = 3 + codec_prefill_count + has_spk;

    int total_prefill_len = instruct_offset + prefill_len;
    Tensor prefill_embeds = Tensor::allocate(ctx, {total_prefill_len, H_talker});
    {
        bf16* dst = prefill_embeds.data_as<bf16>();

        // Copy instruct embeddings if present (they go BEFORE the main prefill)
        if (has_instruct) {
            bf16* src_inst = instruct_embeds.data_as<bf16>();
            ctx.queue().memcpy(dst, src_inst, instruct_offset * H_talker * sizeof(bf16)).wait();
            dst += instruct_offset * H_talker; // advance dst to write main prefill after instruct
        }

        bf16* src_role = projected_text.data_as<bf16>();

        // 1. 前 3 帧：role 前缀投影
        ctx.queue().memcpy(dst, src_role, 3 * H_talker * sizeof(bf16)).wait();

        // 2. codec_prefill tokens + tts_pad_embed
        Tensor codec_ids_dev = Tensor::allocate(ctx, {codec_prefill_count}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(codec_ids_dev.data(), codec_prefill_ids.data(), codec_prefill_count * sizeof(int));
        Tensor codec_embs = Tensor::allocate(ctx, {codec_prefill_count, H_talker});
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, codec_ids_dev.data_as<int>(), codec_prefill_count, codec_embs, H_talker);

        bf16* c_embs = codec_embs.data_as<bf16>();
        bf16* p_pad = tts_pad_embed.data_as<bf16>();
        bf16* p_bos = tts_bos_embed.data_as<bf16>();

        for (int i = 0; i < codec_prefill_count; ++i) {
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                dst[(3 + i) * H_talker + h] = c_embs[i * H_talker + h] + p_pad[h];
            });
        }
        ctx.queue().wait();

        // 3. Speaker slot (if present) — differs by model type
        if (has_spk) {
            if (tts_model_type_ == Qwen3TTSModelType::Base) {
                // Base: ECAPA-TDNN speaker embedding
                Tensor spk_emb_dev = Tensor::allocate(ctx, {1, H_talker});
                std::vector<bf16> spk_bf16(H_talker);
                for (int h = 0; h < H_talker; ++h) {
                    spk_bf16[h] = bf16(speaker_embedding[h]);
                }
                ctx.memcpy_h2d(spk_emb_dev.data(), spk_bf16.data(), H_talker * sizeof(bf16));
                bf16* s_emb = spk_emb_dev.data_as<bf16>();
                ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                    dst[spk_idx * H_talker + h] = s_emb[h] + p_pad[h];
                });
            } else {
                // CustomVoice: direct codec_embed(spk_id) token lookup
                Tensor spk_codec = get_talker_codec_embed(speaker_id);
                bf16* s_emb = spk_codec.data_as<bf16>();
                ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                    dst[spk_idx * H_talker + h] = s_emb[h] + p_pad[h];
                });
            }
            ctx.queue().wait();
        }

        // 4. codec_pad + tts_bos_embed
        bf16* c_pad = embed_codec_pad.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
            dst[pad_bos_idx * H_talker + h] = c_pad[h] + p_bos[h];
        });
        ctx.queue().wait();

        // 5. 最后一帧：首个正文 token + codec_bos
        if (text_body_end > text_body_start) {
            bf16* first_text = projected_text.data_as<bf16>() + text_body_start * H_talker;
            bf16* c_bos = embed_codec_bos.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                dst[first_text_idx * H_talker + h] = first_text[h] + c_bos[h];
            });
        } else {
            // 无正文 token 的退化情况：用 tts_eos_embed + codec_bos
            bf16* p_eos = tts_eos_embed.data_as<bf16>();
            bf16* c_bos = embed_codec_bos.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                dst[first_text_idx * H_talker + h] = p_eos[h] + c_bos[h];
            });
        }
        ctx.queue().wait();
    }

    // 构建 trailing_text_hidden：从第 2 个正文 token 开始到 <|im_end|> 之前的所有正文 token
    // 加上 tts_eos_embed 作为结尾，用于 decode 逐步注入时提供文本时序指导
    int trailing_token_count = std::max(0, text_body_end - text_body_start - 1);
    int trailing_len = trailing_token_count + 1; // +1 for tts_eos_embed at end
    AILA_LOG_INFO("[TTS] text_body_end=%d (detected), trailing_tokens=%d, trailing_len=%d",
                  text_body_end, trailing_token_count, trailing_len);
    Tensor trailing_text_hidden = Tensor::allocate(ctx, {trailing_len, H_talker});
    if (trailing_token_count > 0) {
        bf16* all_proj = projected_text.data_as<bf16>();
        bf16* trailing_ptr = trailing_text_hidden.data_as<bf16>();
        for (int t = 0; t < trailing_token_count; ++t) {
            ctx.queue().memcpy(trailing_ptr + t * H_talker,
                               all_proj + (text_body_start + 1 + t) * H_talker,
                               H_talker * sizeof(bf16));
        }
    }
    // 末尾追加 tts_eos_embed
    {
        ctx.queue().memcpy(trailing_text_hidden.data_as<bf16>() + (trailing_len - 1) * H_talker,
                           tts_eos_embed.data(), H_talker * sizeof(bf16));
    }
    ctx.queue().wait();

    if (tts_debug) print_gpu_tensor(ctx, "prefill_embeds[0, 0, :5]", prefill_embeds, 0);

    auto t_talker_setup_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_prefill_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Text+embed construction: %.1f ms", t_talker_setup_ms);

    // 运行 Talker Prefill
    auto t_talker_fwd_start = std::chrono::high_resolution_clock::now();
    ensure_talker_runtime_buffers(ctx, total_prefill_len);
    ensure_talker_prefill_scores(ctx, total_prefill_len);

    // 拷贝 prefill_embeds 到 t_buf_.hidden
    ctx.queue().memcpy(t_buf_.hidden.data(), prefill_embeds.data(), total_prefill_len * H_talker * sizeof(bf16)).wait();

    // 初始归一化
    ops::rms_norm(ctx, t_buf_.hidden, *talker_layers_[0].input_ln_weight,
                  talker_cfg_.rms_norm_eps, t_buf_.normed, total_prefill_len, H_talker);

    int rotary_dim_talker = talker_cfg_.head_dim;
    if (rotary_dim_talker & 1) --rotary_dim_talker;

    auto t_layer_group_start = std::chrono::high_resolution_clock::now();
    int layer_group_interval = talker_cfg_.num_hidden_layers / 4; // report every 1/4 of layers
    for (int i = 0; i < talker_cfg_.num_hidden_layers; i++) {
        auto t_layer_start = std::chrono::high_resolution_clock::now();
        auto& L = talker_layers_[i];

        L.qkv_proj.forward(ctx, t_buf_.normed, t_buf_.qkv, total_prefill_len);
        ops::split_qkv(ctx, t_buf_.qkv, t_buf_.q, t_buf_.k, t_buf_.v, total_prefill_len, QD_talker, KVD_talker);

        ops::head_rms_norm(ctx, t_buf_.q, *L.q_norm_weight, talker_cfg_.rms_norm_eps, total_prefill_len, talker_cfg_.num_attention_heads, talker_cfg_.head_dim);
        ops::head_rms_norm(ctx, t_buf_.k, *L.k_norm_weight, talker_cfg_.rms_norm_eps, total_prefill_len, talker_cfg_.num_key_value_heads, talker_cfg_.head_dim);

        ops::apply_rope_partial(ctx, t_buf_.q, t_buf_.k, total_prefill_len, 0,
                                talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads,
                                talker_cfg_.head_dim, rotary_dim_talker, talker_cfg_.rope_theta);

        ops::copy_to_cache(ctx, t_buf_.k, talker_kv_cache_.k_cache(i), total_prefill_len, 0,
                           talker_cfg_.num_key_value_heads, talker_cfg_.head_dim, talker_kv_cache_.max_length());
        ops::copy_to_cache(ctx, t_buf_.v, talker_kv_cache_.v_cache(i), total_prefill_len, 0,
                           talker_cfg_.num_key_value_heads, talker_cfg_.head_dim, talker_kv_cache_.max_length());

        ops::attention_prefill(ctx, t_buf_.q, t_buf_.k, t_buf_.v,
                               t_buf_.attn_out, t_buf_.scores, total_prefill_len,
                               talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads, talker_cfg_.head_dim);

        L.o_proj.forward(ctx, t_buf_.attn_out, t_buf_.gate, total_prefill_len);

        ops::fused_add_rms_norm(ctx, t_buf_.hidden, t_buf_.gate, *L.post_attn_ln_weight, talker_cfg_.rms_norm_eps, t_buf_.normed, total_prefill_len, H_talker);

        L.gate_up_proj.forward(ctx, t_buf_.normed, t_buf_.gate_up, total_prefill_len);
        ops::split_gate_up(ctx, t_buf_.gate_up, t_buf_.gate, t_buf_.up, total_prefill_len, FF_talker);
        ops::swiglu(ctx, t_buf_.gate, t_buf_.up, t_buf_.gate, total_prefill_len * FF_talker);

        L.down_proj.forward(ctx, t_buf_.gate, t_buf_.attn_out, total_prefill_len);

        Tensor* next_input_ln = (i < talker_cfg_.num_hidden_layers - 1) ? talker_layers_[i + 1].input_ln_weight : talker_final_norm_weight_;
        ops::fused_add_rms_norm(ctx, t_buf_.hidden, t_buf_.attn_out, *next_input_ln, talker_cfg_.rms_norm_eps, t_buf_.normed, total_prefill_len, H_talker);

        // Report per-layer-group timing
        if ((i > 0 && (i + 1) % layer_group_interval == 0) || i == talker_cfg_.num_hidden_layers - 1) {
            double group_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_layer_group_start).count();
            int group_start = std::max(0, ((i + 1) / layer_group_interval - 1) * layer_group_interval);
            if (tts_profile) AILA_LOG_INFO("[TTS-Profile]     Layers [%d-%d]: %.1f ms (avg %.1f ms/layer)",
                          group_start, i, group_ms, group_ms / (i - group_start + 1));
            t_layer_group_start = std::chrono::high_resolution_clock::now();
        }
    }

    auto t_talker_fwd_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_talker_fwd_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Talker forward (%d layers): %.1f ms (avg %.1f ms/layer)",
                  talker_cfg_.num_hidden_layers, t_talker_fwd_ms, t_talker_fwd_ms / talker_cfg_.num_hidden_layers);

    // Final prefill step to get logits from the last position
    Tensor final_hidden = Tensor::allocate(ctx, {1, H_talker});
    {
        bf16* src = t_buf_.hidden.data_as<bf16>();
        bf16* dst = final_hidden.data_as<bf16>();
        ctx.queue().memcpy(dst, src + (total_prefill_len - 1) * H_talker, H_talker * sizeof(bf16)).wait();
    }

    // Final Norm
    Tensor final_normed = Tensor::allocate(ctx, {1, H_talker});
    ops::rms_norm(ctx, final_hidden, *talker_final_norm_weight_, talker_cfg_.rms_norm_eps, final_normed, 1, H_talker);

    // Logits
    talker_codec_head_.forward(ctx, final_normed, t_buf_.logits, 1);

    // 采样得到首码 (codebook 0)
    int first_token = ops::sample_with_config(ctx, t_buf_.logits, talker_cfg_.vocab_size, gen_config, {});
    talker_kv_cache_.advance(total_prefill_len);
    current_talker_len_ = total_prefill_len;

    // TTS autoregressive generation is prone to degenerate loops without repetition penalty.
    // Use a higher default if the user hasn't explicitly set one.
    GenerationConfig tts_gen = gen_config;
    if (tts_gen.repetition_penalty == 1.0f) {
        tts_gen.repetition_penalty = 1.1f;
    }

    // Track generated CB0 tokens for repetition penalty
    std::vector<int> generated_cb0_tokens;
    generated_cb0_tokens.reserve(tts_gen.max_new_tokens);

    // ------------------------------------------------------------------------
    // 2. 自回归 Decode 循环
    // ------------------------------------------------------------------------
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    double t_prefill_ms = std::chrono::duration<double, std::milli>(t_decode_start - t_prefill_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile] Prefill: %.1f ms", t_prefill_ms);

    int token = first_token;
    int eos_id = talker_cfg_.codec_eos_token_id; // 2150
    if (first_token != eos_id) {
        generated_cb0_tokens.push_back(first_token);
    }
    int gen_step = 0;

    Tensor past_hidden_talker = Tensor::allocate(ctx, {1, H_talker});
    ctx.queue().memcpy(past_hidden_talker.data(), final_normed.data(), H_talker * sizeof(bf16)).wait();
    // 临时张量用于拼接 Code Predictor 的输入
    Tensor pred_input = Tensor::allocate(ctx, {2, H_talker});
    Tensor token_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
    Tensor frame_codes_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::s32);
    Tensor last_id_hidden = Tensor::allocate(ctx, {1, H_talker});
    Tensor predictor_final_hidden = Tensor::allocate(ctx, {1, H_pred});
    Tensor predictor_final_normed = Tensor::allocate(ctx, {1, H_pred});
    Tensor predictor_emb_h = Tensor::allocate(ctx, {1, H_talker});
    Tensor predictor_emb_pred = Tensor::allocate(ctx, {1, H_pred});
    Tensor predictor_step_normed = Tensor::allocate(ctx, {1, H_pred});
    Tensor sum_emb = Tensor::allocate(ctx, {1, H_talker});
    Tensor single_emb = Tensor::allocate(ctx, {1, H_talker});
    Tensor step_normed_talker = Tensor::allocate(ctx, {1, H_talker});
    Tensor final_normed_talker = Tensor::allocate(ctx, {1, H_talker});

    int max_tokens = tts_gen.max_new_tokens;
    std::vector<int> token_upload_storage(static_cast<size_t>(std::max(1, max_tokens) * 16 + 16));
    size_t token_upload_index = 0;
    auto upload_token_async = [&](int value) {
        if (token_upload_index < token_upload_storage.size()) {
            token_upload_storage[token_upload_index] = value;
            ctx.queue().memcpy(token_dev.data(), &token_upload_storage[token_upload_index], sizeof(int));
            ++token_upload_index;
        } else {
            ctx.memcpy_h2d(token_dev.data(), &value, sizeof(int));
        }
    };

    out_codes.reserve(max_tokens * 16);
    const int callback_batch_frames = std::max(1, frame_callback_batch_frames);
    const bool playback_aware_steady_batch =
        aila::env::read_flag("AILA_TTS_PLAYBACK_AWARE_STEADY_BATCH", false);
    const int default_steady_callback_batch_frames =
        std::min(24, std::max(callback_batch_frames, 8));
    const int steady_callback_batch_frames = playback_aware_steady_batch
        ? std::clamp(
              aila::env::read_int_raw("AILA_TTS_STEADY_STREAM_BATCH_FRAMES",
                                       default_steady_callback_batch_frames),
              callback_batch_frames,
              24)
        : callback_batch_frames;
    const int default_initial_callback_batch_frames = std::clamp(
        aila::env::read_int_raw("AILA_TTS_STREAM_BATCH_FRAMES", callback_batch_frames),
        1,
        callback_batch_frames);
    const int initial_callback_batch_frames = std::clamp(
        aila::env::read_int_raw("AILA_TTS_INITIAL_STREAM_BATCH_FRAMES",
                                default_initial_callback_batch_frames),
        1,
        callback_batch_frames);
    const double playback_gap_trigger_ms = static_cast<double>(std::max(
        0,
        aila::env::read_int_raw("AILA_TTS_PLAYBACK_GAP_TRIGGER_MS", 0)));
    last_tts_timing_.stream_batch_frames = callback_batch_frames;
    last_tts_timing_.initial_stream_batch_frames = initial_callback_batch_frames;
    last_tts_timing_.steady_stream_batch_frames = steady_callback_batch_frames;
    last_tts_timing_.playback_aware_steady_batch =
        playback_aware_steady_batch ? 1 : 0;

    int callback_batches_emitted = 0;
    std::vector<int32_t> pending_callback_codes;
    int pending_callback_frames = 0;
    bool use_steady_callback_batch = false;
    bool last_callback_time_valid = false;
    int last_callback_frames = 0;
    double playback_gap_debt_ms = 0.0;
    auto last_callback_time = std::chrono::high_resolution_clock::time_point{};
    if (frame_callback) {
        pending_callback_codes.reserve(static_cast<size_t>(steady_callback_batch_frames) * 16);
    }
    auto flush_frame_callback = [&](bool used_steady_batch) -> bool {
        if (!frame_callback || pending_callback_frames <= 0) {
            return true;
        }
        const int flushed_frames = pending_callback_frames;
        const bool ok = frame_callback(pending_callback_codes, pending_callback_frames);
        const auto callback_time = std::chrono::high_resolution_clock::now();
        if (used_steady_batch) {
            ++last_tts_timing_.steady_batch_callback_count;
        }
        ++callback_batches_emitted;
        pending_callback_codes.clear();
        pending_callback_frames = 0;
        if (playback_aware_steady_batch && ok) {
            if (last_callback_time_valid) {
                const double interval_ms = std::chrono::duration<double, std::milli>(
                                               callback_time - last_callback_time).count();
                const double previous_audio_ms =
                    static_cast<double>(last_callback_frames * kMimiSamplesPerFrame) *
                    1000.0 / 24000.0;
                playback_gap_debt_ms = std::max(
                    0.0,
                    playback_gap_debt_ms + interval_ms - previous_audio_ms);
                use_steady_callback_batch =
                    steady_callback_batch_frames > callback_batch_frames &&
                    playback_gap_debt_ms > playback_gap_trigger_ms;
            }
            last_callback_time = callback_time;
            last_callback_frames = flushed_frames;
            last_callback_time_valid = true;
        }
        return ok;
    };
    auto callback_threshold = [&]() {
        if (callback_batches_emitted == 0) {
            return initial_callback_batch_frames;
        }
        if (playback_aware_steady_batch && use_steady_callback_batch) {
            return steady_callback_batch_frames;
        }
        return callback_batch_frames;
    };

    while (gen_step < max_tokens) {
        if (cancelled()) {
            return false;
        }
        if (token == eos_id) {
            break;
        }

        // 收集这帧 codebook 0
        std::array<int, 16> frame_codes{};
        frame_codes[0] = token;

        // 获取首码 embedding
        upload_token_async(token);
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, token_dev.data_as<int>(), 1, last_id_hidden, H_talker);

        // 拼接 past_hidden_talker 与 last_id_hidden -> pred_input [2, H_talker]
        {
            bf16* dst = pred_input.data_as<bf16>();
            bf16* src_past = past_hidden_talker.data_as<bf16>();
            bf16* src_last = last_id_hidden.data_as<bf16>();
            ctx.queue().memcpy(dst, src_past, H_talker * sizeof(bf16));
            ctx.queue().memcpy(dst + H_talker, src_last, H_talker * sizeof(bf16));
        }

        // 输入到 Code Predictor
        if (has_predictor_projection_) {
            predictor_projection_linear_.forward_bias(ctx, pred_input, *predictor_projection_bias_, p_buf_.pred_input_proj, 2);
        } else {
            ctx.queue().memcpy(p_buf_.pred_input_proj.data(), pred_input.data(), 2 * H_pred * sizeof(bf16));
        }

        // ==========================================
        // 运行 Code Predictor 自回归生成其余 15 个 codes
        // ==========================================
        predictor_kv_cache_.reset();

        // --- Predictor Prefill (seq_len = 2) ---
        ctx.queue().memcpy(p_buf_.hidden.data(), p_buf_.pred_input_proj.data(), 2 * H_pred * sizeof(bf16));
        ops::rms_norm(ctx, p_buf_.hidden, *predictor_layers_[0].input_ln_weight,
                      predictor_cfg_.rms_norm_eps, p_buf_.normed, 2, H_pred);

        int rotary_dim_pred = predictor_cfg_.head_dim;
        if (rotary_dim_pred & 1) --rotary_dim_pred;

        for (int i = 0; i < predictor_cfg_.num_hidden_layers; i++) {
            if (cancelled()) {
                return false;
            }
            auto& L = predictor_layers_[i];
            L.qkv_proj.forward(ctx, p_buf_.normed, p_buf_.qkv, 2);
            ops::split_qkv(ctx, p_buf_.qkv, p_buf_.q, p_buf_.k, p_buf_.v, 2, QD_pred, KVD_pred);

            ops::head_rms_norm(ctx, p_buf_.q, *L.q_norm_weight, predictor_cfg_.rms_norm_eps, 2, predictor_cfg_.num_attention_heads, predictor_cfg_.head_dim);
            ops::head_rms_norm(ctx, p_buf_.k, *L.k_norm_weight, predictor_cfg_.rms_norm_eps, 2, predictor_cfg_.num_key_value_heads, predictor_cfg_.head_dim);

            ops::apply_rope_partial(ctx, p_buf_.q, p_buf_.k, 2, 0,
                                    predictor_cfg_.num_attention_heads, predictor_cfg_.num_key_value_heads,
                                    predictor_cfg_.head_dim, rotary_dim_pred, predictor_cfg_.rope_theta);

            ops::copy_to_cache(ctx, p_buf_.k, predictor_kv_cache_.k_cache(i), 2, 0,
                               predictor_cfg_.num_key_value_heads, predictor_cfg_.head_dim, predictor_kv_cache_.max_length());
            ops::copy_to_cache(ctx, p_buf_.v, predictor_kv_cache_.v_cache(i), 2, 0,
                               predictor_cfg_.num_key_value_heads, predictor_cfg_.head_dim, predictor_kv_cache_.max_length());

            ops::attention_prefill(ctx, p_buf_.q, p_buf_.k, p_buf_.v,
                                   p_buf_.attn_out, p_buf_.scores, 2,
                                   predictor_cfg_.num_attention_heads, predictor_cfg_.num_key_value_heads, predictor_cfg_.head_dim);

            L.o_proj.forward(ctx, p_buf_.attn_out, p_buf_.gate, 2);
            ops::fused_add_rms_norm(ctx, p_buf_.hidden, p_buf_.gate, *L.post_attn_ln_weight, predictor_cfg_.rms_norm_eps, p_buf_.normed, 2, H_pred);

            L.gate_up_proj.forward(ctx, p_buf_.normed, p_buf_.gate_up, 2);
            ops::split_gate_up(ctx, p_buf_.gate_up, p_buf_.gate, p_buf_.up, 2, FF_pred);
            ops::swiglu(ctx, p_buf_.gate, p_buf_.up, p_buf_.gate, 2 * FF_pred);

            L.down_proj.forward(ctx, p_buf_.gate, p_buf_.attn_out, 2);
            Tensor* next_input_ln = (i < predictor_cfg_.num_hidden_layers - 1) ? predictor_layers_[i + 1].input_ln_weight : predictor_final_norm_weight_;
            ops::fused_add_rms_norm(ctx, p_buf_.hidden, p_buf_.attn_out, *next_input_ln, predictor_cfg_.rms_norm_eps, p_buf_.normed, 2, H_pred);
        }

        // Get logits for codebook 1 (cb_idx = 0)
        ctx.queue().memcpy(predictor_final_hidden.data(), p_buf_.hidden.data_as<bf16>() + 1 * H_pred, H_pred * sizeof(bf16));
        
        ops::rms_norm(ctx, predictor_final_hidden, *predictor_final_norm_weight_, predictor_cfg_.rms_norm_eps, predictor_final_normed, 1, H_pred);

        predictor_lm_heads_[0].forward(ctx, predictor_final_normed, p_buf_.logits, 1);
        int tok = ops::sample_with_config(ctx, p_buf_.logits, predictor_cfg_.vocab_size, gen_config, {});
        frame_codes[1] = tok;

        predictor_kv_cache_.advance(2);

        // --- Predictor Decode loop (step 2 to 15, i.e., cb_idx = 1 to 14) ---
        for (int cb_idx = 1; cb_idx < 15; cb_idx++) {
            if (cancelled()) {
                return false;
            }
            upload_token_async(tok);
            ops::embedding_lookup(ctx, *predictor_embed_weights_[cb_idx - 1], token_dev.data_as<int>(), 1, predictor_emb_h, H_talker);

            if (has_predictor_projection_) {
                predictor_projection_linear_.forward_bias(ctx, predictor_emb_h, *predictor_projection_bias_, predictor_emb_pred, 1);
            } else {
                ctx.queue().memcpy(predictor_emb_pred.data(), predictor_emb_h.data(), H_pred * sizeof(bf16));
            }

            // Copy to single slot in p_buf_.hidden at position cb_idx + 1 (since prefill took slots 0, 1)
            ctx.queue().memcpy(p_buf_.hidden.data_as<bf16>() + (cb_idx + 1) * H_pred, predictor_emb_pred.data(), H_pred * sizeof(bf16));
            
            // Norm
            ops::rms_norm(ctx, predictor_emb_pred, *predictor_layers_[0].input_ln_weight,
                          predictor_cfg_.rms_norm_eps, predictor_step_normed, 1, H_pred);

            // Forward layers with decode path (seq_len = 1)
            for (int i = 0; i < predictor_cfg_.num_hidden_layers; i++) {
                if (cancelled()) {
                    return false;
                }
                auto& L = predictor_layers_[i];
                L.qkv_proj.forward(ctx, predictor_step_normed, p_buf_.qkv, 1);

                bf16* qkv_ptr = p_buf_.qkv.data_as<bf16>();
                Tensor q_dec = Tensor::view(ctx, qkv_ptr, {1, QD_pred});
                Tensor k_dec = Tensor::view(ctx, qkv_ptr + QD_pred, {1, KVD_pred});
                Tensor v_dec = Tensor::view(ctx, qkv_ptr + QD_pred + KVD_pred, {1, KVD_pred});

                ops::decode_prepare_qkv_partial(ctx,
                    q_dec, k_dec, v_dec,
                    *L.q_norm_weight, *L.k_norm_weight,
                    predictor_kv_cache_.k_cache(i), predictor_kv_cache_.v_cache(i),
                    predictor_kv_cache_.current_length(),
                    predictor_cfg_.num_attention_heads, predictor_cfg_.num_key_value_heads, predictor_cfg_.head_dim,
                    predictor_cfg_.rms_norm_eps, rotary_dim_pred, predictor_cfg_.rope_theta);

                ops::attention_decode(ctx, q_dec,
                                      predictor_kv_cache_.k_cache(i), predictor_kv_cache_.v_cache(i),
                                      p_buf_.attn_out, p_buf_.decode_scores,
                                      predictor_cfg_.num_attention_heads, predictor_cfg_.num_key_value_heads,
                                      predictor_cfg_.head_dim, predictor_kv_cache_.current_length() + 1,
                                      &p_buf_.decode_attn_partials);

                L.o_proj.forward(ctx, p_buf_.attn_out, p_buf_.gate, 1);
                ops::fused_add_rms_norm(ctx, predictor_emb_pred, p_buf_.gate, *L.post_attn_ln_weight, predictor_cfg_.rms_norm_eps, predictor_step_normed, 1, H_pred);

                L.gate_up_proj.forward(ctx, predictor_step_normed, p_buf_.gate_up, 1);
                ops::fused_gate_up_swiglu(ctx, p_buf_.gate_up, p_buf_.gate, FF_pred);

                L.down_proj.forward(ctx, p_buf_.gate, p_buf_.attn_out, 1);
                Tensor* next_input_ln = (i < predictor_cfg_.num_hidden_layers - 1) ? predictor_layers_[i + 1].input_ln_weight : predictor_final_norm_weight_;
                ops::fused_add_rms_norm(ctx, predictor_emb_pred, p_buf_.attn_out, *next_input_ln, predictor_cfg_.rms_norm_eps, predictor_step_normed, 1, H_pred);
            }
            ctx.queue().memcpy(p_buf_.hidden.data_as<bf16>() + (cb_idx + 1) * H_pred, predictor_emb_pred.data(), H_pred * sizeof(bf16));

            // Compute logits and sample tok
            ops::rms_norm(ctx, predictor_emb_pred, *predictor_final_norm_weight_, predictor_cfg_.rms_norm_eps, predictor_final_normed, 1, H_pred);
            predictor_lm_heads_[cb_idx].forward(ctx, predictor_final_normed, p_buf_.logits, 1);
            
            tok = ops::sample_with_config(ctx, p_buf_.logits, predictor_cfg_.vocab_size, tts_gen, {});
            frame_codes[cb_idx + 1] = tok;

            predictor_kv_cache_.advance(1);
        }

        // 保存这帧的 16 个 codes
        for (int c : frame_codes) {
            out_codes.push_back(static_cast<int32_t>(c));
            if (frame_callback) {
                pending_callback_codes.push_back(static_cast<int32_t>(c));
            }
        }
        out_n_frames++;
        if (frame_callback) {
            ++pending_callback_frames;
            const int threshold = callback_threshold();
            const bool using_steady_threshold =
                playback_aware_steady_batch &&
                threshold == steady_callback_batch_frames &&
                steady_callback_batch_frames > callback_batch_frames;
            if (pending_callback_frames >= threshold &&
                !flush_frame_callback(using_steady_threshold)) {
                return false;
            }
        }

        // ==========================================
        // 运行 Talker Decode (seq_len = 1) 并生成下一个首码
        // ==========================================
        
        // 查找 16 个 codes 的 embeddings
        // 查找 codebook 0
        ctx.queue().memcpy(frame_codes_dev.data(), frame_codes.data(), frame_codes.size() * sizeof(int));
        int* frame_codes_ptr = frame_codes_dev.data_as<int>();
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, frame_codes_ptr, 1, sum_emb, H_talker);

        // 查找 predictor 对应的 15 个 embeddings 并累加
        for (int i = 0; i < 15; i++) {
            ops::embedding_lookup(ctx, *predictor_embed_weights_[i], frame_codes_ptr + i + 1, 1, single_emb, H_talker);

            // 累加：sum_emb += single_emb
            bf16* sum_ptr = sum_emb.data_as<bf16>();
            bf16* sgl_ptr = single_emb.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> idx) {
                sum_ptr[idx[0]] = sum_ptr[idx[0]] + sgl_ptr[idx[0]];
            });
        }

        // 加上 trailing_text_hidden[gen_step] 或 tts_pad_embed（对齐 ggml）
        bf16* sum_ptr = sum_emb.data_as<bf16>();
        bf16* add_ptr = gen_step < trailing_len
            ? trailing_text_hidden.data_as<bf16>() + gen_step * H_talker
            : tts_pad_embed.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> idx) {
            sum_ptr[idx[0]] = sum_ptr[idx[0]] + add_ptr[idx[0]];
        });

        // 运行 Talker Decode Step (seq_len = 1)
        ops::rms_norm(ctx, sum_emb, *talker_layers_[0].input_ln_weight,
                      talker_cfg_.rms_norm_eps, step_normed_talker, 1, H_talker);

        int current_pos = current_talker_len_;

        for (int i = 0; i < talker_cfg_.num_hidden_layers; i++) {
            if (cancelled()) {
                return false;
            }
            auto& L = talker_layers_[i];
            L.qkv_proj.forward(ctx, step_normed_talker, t_buf_.qkv, 1);

            bf16* qkv_ptr = t_buf_.qkv.data_as<bf16>();
            Tensor q_dec = Tensor::view(ctx, qkv_ptr, {1, QD_talker});
            Tensor k_dec = Tensor::view(ctx, qkv_ptr + QD_talker, {1, KVD_talker});
            Tensor v_dec = Tensor::view(ctx, qkv_ptr + QD_talker + KVD_talker, {1, KVD_talker});

            ops::decode_prepare_qkv_partial(ctx,
                q_dec, k_dec, v_dec,
                *L.q_norm_weight, *L.k_norm_weight,
                talker_kv_cache_.k_cache(i), talker_kv_cache_.v_cache(i),
                current_pos,
                talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads, talker_cfg_.head_dim,
                talker_cfg_.rms_norm_eps, rotary_dim_talker, talker_cfg_.rope_theta);

            ops::attention_decode(ctx, q_dec,
                                  talker_kv_cache_.k_cache(i), talker_kv_cache_.v_cache(i),
                                  t_buf_.attn_out, t_buf_.decode_scores,
                                  talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads,
                                  talker_cfg_.head_dim, current_pos + 1,
                                  &t_buf_.decode_attn_partials);

            L.o_proj.forward(ctx, t_buf_.attn_out, t_buf_.gate, 1);
            ops::fused_add_rms_norm(ctx, sum_emb, t_buf_.gate, *L.post_attn_ln_weight, talker_cfg_.rms_norm_eps, step_normed_talker, 1, H_talker);

            L.gate_up_proj.forward(ctx, step_normed_talker, t_buf_.gate_up, 1);
            ops::fused_gate_up_swiglu(ctx, t_buf_.gate_up, t_buf_.gate, FF_talker);

            L.down_proj.forward(ctx, t_buf_.gate, t_buf_.attn_out, 1);
            Tensor* next_input_ln = (i < talker_cfg_.num_hidden_layers - 1) ? talker_layers_[i + 1].input_ln_weight : talker_final_norm_weight_;
            ops::fused_add_rms_norm(ctx, sum_emb, t_buf_.attn_out, *next_input_ln, talker_cfg_.rms_norm_eps, step_normed_talker, 1, H_talker);
        }

        // 预测下一个首码
        ops::rms_norm(ctx, sum_emb, *talker_final_norm_weight_, talker_cfg_.rms_norm_eps, final_normed_talker, 1, H_talker);

        // 保存新的 past_hidden_talker (保存归一化后的值以对齐 Python)
        ctx.queue().memcpy(past_hidden_talker.data(), final_normed_talker.data(), H_talker * sizeof(bf16));

        talker_codec_head_.forward(ctx, final_normed_talker, t_buf_.logits, 1);
        token = ops::sample_with_config(ctx, t_buf_.logits, talker_cfg_.vocab_size, tts_gen, generated_cb0_tokens);

        if (token != eos_id) {
            generated_cb0_tokens.push_back(token);
        }
        talker_kv_cache_.advance(1);
        current_talker_len_++;
        gen_step++;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double t_decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    double t_total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_total_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile] Decode: %.1f ms (%d steps, avg %.1f ms/step)",
                  t_decode_ms, out_n_frames, t_decode_ms / std::max(1, out_n_frames));
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile] Talker+CodePredictor total: %.1f ms", t_total_ms);

    if (!flush_frame_callback(false)) {
        return false;
    }

    return true;
}

bool Qwen3TTSBackend::synthesize_codes_stream(Context& ctx,
    const std::vector<int>& text_tokens,
    const std::vector<float>& speaker_embedding,
    int speaker_id,
    const std::vector<int>& instruct_tokens,
    int language_id,
    const GenerationConfig& gen_config,
    int stream_batch_frames,
    AudioChunkCallback audio_callback,
    std::function<bool()> should_cancel) {
    auto cancelled = [&]() {
        return should_cancel && should_cancel();
    };
    if (cancelled()) {
        return false;
    }

    last_tts_timing_ = TtsBackendTiming{};
    const auto timing_start = std::chrono::high_resolution_clock::now();

    // Initialize Mimi before codec generation so the first generated frame batch
    // can be decoded immediately instead of waiting for the whole utterance.
    MimiStreamState mimi_state;
    const int max_stream_frames = std::max(128, gen_config.max_new_tokens + 16);
    if (!init_mimi_stream(ctx, mimi_state, max_stream_frames)) {
        return false;
    }
    const auto codes_start = std::chrono::high_resolution_clock::now();
    last_tts_timing_.mimi_init_ms =
        std::chrono::duration<double, std::milli>(codes_start - timing_start).count();

    auto emit_audio_batch = [&](const std::vector<int32_t>& batch_codes,
                                int batch_frames) -> bool {
        if (cancelled()) {
            return false;
        }
        if (last_tts_timing_.total_frames == 0) {
            last_tts_timing_.codes_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - codes_start).count();
        }
        last_tts_timing_.total_frames += batch_frames;

        std::vector<float> audio_chunk;
        if (!decode_mimi_incremental(ctx, batch_codes, batch_frames, mimi_state, audio_chunk)) {
            return false;
        }
        if (!audio_chunk.empty()) {
            if (cancelled()) {
                return false;
            }
            if (last_tts_timing_.callback_count == 0) {
                last_tts_timing_.first_audio_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - timing_start).count();
                last_tts_timing_.first_audio_samples =
                    static_cast<int>(audio_chunk.size());
            }
            ++last_tts_timing_.callback_count;
            audio_callback(audio_chunk);
        }
        return true;
    };

    std::vector<int32_t> all_codes;
    int total_frames = 0;
    const int batch_size = std::max(1, stream_batch_frames);
    if (!synthesize_codes(ctx, text_tokens, speaker_embedding, speaker_id,
                           instruct_tokens, language_id, gen_config,
                           all_codes, total_frames, should_cancel,
                           emit_audio_batch, batch_size)) {
        return false;
    }
    const auto codes_done = std::chrono::high_resolution_clock::now();
    if (last_tts_timing_.codes_ms < 0.0) {
        last_tts_timing_.codes_ms =
            std::chrono::duration<double, std::milli>(codes_done - codes_start).count();
    }
    last_tts_timing_.total_frames = total_frames;

    if (cancelled()) {
        return false;
    }
    std::vector<float> flush_samples;
    decode_mimi_flush(ctx, mimi_state, flush_samples);
    if (!flush_samples.empty()) {
        if (cancelled()) {
            return false;
        }
        if (last_tts_timing_.callback_count == 0) {
            last_tts_timing_.first_audio_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - timing_start).count();
            last_tts_timing_.first_audio_samples =
                static_cast<int>(flush_samples.size());
        }
        ++last_tts_timing_.callback_count;
        audio_callback(flush_samples);
    }

    last_tts_timing_.total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - timing_start).count();
    return true;
}

bool Qwen3TTSBackend::synthesize_tts_stream(
    Context& ctx,
    const std::vector<int>& text_tokens,
    const GenerationConfig& gen_config,
    int stream_batch_frames,
    std::function<void(const std::vector<float>&)> audio_callback,
    std::string* error_message,
    std::function<bool()> should_cancel) {
    if (text_tokens.empty()) {
        if (error_message) {
            *error_message = "TTS text encoded to zero tokens";
        }
        return false;
    }
    if (!audio_callback) {
        if (error_message) {
            *error_message = "TTS audio callback is empty";
        }
        return false;
    }
    if (should_cancel && should_cancel()) {
        if (error_message) {
            *error_message = "Qwen3-TTS streaming synthesis cancelled";
        }
        return false;
    }

    const int batch_frames = std::max(1, stream_batch_frames);
    if (!synthesize_codes_stream(ctx, text_tokens, {}, 0, {}, 0, gen_config,
                                 batch_frames, std::move(audio_callback),
                                 should_cancel)) {
        if (error_message) {
            *error_message = (should_cancel && should_cancel())
                ? "Qwen3-TTS streaming synthesis cancelled"
                : "Qwen3-TTS streaming synthesis failed";
        }
        return false;
    }
    if (error_message) {
        error_message->clear();
    }
    return true;
}

bool Qwen3TTSBackend::load_mimi_vocoder(Context& ctx, const std::string& model_dir, std::string* error_message) {
    if (mimi_loaded_) return true;

    std::string safetensors_path = model_dir + "/speech_tokenizer/model.safetensors";
    try {
        AILA_LOG_INFO("[MimiLoader] Loading safetensors from: %s", safetensors_path.c_str());
        mimi_weights_ = LoadSafetensors(safetensors_path, ctx);
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Failed to load mimi safetensors: ") + e.what();
        return false;
    }

    // VQ Codebooks GPU 归一化辅助函数，计算完毕后直接生成归一化后的 bf16 权重并替换原 f32 权重
    auto normalize_codebook_gpu = [&](const std::string& emb_name, Tensor& cluster_usage) {
        Tensor& embed_sum = mimi_weights_.get(emb_name);
        bool emb_is_f32 = (embed_sum.dtype() == dnnl::memory::data_type::f32);
        bool usage_is_f32 = (cluster_usage.dtype() == dnnl::memory::data_type::f32);
        
        const void* emb_raw = embed_sum.data();
        const void* usage_raw = cluster_usage.data();
        
        int64_t codebook_size = embed_sum.shape(0);
        int64_t codebook_dim = embed_sum.shape(1);

        // 分配一个新 bf16 Tensor
        Tensor norm_emb_bf16 = Tensor::allocate(ctx, embed_sum.shape(), dnnl::memory::data_type::bf16);
        bf16* dst_ptr = norm_emb_bf16.data_as<bf16>();

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(codebook_size, codebook_dim), [=](sycl::id<2> idx) {
                int64_t i = idx[0];
                int64_t d = idx[1];

                float u = usage_is_f32 ? 
                    static_cast<const float*>(usage_raw)[i] : 
                    static_cast<float>(static_cast<const bf16*>(usage_raw)[i]);
                
                if (u < 1e-5f) u = 1e-5f;

                int64_t offset = i * codebook_dim + d;
                float val = emb_is_f32 ? 
                    static_cast<const float*>(emb_raw)[offset] : 
                    static_cast<float>(static_cast<const bf16*>(emb_raw)[offset]);

                float norm_val = val / u;
                dst_ptr[offset] = bf16(norm_val);
            });
        });
        ctx.queue().wait();

        // 替换为转换后的 bf16
        mimi_weights_.replace(emb_name, std::move(norm_emb_bf16));
    };

    // 1. 归一化 first codebook
    if (mimi_weights_.has("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum") &&
        mimi_weights_.has("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")) {
        Tensor& usage = mimi_weights_.get("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage");
        normalize_codebook_gpu("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum", usage);
    } else {
        if (error_message) *error_message = "Mimi weights missing first codebook tensors";
        return false;
    }

    // 2. 归一化 rest codebooks
    for (int c = 0; c < 15; ++c) {
        std::string emb_name = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(c) + "._codebook.embedding_sum";
        std::string usage_name = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(c) + "._codebook.cluster_usage";
        if (mimi_weights_.has(emb_name) && mimi_weights_.has(usage_name)) {
            Tensor& usage = mimi_weights_.get(usage_name);
            normalize_codebook_gpu(emb_name, usage);
        } else {
            if (error_message) *error_message = "Mimi weights missing rest codebook tensor at layer " + std::to_string(c);
            return false;
        }
    }

    // 3. 将除了 cluster_usage 之外的所有其它 f32 权重转换成 bf16
    AILA_LOG_INFO("[MimiLoader] Converting non-codebook f32 weights to bf16 on GPU...");
    std::vector<std::string> mimi_tensor_names = mimi_weights_.names();
    for (const auto& name : mimi_tensor_names) {
        // cluster_usage 字段不参与网络前向，跳过不转换
        if (name.find("_codebook.cluster_usage") != std::string::npos) {
            continue;
        }

        Tensor& t = mimi_weights_.get(name);
        if (t.dtype() == dnnl::memory::data_type::f32) {
            int64_t total = t.numel();
            Tensor t_bf16 = Tensor::allocate(ctx, t.shape(), dnnl::memory::data_type::bf16);
            
            float* src_ptr = t.data_as<float>();
            bf16* dst_ptr = t_bf16.data_as<bf16>();

            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                    int64_t i = idx[0];
                    if (i < total) {
                        dst_ptr[i] = bf16(src_ptr[i]);
                    }
                });
            });
            ctx.queue().wait();
            
            mimi_weights_.replace(name, std::move(t_bf16));
        }
    }

    ctx.synchronize();
    AILA_LOG_INFO("[MimiLoader] Loaded and converted all codebooks/weights successfully to bf16!");

    // Transpose conv1d weights from [out_ch, in_ch, kernel_size] to
    // [out_ch, kernel_size, in_ch] for vec8-compatible contiguous ic access.
    // Scan all loaded weights for 3D conv tensors and reorder in-place.
    {
        AILA_LOG_INFO("[MimiLoader] Transposing conv1d weights for vec8 access...");
        auto transpose_conv_weight = [&](const std::string& name) {
            if (!mimi_weights_.has(name)) return;
            Tensor& w = mimi_weights_.get(name);
            if (w.ndim() != 3) return;
            int OC = static_cast<int>(w.shape(0));
            int IC = static_cast<int>(w.shape(1));
            int KS = static_cast<int>(w.shape(2));
            if (KS <= 1) return; // k=1 doesn't benefit from transpose

            Tensor w_new = Tensor::allocate(ctx, {OC, KS, IC});
            auto* old_ptr = w.data_as<bf16>();
            auto* new_ptr = w_new.data_as<bf16>();

            // w_old[oc, ic, k] → w_new[oc, k, ic]
            int total = OC * IC * KS;
            ctx.queue().parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                int ic = i % IC;
                int tmp = i / IC;
                int k = tmp % KS;
                int oc = tmp / KS;
                int old_idx = (oc * IC + ic) * KS + k;
                int new_idx = (oc * KS + k) * IC + ic;
                new_ptr[new_idx] = old_ptr[old_idx];
            });
            ctx.queue().wait();
            mimi_weights_.replace(name, std::move(w_new));
        };

        // Pre-conv
        transpose_conv_weight("decoder.pre_conv.conv.weight");
        // Decoder blocks convs (kernel=7)
        for (int i = 1; i <= 4; ++i) {
            for (int r = 2; r <= 4; ++r) {
                transpose_conv_weight(
                    "decoder.decoder." + std::to_string(i) + ".block." +
                    std::to_string(r) + ".conv1.conv.weight");
            }
        }
        // dec0 and dec6 convs
        transpose_conv_weight("decoder.decoder.0.conv.weight");
        transpose_conv_weight("decoder.decoder.6.conv.weight");

        ctx.synchronize();
        AILA_LOG_INFO("[MimiLoader] Conv1d weight transpose complete");
    }

    if (aila::env::read_flag("AILA_TTS_MIMI_TRANSPOSE_CONV_VEC8", true)) {
        AILA_LOG_INFO("[MimiLoader] Transposing Mimi transpose-conv weights for vec8 access...");
        auto transpose_conv_transpose_weight = [&](const std::string& name) {
            if (!mimi_weights_.has(name)) return;
            Tensor& w = mimi_weights_.get(name);
            if (w.ndim() != 3) return;
            const int IC = static_cast<int>(w.shape(0));
            const int OC = static_cast<int>(w.shape(1));
            const int KS = static_cast<int>(w.shape(2));
            if (IC <= 0 || OC <= 0 || KS <= 0) return;

            Tensor w_new = Tensor::allocate(ctx, {OC, KS, IC});
            auto* old_ptr = w.data_as<bf16>();
            auto* new_ptr = w_new.data_as<bf16>();

            // w_old[ic, oc, k] -> w_new[oc, k, ic]
            const int total = IC * OC * KS;
            ctx.queue().parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                int k = i % KS;
                int tmp = i / KS;
                int oc = tmp % OC;
                int ic = tmp / OC;
                int new_idx = (oc * KS + k) * IC + ic;
                new_ptr[new_idx] = old_ptr[i];
            });
            ctx.queue().wait();
            mimi_weights_.replace(name, std::move(w_new));
        };

        for (int i = 0; i < 2; ++i) {
            transpose_conv_transpose_weight(
                "decoder.upsample." + std::to_string(i) + ".0.conv.weight");
        }
        for (int i = 1; i <= 4; ++i) {
            transpose_conv_transpose_weight(
                "decoder.decoder." + std::to_string(i) + ".block.1.conv.weight");
        }

        ctx.synchronize();
        AILA_LOG_INFO("[MimiLoader] Mimi transpose-conv weight transpose complete");
    }

    mimi_loaded_ = true;
    init_mimi_runtime_linears(ctx);

    // Warmup: run minimal mimi decode to trigger conv/attention JIT compilation
    {
        AILA_LOG_INFO("[TTS] Running Mimi vocoder warmup...");
        std::vector<int32_t> dummy_codes(4 * 16, 0); // 4 frames, 16 codebooks, all token 0
        std::vector<float> dummy_samples;
        decode_mimi_vocoder(ctx, dummy_codes, 4, dummy_samples);

        if (aila::env::read_flag("AILA_TTS_MIMI_STREAM_SHAPE_WARMUP", true)) {
            const std::array<int, 3> stream_warmup_frames = {17, 28, 32};
            for (int frames : stream_warmup_frames) {
                Tensor dummy_pre_tfm = Tensor::allocate(ctx, {frames, 1024});
                ctx.queue().memset(dummy_pre_tfm.data(), 0, dummy_pre_tfm.size_bytes()).wait();
                dummy_samples.clear();
                mimi_conv_stages(ctx, dummy_pre_tfm, frames, dummy_samples,
                                 frames * kMimiSamplesPerFrame);
            }
            AILA_LOG_INFO("[TTS] Mimi stream shape warmup complete (frames=17,28,32)");
        }

        AILA_LOG_INFO("[TTS] Mimi warmup complete");
    }

    return true;
}

void Qwen3TTSBackend::init_mimi_runtime_linears(Context& ctx) {
    if (mimi_runtime_linears_initialized_) return;

    mimi_first_proj_weight_view_ =
        mimi_weights_.get("decoder.quantizer.rvq_first.output_proj.weight").reshape_view({512, 256});
    mimi_rest_proj_weight_view_ =
        mimi_weights_.get("decoder.quantizer.rvq_rest.output_proj.weight").reshape_view({512, 256});
    mimi_first_proj_.init(ctx, mimi_first_proj_weight_view_, 256, 512, false);
    mimi_rest_proj_.init(ctx, mimi_rest_proj_weight_view_, 256, 512, false);

    mimi_pre_tfm_in_proj_.init(ctx,
        mimi_weights_.get("decoder.pre_transformer.input_proj.weight"), 1024, 512, false);
    mimi_pre_tfm_out_proj_.init(ctx,
        mimi_weights_.get("decoder.pre_transformer.output_proj.weight"), 512, 1024, false);

    for (int l = 0; l < 8; ++l) {
        const std::string layer_prefix = "decoder.pre_transformer.layers." + std::to_string(l) + ".";
        auto& layer = mimi_pre_tfm_linears_[static_cast<size_t>(l)];
        layer.q_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.q_proj.weight"), 512, 1024, false);
        layer.k_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.k_proj.weight"), 512, 1024, false);
        layer.v_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.v_proj.weight"), 512, 1024, false);
        layer.o_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.o_proj.weight"), 1024, 512, false);
        layer.gate_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.gate_proj.weight"), 512, 1024, false);
        layer.up_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.up_proj.weight"), 512, 1024, false);
        layer.down_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.down_proj.weight"), 1024, 512, false);
    }

    for (int i = 0; i < 2; ++i) {
        const std::string up_prefix = "decoder.upsample." + std::to_string(i) + ".1.";
        auto& layer = mimi_upsample_linears_[static_cast<size_t>(i)];
        layer.pwconv1.init(ctx, mimi_weights_.get(up_prefix + "pwconv1.weight"), 1024, 4096, false);
        layer.pwconv2.init(ctx, mimi_weights_.get(up_prefix + "pwconv2.weight"), 4096, 1024, false);
    }

    mimi_runtime_linears_initialized_ = true;
}

bool Qwen3TTSBackend::mimi_conv_stages(Context& ctx, Tensor& pre_tfm_out, int n_frames,
    std::vector<float>& out_samples, int tail_samples) {
    static const bool tts_profile = aila::env::read_flag("AILA_TTS_PROFILE", false);
    static const bool tts_alloc_profile = aila::env::read_flag("AILA_TTS_MIMI_ALLOC_PROFILE", false);
    static const bool tts_decoder_fused_conv2_residual =
        aila::env::read_flag("AILA_TTS_MIMI_DECODER_FUSED_CONV2_RESIDUAL", true);
    static const bool tts_decoder_block_profile =
        aila::env::read_flag("AILA_TTS_MIMI_DECODER_BLOCK_PROFILE", false);
    ScopedAllocationProfile alloc_profile(ctx, tts_alloc_profile, "Mimi conv stages");
    auto t_conv_start = std::chrono::high_resolution_clock::now();

    // 辅助 layer scale 核函数
    auto apply_layer_scale_gpu = [&](Tensor& x, Tensor& scale, int length, int channels) {
        auto* x_ptr = x.data_as<bf16>();
        auto* scale_ptr = scale.data_as<bf16>();
        int total = length * channels;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                if (i >= total) return;
                int c = i % channels;
                x_ptr[i] = bf16(static_cast<float>(x_ptr[i]) * static_cast<float>(scale_ptr[c]));
            });
        });
    };

    // ==========================================
    // 4. 两层 ConvNeXt 上采样 (总共上采样 4 倍)
    // ==========================================
    auto t_upsample_start = std::chrono::high_resolution_clock::now();
    Tensor upsample_in = Tensor::allocate(ctx, {n_frames, 1024});
    ops::copy_tensor(ctx, pre_tfm_out, upsample_in, n_frames * 1024);
    int L = n_frames;

    for (int i = 0; i < 2; ++i) {
        std::string up_prefix = "decoder.upsample." + std::to_string(i) + ".";

        // 转置卷积上采样 2 倍: kernel_size=2, stride=2, output_channels=1024
        Tensor conv_t_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& conv_w = mimi_weights_.get(up_prefix + "0.conv.weight");
        Tensor& conv_b = mimi_weights_.get(up_prefix + "0.conv.bias");
        ops::causal_conv_transpose1d(ctx, upsample_in, conv_w, conv_b, conv_t_out, 1, 1024, 1024, L, 2, 2);

        // ConvNeXt block
        Tensor dw_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& dw_w = mimi_weights_.get(up_prefix + "1.dwconv.conv.weight");
        Tensor& dw_b = mimi_weights_.get(up_prefix + "1.dwconv.conv.bias");
        ops::causal_conv1d_dw(ctx, conv_t_out, dw_w, dw_b, dw_out, 1, 1024, 2 * L, 7, 1);

        // LayerNorm
        Tensor normed_dw = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& norm_w = mimi_weights_.get(up_prefix + "1.norm.weight");
        Tensor& norm_b = mimi_weights_.get(up_prefix + "1.norm.bias");
        ops::layer_norm(ctx, dw_out, norm_w, norm_b, 1e-6f, normed_dw, 2 * L, 1024);

        // Linear pwconv1
        Tensor pw1_out = Tensor::allocate(ctx, {2 * L, 4096});
        Tensor& pw1_b = mimi_weights_.get(up_prefix + "1.pwconv1.bias");
        auto& up_linears = mimi_upsample_linears_[static_cast<size_t>(i)];
        up_linears.pwconv1.forward_bias(ctx, normed_dw, pw1_b, pw1_out, 2 * L);

        // GELU
        ops::gelu_tanh_inplace(ctx, pw1_out, 2 * L * 4096);

        // Linear pwconv2
        Tensor pw2_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& pw2_b = mimi_weights_.get(up_prefix + "1.pwconv2.bias");
        up_linears.pwconv2.forward_bias(ctx, pw1_out, pw2_b, pw2_out, 2 * L);

        // Scale by gamma and add to residual
        Tensor& gamma = mimi_weights_.get(up_prefix + "1.gamma");
        apply_layer_scale_gpu(pw2_out, gamma, 2 * L, 1024);

        ops::residual_add(ctx, conv_t_out, pw2_out, 2 * L * 1024);

        // upsample_in is about to be replaced, which frees the previous input.
        // Wait here so queued kernels no longer reference that storage.
        ctx.synchronize();

        L = 2 * L;
        upsample_in = std::move(conv_t_out);
    }

    auto t_upsample_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_upsample_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   ConvNeXt upsample (2 layers): %.1f ms", t_upsample_ms);

    // ==========================================
    // 5. Decoder Blocks (4 Blocks, 总共上采样 480 倍)
    // ==========================================
    auto t_decoder_start = std::chrono::high_resolution_clock::now();
    double decoder_dec0_ms = 0.0;
    double decoder_stage_act_ms = 0.0;
    double decoder_stage_transpose_ms = 0.0;
    double decoder_res_copy_ms = 0.0;
    double decoder_res_act1_ms = 0.0;
    double decoder_res_conv1_ms = 0.0;
    double decoder_res_act2_ms = 0.0;
    double decoder_res_conv2_ms = 0.0;
    double decoder_res_add_ms = 0.0;
    auto profile_step_start = std::chrono::high_resolution_clock::now();
    auto profile_step_done = [&](double& bucket) {
        if (!tts_decoder_block_profile) return;
        ctx.synchronize();
        const auto now = std::chrono::high_resolution_clock::now();
        bucket += std::chrono::duration<double, std::milli>(now - profile_step_start).count();
        profile_step_start = now;
    };

    // 先通过 dec0 一维卷积 [7, 1024, 1536]
    Tensor dec0_out = Tensor::allocate(ctx, {L, 1536});
    Tensor& dec0_w = mimi_weights_.get("decoder.decoder.0.conv.weight");
    Tensor& dec0_b = mimi_weights_.get("decoder.decoder.0.conv.bias");
    profile_step_start = std::chrono::high_resolution_clock::now();
    ops::causal_conv1d(ctx, upsample_in, dec0_w, dec0_b, dec0_out, 1, 1024, 1536, L, 7, 1);
    profile_step_done(decoder_dec0_ms);

    Tensor dec_in = std::move(dec0_out);

    int upsample_rates[4] = {8, 5, 4, 3};
    int in_dims[4] = {1536, 768, 384, 192};
    int out_dims[4] = {768, 384, 192, 96};

    for (int i = 0; i < 4; ++i) {
        int in_d = in_dims[i];
        int out_d = out_dims[i];
        int stride = upsample_rates[i];
        int kernel = 2 * stride;
        std::string dec_prefix = "decoder.decoder." + std::to_string(i + 1) + ".block.";

        // 1. SnakeBeta
        Tensor& snake_a = mimi_weights_.get(dec_prefix + "0.alpha");
        Tensor& snake_b = mimi_weights_.get(dec_prefix + "0.beta");
        profile_step_start = std::chrono::high_resolution_clock::now();
        ops::snake_beta(ctx, dec_in, snake_a, snake_b, dec_in, L * in_d, in_d, L);
        profile_step_done(decoder_stage_act_ms);

        // 2. Transposed Convolution
        Tensor conv_t_out = Tensor::allocate(ctx, {L * stride, out_d});
        Tensor& conv_t_w = mimi_weights_.get(dec_prefix + "1.conv.weight");
        Tensor& conv_t_b = mimi_weights_.get(dec_prefix + "1.conv.bias");
        profile_step_start = std::chrono::high_resolution_clock::now();
        ops::causal_conv_transpose1d(ctx, dec_in, conv_t_w, conv_t_b, conv_t_out, 1, in_d, out_d, L, kernel, stride);
        profile_step_done(decoder_stage_transpose_ms);

        // 3. 3 Residual blocks with dilations 1, 3, 9
        int dilations[3] = {1, 3, 9};
        int Ls = L * stride;
        int res_elems = Ls * out_d;
        Tensor xx = std::move(conv_t_out);

        // Pre-allocate residual block buffers (reused across 3 iterations)
        Tensor res_in  = Tensor::allocate(ctx, {Ls, out_d});
        Tensor xx_act1 = Tensor::allocate(ctx, {Ls, out_d});
        Tensor conv1_out = Tensor::allocate(ctx, {Ls, out_d});
        Tensor conv2_out;
        if (!tts_decoder_fused_conv2_residual) {
            conv2_out = Tensor::allocate(ctx, {Ls, out_d});
        }
        Tensor* xx_cur = &xx;
        Tensor* residual_buf = &res_in;

        for (int r = 0; r < 3; ++r) {
            std::string res_prefix = dec_prefix + std::to_string(r + 2) + ".";

            profile_step_start = std::chrono::high_resolution_clock::now();
            ops::copy_tensor(ctx, *xx_cur, *residual_buf, res_elems);
            profile_step_done(decoder_res_copy_ms);

            // act1: snake(xx_cur) -> xx_act1
            Tensor& act1_a = mimi_weights_.get(res_prefix + "act1.alpha");
            Tensor& act1_b = mimi_weights_.get(res_prefix + "act1.beta");
            profile_step_start = std::chrono::high_resolution_clock::now();
            ops::snake_beta(ctx, *xx_cur, act1_a, act1_b, xx_act1, res_elems, out_d, Ls);
            profile_step_done(decoder_res_act1_ms);

            // conv1: conv(xx_act1) -> conv1_out
            Tensor& conv1_w = mimi_weights_.get(res_prefix + "conv1.conv.weight");
            Tensor& conv1_b = mimi_weights_.get(res_prefix + "conv1.conv.bias");
            profile_step_start = std::chrono::high_resolution_clock::now();
            ops::causal_conv1d(ctx, xx_act1, conv1_w, conv1_b, conv1_out, 1, out_d, out_d, Ls, 7, dilations[r]);
            profile_step_done(decoder_res_conv1_ms);

            // act2: snake(conv1_out) in-place
            Tensor& act2_a = mimi_weights_.get(res_prefix + "act2.alpha");
            Tensor& act2_b = mimi_weights_.get(res_prefix + "act2.beta");
            profile_step_start = std::chrono::high_resolution_clock::now();
            ops::snake_beta(ctx, conv1_out, act2_a, act2_b, conv1_out, res_elems, out_d, Ls);
            profile_step_done(decoder_res_act2_ms);

            // conv2: conv(conv1_out) -> conv2_out (kernel=1, pointwise)
            Tensor& conv2_w = mimi_weights_.get(res_prefix + "conv2.conv.weight");
            Tensor& conv2_b = mimi_weights_.get(res_prefix + "conv2.conv.bias");
            profile_step_start = std::chrono::high_resolution_clock::now();
            if (tts_decoder_fused_conv2_residual) {
                ops::causal_conv1d_k1_residual_add(ctx, conv1_out, conv2_w, conv2_b,
                                                   *residual_buf, 1, out_d, out_d, Ls);
                profile_step_done(decoder_res_conv2_ms);
            } else {
                ops::causal_conv1d(ctx, conv1_out, conv2_w, conv2_b, conv2_out,
                                   1, out_d, out_d, Ls, 1, 1);
                profile_step_done(decoder_res_conv2_ms);

                // residual_buf now holds the next xx. Swap buffer roles instead of
                // copying the whole residual output back into xx.
                profile_step_start = std::chrono::high_resolution_clock::now();
                ops::residual_add(ctx, *residual_buf, conv2_out, res_elems);
                profile_step_done(decoder_res_add_ms);
            }
            std::swap(xx_cur, residual_buf);
        }

        // dec_in is about to be replaced, so the queued conv-transpose work
        // that consumed the previous dec_in must be complete first.
        ctx.synchronize();

        L = L * stride;
        dec_in = std::move(*xx_cur);
    }

    auto t_decoder_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_decoder_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Decoder blocks (4 layers): %.1f ms", t_decoder_ms);
    if (tts_decoder_block_profile) {
        AILA_LOG_INFO("[TTS-Profile]   Decoder detail: dec0=%.1f ms stage_act=%.1f ms transpose=%.1f ms res_copy=%.1f ms res_act1=%.1f ms res_conv1=%.1f ms res_act2=%.1f ms res_conv2=%.1f ms res_add=%.1f ms",
                      decoder_dec0_ms,
                      decoder_stage_act_ms,
                      decoder_stage_transpose_ms,
                      decoder_res_copy_ms,
                      decoder_res_act1_ms,
                      decoder_res_conv1_ms,
                      decoder_res_act2_ms,
                      decoder_res_conv2_ms,
                      decoder_res_add_ms);
    }

    // ==========================================
    // 6. 最终 SnakeBeta 和映射 (channels=3 -> 1)
    // ==========================================
    auto t_final_start = std::chrono::high_resolution_clock::now();
    // 最终 SnakeBeta (输入维度是最后的 out_dims[3] 即 96)
    Tensor& dec5_a = mimi_weights_.get("decoder.decoder.5.alpha");
    Tensor& dec5_b = mimi_weights_.get("decoder.decoder.5.beta");
    ops::snake_beta(ctx, dec_in, dec5_a, dec5_b, dec_in, L * 96, 96, L);

    // dec6一维卷积 [7, 96, 1]
    Tensor dec6_out = Tensor::allocate(ctx, {L, 1});
    Tensor& dec6_w = mimi_weights_.get("decoder.decoder.6.conv.weight");
    Tensor& dec6_b = mimi_weights_.get("decoder.decoder.6.conv.bias");
    ops::causal_conv1d(ctx, dec_in, dec6_w, dec6_b, dec6_out, 1, 96, 1, L, 7, 1);

    // Clamp and copy back only the requested tail when incremental streaming
    // used a history window. Full vocoder decode still reads the whole tensor.
    int read_samples = L;
    if (tail_samples > 0) {
        read_samples = std::min(tail_samples, L);
    }
    const int read_start = L - read_samples;
    out_samples.resize(read_samples);
    float* host_ptr = out_samples.data();
    auto* dev_ptr = dec6_out.data_as<bf16>() + read_start;

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(read_samples), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            float val = static_cast<float>(dev_ptr[i]);
            // PyTorch clamp(min=-1, max=1)
            if (val < -1.0f) val = -1.0f;
            else if (val > 1.0f) val = 1.0f;
            dev_ptr[i] = bf16(val);
        });
    }).wait();

    // Now copy back as bf16 and convert to float on Host
    std::vector<bf16> cpu_bf16(read_samples);
    ctx.queue().memcpy(cpu_bf16.data(), dev_ptr, read_samples * sizeof(bf16)).wait();

    for (int i = 0; i < read_samples; ++i) {
        out_samples[i] = static_cast<float>(cpu_bf16[i]);
    }

#pragma pack(push, 1)
    struct WavHeader {
        char riff[4] = {'R', 'I', 'F', 'F'};
        uint32_t overall_size;
        char wave[4] = {'W', 'A', 'V', 'E'};
        char fmt_chunk_marker[4] = {'f', 'm', 't', ' '};
        uint32_t length_of_fmt = 16;
        uint16_t format_type = 1; // PCM
        uint16_t channels = 1;
        uint32_t sample_rate = 24000;
        uint32_t byterate = 24000 * 1 * 2;
        uint16_t block_align = 1 * 2;
        uint16_t bits_per_sample = 16;
        char data_chunk_header[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    };
#pragma pack(pop)

    static const bool tts_debug_wav = aila::env::read_flag("AILA_TTS_DEBUG_WAV", false);
    if (tts_debug_wav) {
        std::ofstream wav_file("mimi_output.wav", std::ios::binary);
        if (wav_file.is_open()) {
            WavHeader header;
            uint32_t num_samples = static_cast<uint32_t>(L);
            header.data_size = num_samples * 2;
            header.overall_size = header.data_size + 36;

            wav_file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            std::vector<int16_t> pcm_data(num_samples);
            for (size_t i = 0; i < num_samples; ++i) {
                float sample = out_samples[i];
                if (sample < -1.0f) sample = -1.0f;
                else if (sample > 1.0f) sample = 1.0f;
                pcm_data[i] = static_cast<int16_t>(sample * 32767.0f);
            }
            wav_file.write(reinterpret_cast<const char*>(pcm_data.data()), header.data_size);
        } else {
            AILA_LOG_WARN("[MimiDebug] Warning: failed to write mimi_output.wav");
        }
    }

    auto t_final_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_final_start).count();
    auto t_conv_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_conv_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Final conv+tanh: %.1f ms", t_final_ms);
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile] Conv stages total: %.1f ms (%d frames, %d/%d samples)",
        t_conv_total_ms, n_frames, read_samples, L);

    AILA_LOG_DEBUG("[MimiDebug] Decoded into %d samples.", L);
    return true;
}

bool Qwen3TTSBackend::decode_mimi_vocoder(Context& ctx,
                                         const std::vector<int32_t>& codes,
                                         int n_frames,
                                         std::vector<float>& out_samples) {
    if (!mimi_loaded_) {
        AILA_LOG_ERROR("[MimiDecoder] Error: Mimi Vocoder weights not loaded.");
        return false;
    }

    static const bool tts_profile = aila::env::read_flag("AILA_TTS_PROFILE", false);

    if (codes.size() < static_cast<size_t>(n_frames * 16)) {
        AILA_LOG_ERROR("[MimiDecoder] Error: input codes size does not match n_frames * 16.");
        return false;
    }

    const bool debug_print = aila::env::read_flag("AILA_MIMI_DEBUG", false);

    auto print_stats = [&](const std::string& name, Tensor& t) {
        if (!debug_print) return;
        int total = static_cast<int>(t.numel());
        if (total <= 0) return;
        std::vector<bf16> host(total);
        ctx.queue().memcpy(host.data(), t.data(), total * sizeof(bf16)).wait();
        float min_val = 1e30f;
        float max_val = -1e30f;
        float sum_val = 0.0f;
        int nan_count = 0;
        for (int i = 0; i < total; ++i) {
            float val = static_cast<float>(host[i]);
            if (std::isnan(val)) {
                nan_count++;
            } else {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                sum_val += val;
            }
        }
        std::cout << "    [TensorStats] " << name << ": size=" << total 
                  << ", min=" << min_val << ", max=" << max_val 
                  << ", mean=" << (nan_count < total ? sum_val / (total - nan_count) : 0.0f)
                  << ", NaNs=" << nan_count << std::endl;
        if (t.ndim() == 2 && t.shape(1) >= 5) {
            std::cout << "      [C++ Slice t=0, c=0..4]: " 
                      << static_cast<float>(host[0]) << ", "
                      << static_cast<float>(host[1]) << ", "
                      << static_cast<float>(host[2]) << ", "
                      << static_cast<float>(host[3]) << ", "
                      << static_cast<float>(host[4]) << std::endl;
            std::cout << "      [C++ Slice c=0, t=0..4]: " 
                      << static_cast<float>(host[0 * t.shape(1) + 0]) << ", "
                      << static_cast<float>(host[1 * t.shape(1) + 0]) << ", "
                      << static_cast<float>(host[2 * t.shape(1) + 0]) << ", "
                      << static_cast<float>(host[3 * t.shape(1) + 0]) << ", "
                      << static_cast<float>(host[4 * t.shape(1) + 0]) << std::endl;
        }
    };

    auto t_mimi_start = std::chrono::high_resolution_clock::now();

    // 0. 将 codes 复制 to GPU 设备端
    Tensor codes_dev = Tensor::allocate(ctx, {n_frames, 16}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(codes_dev.data(), codes.data(), n_frames * 16 * sizeof(int32_t));


    // ==========================================
    // 1. VQ 解量化查表与投影
    // ==========================================
    auto t_vq_start = std::chrono::high_resolution_clock::now();
    Tensor temp_first = Tensor::allocate(ctx, {n_frames, 256});
    Tensor temp_rest = Tensor::allocate(ctx, {n_frames, 256});

    // first codebook cols=0
    ops::vq_lookup_add(ctx, codes_dev, mimi_weights_.get("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
                       0, temp_first, n_frames, 256, false);

    // rest codebooks cols=1..15
    ops::vq_lookup_add(ctx, codes_dev, mimi_weights_.get("decoder.quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum"),
                       1, temp_rest, n_frames, 256, false);

    for (int c = 1; c < 15; ++c) {
        std::string emb_name = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(c) + "._codebook.embedding_sum";
        ops::vq_lookup_add(ctx, codes_dev, mimi_weights_.get(emb_name),
                           c + 1, temp_rest, n_frames, 256, true);
    }


    Tensor proj_first = Tensor::allocate(ctx, {n_frames, 512});
    Tensor proj_rest = Tensor::allocate(ctx, {n_frames, 512});

    mimi_first_proj_.forward(ctx, temp_first, proj_first, n_frames);
    mimi_rest_proj_.forward(ctx, temp_rest, proj_rest, n_frames);


    // 向量相加得到 latent
    Tensor latent = Tensor::allocate(ctx, {n_frames, 512});
    ops::copy_tensor(ctx, proj_first, latent, n_frames * 512);
    ops::residual_add(ctx, latent, proj_rest, n_frames * 512);

    auto t_vq_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_vq_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   VQ lookup+proj: %.1f ms", t_vq_ms);

    // ==========================================
    // 2. Pre-conv 卷积 [3, 512, 1024]
    // ==========================================
    auto t_preconv_start = std::chrono::high_resolution_clock::now();
    Tensor pre_conv_out = Tensor::allocate(ctx, {n_frames, 1024});
    Tensor& pre_conv_w = mimi_weights_.get("decoder.pre_conv.conv.weight");
    Tensor& pre_conv_b = mimi_weights_.get("decoder.pre_conv.conv.bias");
    ops::causal_conv1d(ctx, latent, pre_conv_w, pre_conv_b, pre_conv_out, 1, 512, 1024, n_frames, 3, 1);


    auto t_preconv_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_preconv_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Pre-conv: %.1f ms", t_preconv_ms);

    // ==========================================
    // 3. Pre-transformer (8 Layers Causal Attention)
    // ==========================================
    auto t_pretfm_start = std::chrono::high_resolution_clock::now();
    Tensor pre_tfm_in = Tensor::allocate(ctx, {n_frames, 512});
    Tensor& pre_tfm_in_proj_b = mimi_weights_.get("decoder.pre_transformer.input_proj.bias");
    mimi_pre_tfm_in_proj_.forward_bias(ctx, pre_conv_out, pre_tfm_in_proj_b, pre_tfm_in, n_frames);


    // 辅助 layer scale 核函数
    auto apply_layer_scale_gpu = [&](Tensor& x, Tensor& scale, int length, int channels) {
        auto* x_ptr = x.data_as<bf16>();
        auto* scale_ptr = scale.data_as<bf16>();
        int total = length * channels;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                if (i >= total) return;
                int c = i % channels;
                x_ptr[i] = bf16(static_cast<float>(x_ptr[i]) * static_cast<float>(scale_ptr[c]));
            });
        });
    };

    AILA_LOG_DEBUG("[MimiDebug] Step 6: Pre-transformer 8 layers loop");
    Tensor x = Tensor::allocate(ctx, {n_frames, 512});
    ops::copy_tensor(ctx, pre_tfm_in, x, n_frames * 512);

    for (int l = 0; l < 8; ++l) {
        std::string layer_prefix = "decoder.pre_transformer.layers." + std::to_string(l) + ".";

        Tensor residual = Tensor::allocate(ctx, {n_frames, 512});
        ops::copy_tensor(ctx, x, residual, n_frames * 512);


        // input rms norm
        Tensor normed = Tensor::allocate(ctx, {n_frames, 512});
        Tensor& input_ln_w = mimi_weights_.get(layer_prefix + "input_layernorm.weight");
        ops::rms_norm(ctx, x, input_ln_w, 1e-5f, normed, n_frames, 512);


        Tensor q = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor k = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor v = Tensor::allocate(ctx, {n_frames, 1024});

        auto& layer_linears = mimi_pre_tfm_linears_[static_cast<size_t>(l)];
        layer_linears.q_proj.forward(ctx, normed, q, n_frames);
        layer_linears.k_proj.forward(ctx, normed, k, n_frames);
        layer_linears.v_proj.forward(ctx, normed, v, n_frames);


        // Apply RoPE positions (num_heads=16, head_dim=64)
        ops::apply_rope(ctx, q, k, n_frames, 0, 16, 16, 64, 10000.0f);


        // Attention Prefill (num_heads=16, head_dim=64 -> output_dim=1024)
        Tensor attn_out = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor scores_buf = Tensor::allocate(ctx, {16, n_frames, n_frames}, dnnl::memory::data_type::f32);
        ops::attention_prefill(ctx, q, k, v, attn_out, scores_buf, n_frames, 16, 16, 64);


        // Out proj
        Tensor proj_out = Tensor::allocate(ctx, {n_frames, 512});
        layer_linears.o_proj.forward(ctx, attn_out, proj_out, n_frames);


        // Attention layer scale
        Tensor& attn_scale = mimi_weights_.get(layer_prefix + "self_attn_layer_scale.scale");
        apply_layer_scale_gpu(proj_out, attn_scale, n_frames, 512);


        // Add to residual
        ops::residual_add(ctx, residual, proj_out, n_frames * 512);
        ops::copy_tensor(ctx, residual, x, n_frames * 512);


        // MLP
        Tensor mlp_residual = Tensor::allocate(ctx, {n_frames, 512});
        ops::copy_tensor(ctx, x, mlp_residual, n_frames * 512);


        Tensor normed_post = Tensor::allocate(ctx, {n_frames, 512});
        Tensor& post_attn_ln_w = mimi_weights_.get(layer_prefix + "post_attention_layernorm.weight");
        ops::rms_norm(ctx, x, post_attn_ln_w, 1e-5f, normed_post, n_frames, 512);


        Tensor gate_out = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor up_out = Tensor::allocate(ctx, {n_frames, 1024});

        layer_linears.gate_proj.forward(ctx, normed_post, gate_out, n_frames);
        layer_linears.up_proj.forward(ctx, normed_post, up_out, n_frames);


        // SwiGLU activation: gate = silu(gate) * up
        ops::swiglu(ctx, gate_out, up_out, gate_out, n_frames * 1024);
        ctx.synchronize();

        Tensor down_out = Tensor::allocate(ctx, {n_frames, 512});
        layer_linears.down_proj.forward(ctx, gate_out, down_out, n_frames);
        ctx.synchronize();

        // MLP layer scale
        Tensor& mlp_scale = mimi_weights_.get(layer_prefix + "mlp_layer_scale.scale");
        apply_layer_scale_gpu(down_out, mlp_scale, n_frames, 512);
        ctx.synchronize();

        // Add to residual
        ops::residual_add(ctx, mlp_residual, down_out, n_frames * 512);
        ops::copy_tensor(ctx, mlp_residual, x, n_frames * 512);
        ctx.synchronize();
    }

    // Final RMS norm and projection out
    Tensor final_normed = Tensor::allocate(ctx, {n_frames, 512});
    Tensor& pre_tfm_norm_w = mimi_weights_.get("decoder.pre_transformer.norm.weight");
    ops::rms_norm(ctx, x, pre_tfm_norm_w, 1e-5f, final_normed, n_frames, 512);
    ctx.synchronize();

    Tensor pre_tfm_out = Tensor::allocate(ctx, {n_frames, 1024});
    Tensor& pre_tfm_out_proj_b = mimi_weights_.get("decoder.pre_transformer.output_proj.bias");
    mimi_pre_tfm_out_proj_.forward_bias(ctx, final_normed, pre_tfm_out_proj_b, pre_tfm_out, n_frames);
    ctx.synchronize();

    auto t_pretfm_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_pretfm_start).count();
    if (tts_profile) AILA_LOG_INFO("[TTS-Profile]   Pre-transformer (8 layers): %.1f ms", t_pretfm_ms);

    return mimi_conv_stages(ctx, pre_tfm_out, n_frames, out_samples);
}

bool Qwen3TTSBackend::init_mimi_stream(Context& ctx, MimiStreamState& state, int max_frames) {
    if (!mimi_loaded_) return false;
    state.reset();
    state.max_frames = max_frames;

    // Accumulation buffer for full-history Mimi decode stages.
    state.latent_buffer = Tensor::allocate(ctx, {static_cast<int64_t>(max_frames), 512});
    state.pre_tfm_out_buffer = Tensor::allocate(ctx, {static_cast<int64_t>(max_frames), 1024});
    for (int l = 0; l < 8; ++l) {
        state.pre_tfm_k_cache[static_cast<size_t>(l)] =
            Tensor::allocate(ctx, {16, static_cast<int64_t>(max_frames), 64});
        state.pre_tfm_v_cache[static_cast<size_t>(l)] =
            Tensor::allocate(ctx, {16, static_cast<int64_t>(max_frames), 64});
    }

    return true;
}

bool Qwen3TTSBackend::decode_mimi_incremental(Context& ctx,
    const std::vector<int32_t>& codes, int new_frames,
    MimiStreamState& state, std::vector<float>& out_samples) {
    if (!mimi_loaded_ || new_frames <= 0) return false;
    static const bool tts_profile = aila::env::read_flag("AILA_TTS_PROFILE", false);
    static const bool tts_ptfm_fused_residual =
        aila::env::read_flag("AILA_TTS_MIMI_PTFM_FUSED_RESIDUAL", false);
    const auto t_decode_start = std::chrono::high_resolution_clock::now();

    int start_pos = state.total_frames;
    int total_frames = start_pos + new_frames;

    // === 1. VQ lookup on NEW codes only ===
    const auto t_vq_start = std::chrono::high_resolution_clock::now();
    Tensor codes_dev = Tensor::allocate(ctx, {new_frames, 16}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(codes_dev.data(), codes.data(), new_frames * 16 * sizeof(int32_t));

    Tensor temp_first = Tensor::allocate(ctx, {new_frames, 256});
    ops::vq_lookup_add(ctx, codes_dev,
        mimi_weights_.get("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
        0, temp_first, new_frames, 256, false);
    Tensor temp_rest = Tensor::allocate(ctx, {new_frames, 256});
    ops::vq_lookup_add(ctx, codes_dev,
        mimi_weights_.get("decoder.quantizer.rvq_rest.vq.layers.0._codebook.embedding_sum"),
        1, temp_rest, new_frames, 256, false);
    for (int c = 1; c < 15; ++c) {
        std::string emb_name = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(c) + "._codebook.embedding_sum";
        ops::vq_lookup_add(ctx, codes_dev, mimi_weights_.get(emb_name), c + 1, temp_rest, new_frames, 256, true);
    }
    ctx.synchronize();

    Tensor proj_first = Tensor::allocate(ctx, {new_frames, 512});
    Tensor proj_rest = Tensor::allocate(ctx, {new_frames, 512});
    mimi_first_proj_.forward(ctx, temp_first, proj_first, new_frames);
    mimi_rest_proj_.forward(ctx, temp_rest, proj_rest, new_frames);
    ctx.synchronize();

    Tensor latent_new = Tensor::allocate(ctx, {new_frames, 512});
    ops::copy_tensor(ctx, proj_first, latent_new, new_frames * 512);
    ops::residual_add(ctx, latent_new, proj_rest, new_frames * 512);
    ctx.synchronize();
    if (tts_profile) {
        const auto t_vq_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_vq_start).count();
        AILA_LOG_INFO("[TTS-Profile]   Mimi incremental VQ+proj: %.1f ms (%d new/%d total frames)",
                      t_vq_ms, new_frames, total_frames);
    }

    // === 2. Append to latent_buffer ===
    bf16* lat_dst = state.latent_buffer.data_as<bf16>() + start_pos * 512;
    ctx.memcpy_h2d_async(lat_dst, latent_new.data(), new_frames * 512 * sizeof(bf16));

    if (total_frames > state.max_frames) {
        AILA_LOG_ERROR("[MimiDecoder] Stream frame capacity exceeded: %d > %d",
                       total_frames, state.max_frames);
        return false;
    }

    // === 3. Pre-conv on NEW frames with causal latent overlap ===
    const auto t_preconv_start = std::chrono::high_resolution_clock::now();
    const int preconv_window_start = std::max(0, start_pos - 2);
    const int preconv_window_frames = total_frames - preconv_window_start;
    bf16* latent_window_ptr = state.latent_buffer.data_as<bf16>() + preconv_window_start * 512;
    Tensor latent_window = Tensor::view(ctx, latent_window_ptr, {preconv_window_frames, 512},
                                        state.latent_buffer.dtype());
    Tensor preconv_window = Tensor::allocate(ctx, {preconv_window_frames, 1024});
    ops::causal_conv1d(ctx, latent_window,
        mimi_weights_.get("decoder.pre_conv.conv.weight"),
        mimi_weights_.get("decoder.pre_conv.conv.bias"),
        preconv_window, 1, 512, 1024, preconv_window_frames, 3, 1);
    ctx.synchronize();
    if (tts_profile) {
        const auto t_preconv_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_preconv_start).count();
        AILA_LOG_INFO("[TTS-Profile]   Mimi incremental pre-conv: %.1f ms (%d/%d frames)",
                      t_preconv_ms, preconv_window_frames, total_frames);
    }

    // === 4. Pre-transformer on NEW frames with persistent K/V cache ===
    const auto t_pretfm_start = std::chrono::high_resolution_clock::now();
    bf16* preconv_new_ptr = preconv_window.data_as<bf16>() +
        (preconv_window_frames - new_frames) * 1024;
    Tensor preconv_new = Tensor::view(ctx, preconv_new_ptr, {new_frames, 1024},
                                      preconv_window.dtype());
    Tensor pre_tfm_in = Tensor::allocate(ctx, {new_frames, 512});
    mimi_pre_tfm_in_proj_.forward_bias(ctx, preconv_new,
        mimi_weights_.get("decoder.pre_transformer.input_proj.bias"), pre_tfm_in, new_frames);
    ctx.synchronize();

    Tensor x = Tensor::allocate(ctx, {new_frames, 512});
    ops::copy_tensor(ctx, pre_tfm_in, x, new_frames * 512);
    ctx.synchronize();

    auto apply_layer_scale_gpu = [&](Tensor& t, Tensor& scale, int len, int ch) {
        auto* tp = t.data_as<bf16>();
        auto* sp = scale.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(len * ch)),
            [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                int c = i % ch;
                tp[i] = bf16(static_cast<float>(tp[i]) * static_cast<float>(sp[c]));
            });
    };

    auto add_scaled_to_x_gpu = [&](Tensor& update, Tensor& scale, int len, int ch) {
        auto* xp = x.data_as<bf16>();
        auto* up = update.data_as<bf16>();
        auto* sp = scale.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(len * ch)),
            [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                int c = i % ch;
                float base = static_cast<float>(xp[i]);
                float delta = static_cast<float>(up[i]) * static_cast<float>(sp[c]);
                xp[i] = bf16(base + delta);
            });
    };

    for (int l = 0; l < 8; ++l) {
        std::string lp = "decoder.pre_transformer.layers." + std::to_string(l) + ".";

        Tensor residual;
        if (!tts_ptfm_fused_residual) {
            residual = Tensor::allocate(ctx, {new_frames, 512});
            ops::copy_tensor(ctx, x, residual, new_frames * 512);
            ctx.synchronize();
        }

        Tensor normed = Tensor::allocate(ctx, {new_frames, 512});
        ops::rms_norm(ctx, x, mimi_weights_.get(lp + "input_layernorm.weight"),
                      1e-5f, normed, new_frames, 512);
        ctx.synchronize();

        Tensor q = Tensor::allocate(ctx, {new_frames, 1024});
        Tensor k = Tensor::allocate(ctx, {new_frames, 1024});
        Tensor v = Tensor::allocate(ctx, {new_frames, 1024});
        auto& layer_linears = mimi_pre_tfm_linears_[static_cast<size_t>(l)];
        layer_linears.q_proj.forward(ctx, normed, q, new_frames);
        layer_linears.k_proj.forward(ctx, normed, k, new_frames);
        layer_linears.v_proj.forward(ctx, normed, v, new_frames);
        ctx.synchronize();

        ops::apply_rope(ctx, q, k, new_frames, start_pos, 16, 16, 64, 10000.0f);
        ctx.synchronize();

        Tensor& k_cache = state.pre_tfm_k_cache[static_cast<size_t>(l)];
        Tensor& v_cache = state.pre_tfm_v_cache[static_cast<size_t>(l)];
        ops::copy_to_cache(ctx, k, k_cache, new_frames, start_pos, 16, 64, state.max_frames);
        ops::copy_to_cache(ctx, v, v_cache, new_frames, start_pos, 16, 64, state.max_frames);
        ctx.synchronize();

        Tensor attn_out = Tensor::allocate(ctx, {new_frames, 1024});
        if (start_pos == 0) {
            Tensor scores_buf = Tensor::allocate(ctx, {16, new_frames, new_frames}, dnnl::memory::data_type::f32);
            ops::attention_prefill(ctx, q, k, v, attn_out, scores_buf, new_frames, 16, 16, 64);
        } else {
            Tensor scores_buf = Tensor::allocate(ctx, {16, new_frames, total_frames}, dnnl::memory::data_type::f32);
            ops::attention_prefill_cached(ctx, q, k_cache, v_cache, attn_out, scores_buf,
                                          new_frames, start_pos, 16, 16, 64, state.max_frames);
        }
        ctx.synchronize();

        Tensor proj_out = Tensor::allocate(ctx, {new_frames, 512});
        layer_linears.o_proj.forward(ctx, attn_out, proj_out, new_frames);
        ctx.synchronize();

        Tensor& attn_scale = mimi_weights_.get(lp + "self_attn_layer_scale.scale");
        if (tts_ptfm_fused_residual) {
            add_scaled_to_x_gpu(proj_out, attn_scale, new_frames, 512);
        } else {
            apply_layer_scale_gpu(proj_out, attn_scale, new_frames, 512);
            ctx.queue().wait();
            ops::residual_add(ctx, residual, proj_out, new_frames * 512);
            ops::copy_tensor(ctx, residual, x, new_frames * 512);
        }
        ctx.synchronize();

        Tensor mlp_res;
        if (!tts_ptfm_fused_residual) {
            mlp_res = Tensor::allocate(ctx, {new_frames, 512});
            ops::copy_tensor(ctx, x, mlp_res, new_frames * 512);
            ctx.synchronize();
        }

        Tensor normed_post = Tensor::allocate(ctx, {new_frames, 512});
        ops::rms_norm(ctx, x, mimi_weights_.get(lp + "post_attention_layernorm.weight"),
                      1e-5f, normed_post, new_frames, 512);
        ctx.synchronize();

        Tensor gate_out = Tensor::allocate(ctx, {new_frames, 1024});
        Tensor up_out   = Tensor::allocate(ctx, {new_frames, 1024});
        layer_linears.gate_proj.forward(ctx, normed_post, gate_out, new_frames);
        layer_linears.up_proj.forward(ctx, normed_post, up_out, new_frames);
        ctx.synchronize();

        ops::swiglu(ctx, gate_out, up_out, gate_out, new_frames * 1024);
        ctx.synchronize();

        Tensor down_out = Tensor::allocate(ctx, {new_frames, 512});
        layer_linears.down_proj.forward(ctx, gate_out, down_out, new_frames);
        ctx.synchronize();

        Tensor& mlp_scale = mimi_weights_.get(lp + "mlp_layer_scale.scale");
        if (tts_ptfm_fused_residual) {
            add_scaled_to_x_gpu(down_out, mlp_scale, new_frames, 512);
        } else {
            apply_layer_scale_gpu(down_out, mlp_scale, new_frames, 512);
            ctx.queue().wait();
            ops::residual_add(ctx, mlp_res, down_out, new_frames * 512);
            ops::copy_tensor(ctx, mlp_res, x, new_frames * 512);
        }
        ctx.synchronize();
    }

    Tensor final_normed = Tensor::allocate(ctx, {new_frames, 512});
    ops::rms_norm(ctx, x, mimi_weights_.get("decoder.pre_transformer.norm.weight"),
                  1e-5f, final_normed, new_frames, 512);
    ctx.synchronize();

    Tensor pre_tfm_out_new = Tensor::allocate(ctx, {new_frames, 1024});
    mimi_pre_tfm_out_proj_.forward_bias(ctx, final_normed,
        mimi_weights_.get("decoder.pre_transformer.output_proj.bias"), pre_tfm_out_new, new_frames);
    bf16* pre_tfm_dst = state.pre_tfm_out_buffer.data_as<bf16>() + start_pos * 1024;
    Tensor pre_tfm_dst_view = Tensor::view(ctx, pre_tfm_dst, {new_frames, 1024},
                                           state.pre_tfm_out_buffer.dtype());
    ops::copy_tensor(ctx, pre_tfm_out_new, pre_tfm_dst_view, new_frames * 1024);
    ctx.synchronize();
    if (tts_profile) {
        const auto t_pretfm_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_pretfm_start).count();
        AILA_LOG_INFO("[TTS-Profile]   Mimi incremental pre-transformer cached: %.1f ms (%d new/%d total frames)",
                      t_pretfm_ms, new_frames, total_frames);
    }

    // === Conv stages (shared with full decode_mimi_vocoder) ===
    const auto t_conv_start = std::chrono::high_resolution_clock::now();
    static const int s_conv_window_history_frames = std::max(
        0, aila::env::read_int_raw("AILA_TTS_MIMI_CONV_WINDOW_FRAMES", 24));
    const int target_new_samples = new_frames * kMimiSamplesPerFrame;
    int conv_window_start = 0;
    int conv_window_frames = total_frames;
    if (s_conv_window_history_frames > 0 && start_pos > s_conv_window_history_frames) {
        conv_window_start = start_pos - s_conv_window_history_frames;
        conv_window_frames = total_frames - conv_window_start;
    }

    bf16* window_ptr = state.pre_tfm_out_buffer.data_as<bf16>() + conv_window_start * 1024;
    Tensor conv_input_view = Tensor::view(ctx, window_ptr, {conv_window_frames, 1024},
                                          state.pre_tfm_out_buffer.dtype());

    if (!mimi_conv_stages(ctx, conv_input_view, conv_window_frames, out_samples, target_new_samples)) {
        return false;
    }
    if (tts_profile) {
        const auto t_conv_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_conv_start).count();
        const auto t_decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_decode_start).count();
        AILA_LOG_INFO("[TTS-Profile]   Mimi incremental total: %.1f ms (conv+readback %.1f ms)",
                      t_decode_ms, t_conv_ms);
        if (conv_window_start > 0) {
            AILA_LOG_INFO("[TTS-Profile]   Mimi conv window: %d+%d/%d frames, tail=%d samples",
                          s_conv_window_history_frames, new_frames, total_frames,
                          static_cast<int>(out_samples.size()));
        }
    }

    state.last_audio_sample_count = total_frames * kMimiSamplesPerFrame;
    state.total_frames = total_frames;

    return true;
}

bool Qwen3TTSBackend::decode_mimi_flush(Context& ctx, MimiStreamState& state,
    std::vector<float>& out_samples) {
    (void)ctx; (void)state; (void)out_samples;
    return true;
}
