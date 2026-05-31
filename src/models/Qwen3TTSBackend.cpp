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

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {
int round_up_seq(int v, int g) { return ((v + g - 1) / g) * g; }

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

    talker_kv_cache_.init(ctx, talker_cfg_, max_seq_len_);

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

    predictor_kv_cache_.init(ctx, predictor_cfg_, 17);

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
                                       const GenerationConfig& gen_config,
                                       std::vector<int32_t>& out_codes,
                                       int& out_n_frames) {
    out_codes.clear();
    out_n_frames = 0;

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
    reset();

    // 临时分配用于文本投影的 GPU 张量
    Tensor text_ids_dev = Tensor::allocate(ctx, {L}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(text_ids_dev.data(), text_tokens.data(), L * sizeof(int));

    Tensor text_emb = Tensor::allocate(ctx, {L, 2048});
    ops::embedding_lookup(ctx, *talker_text_embed_weight_, text_ids_dev.data_as<int>(), L, text_emb, 2048);
    print_gpu_tensor(ctx, "text_emb[0, 0, :5]", text_emb, 0);

    // 对 text_emb 过 ResizeMLP (text_projection)
    Tensor fc1_out = Tensor::allocate(ctx, {L, 2048});
    talker_text_proj_fc1_.forward_bias(ctx, text_emb, *talker_text_proj_fc1_bias_, fc1_out, L);
    print_gpu_tensor(ctx, "fc1_out[0, 0, :5]", fc1_out, 0);

    // SiLU(x) = x * sigmoid(x)
    Tensor silu_out = Tensor::allocate(ctx, {L, 2048});
    ops::sigmoid_mul(ctx, fc1_out, fc1_out, silu_out, L * 2048);
    print_gpu_tensor(ctx, "silu_out[0, 0, :5]", silu_out, 0);

    Tensor projected_text = Tensor::allocate(ctx, {L, H_talker});
    talker_text_proj_fc2_.forward_bias(ctx, silu_out, *talker_text_proj_fc2_bias_, projected_text, L);
    print_gpu_tensor(ctx, "projected_text[0, 0, :5]", projected_text, 0);

    // 按照 python 的规则从输入投影中提取 bos, eos, pad 的 embedding 并切片
    // 映射 tts_bos_token_id (151672), tts_eos_token_id (151673), tts_pad_token_id (151671)
    // 简单起见，我们在 C++ 直接用 embedding 逻辑实时计算它们的值即可：
    auto get_resize_mlp_embed = [&](int token_id) {
        Tensor t_id = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(t_id.data(), &token_id, sizeof(int));
        Tensor emb_pt = Tensor::allocate(ctx, {1, 2048});
        ops::embedding_lookup(ctx, *talker_text_embed_weight_, t_id.data_as<int>(), 1, emb_pt, 2048);

        Tensor f1 = Tensor::allocate(ctx, {1, 2048});
        talker_text_proj_fc1_.forward_bias(ctx, emb_pt, *talker_text_proj_fc1_bias_, f1, 1);
        Tensor s1 = Tensor::allocate(ctx, {1, 2048});
        ops::sigmoid_mul(ctx, f1, f1, s1, 2048);

        Tensor out = Tensor::allocate(ctx, {1, H_talker});
        talker_text_proj_fc2_.forward_bias(ctx, s1, *talker_text_proj_fc2_bias_, out, 1);
        return out;
    };

    Tensor tts_bos_embed = get_resize_mlp_embed(151672);
    Tensor tts_eos_embed = get_resize_mlp_embed(151673);
    Tensor tts_pad_embed = get_resize_mlp_embed(151671);
    print_gpu_tensor(ctx, "tts_pad_embed[0, 0, :5]", tts_pad_embed, 0);

    auto get_talker_codec_embed = [&](int codec_id) {
        Tensor c_id = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(c_id.data(), &codec_id, sizeof(int));
        Tensor out = Tensor::allocate(ctx, {1, H_talker});
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, c_id.data_as<int>(), 1, out, H_talker);
        return out;
    };

    Tensor embed_codec_pad = get_talker_codec_embed(2148);
    Tensor embed_codec_bos = get_talker_codec_embed(2149);

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

    // 构建 non-streaming Prefill (对齐 ggml 结构)
    bool has_spk = (speaker_embedding.size() == static_cast<size_t>(H_talker));
    // ggml prefill 结构:
    // [role0] [role1] [role2] [c0+pad] [c1+pad] [c2+pad] [c3+pad] [spk+pad] [codec_pad+bos] [first_text+codec_bos]
    int prefill_len = has_spk ? 10 : 9;
    int first_text_idx = prefill_len - 1; // 最后一个位置 = first_text + codec_bos
    int spk_idx = 7; // speaker embedding position (if present)
    Tensor prefill_embeds = Tensor::allocate(ctx, {prefill_len, H_talker});
    {
        bf16* dst = prefill_embeds.data_as<bf16>();
        bf16* src_role = projected_text.data_as<bf16>();

        // 1. 前 3 帧：role 前缀投影
        ctx.queue().memcpy(dst, src_role, 3 * H_talker * sizeof(bf16)).wait();

        // 2. 第 3-6 帧：codec_prefill (4 tokens) + tts_pad_embed
        std::vector<int> codec_prefill_ids = {2154, 2156, 2055, 2157}; // think, think_bos, language(zh), think_eos
        Tensor codec_ids_dev = Tensor::allocate(ctx, {4}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(codec_ids_dev.data(), codec_prefill_ids.data(), 4 * sizeof(int));
        Tensor codec_embs = Tensor::allocate(ctx, {4, H_talker});
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, codec_ids_dev.data_as<int>(), 4, codec_embs, H_talker);

        bf16* c_embs = codec_embs.data_as<bf16>();
        bf16* p_pad = tts_pad_embed.data_as<bf16>();
        bf16* p_bos = tts_bos_embed.data_as<bf16>();

        for (int i = 0; i < 4; ++i) {
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> h) {
                dst[(3 + i) * H_talker + h] = c_embs[i * H_talker + h] + p_pad[h];
            });
        }
        ctx.queue().wait();

        // 3. 第 7 帧（可选）：speaker_embedding + tts_pad_embed
        if (has_spk) {
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
            ctx.queue().wait();
        }

        // 4. 第 8（无 spk 则为 7）帧：codec_pad + tts_bos_embed
        int pad_bos_idx = has_spk ? 8 : 7;
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
    std::vector<bf16> trailing_text_hidden(trailing_len * H_talker);
    if (trailing_token_count > 0) {
        // 拷贝第 5 个 token (index=4) 起的正文投影
        bf16* all_proj = projected_text.data_as<bf16>();
        std::vector<bf16> proj_cpu(L * H_talker);
        ctx.queue().memcpy(proj_cpu.data(), all_proj, L * H_talker * sizeof(bf16)).wait();
        for (int t = 0; t < trailing_token_count; ++t) {
            std::memcpy(trailing_text_hidden.data() + t * H_talker,
                       proj_cpu.data() + (text_body_start + 1 + t) * H_talker,
                       H_talker * sizeof(bf16));
        }
    }
    // 末尾追加 tts_eos_embed
    {
        std::vector<bf16> eos_cpu(H_talker);
        ctx.queue().memcpy(eos_cpu.data(), tts_eos_embed.data(), H_talker * sizeof(bf16)).wait();
        std::memcpy(trailing_text_hidden.data() + (trailing_len - 1) * H_talker,
                   eos_cpu.data(), H_talker * sizeof(bf16));
    }

    print_gpu_tensor(ctx, "prefill_embeds[0, 0, :5]", prefill_embeds, 0);

    // 运行 Talker Prefill
    ensure_talker_runtime_buffers(ctx, prefill_len);
    ensure_talker_prefill_scores(ctx, prefill_len);

    // 拷贝 prefill_embeds 到 t_buf_.hidden
    ctx.queue().memcpy(t_buf_.hidden.data(), prefill_embeds.data(), prefill_len * H_talker * sizeof(bf16)).wait();

    // 初始归一化
    ops::rms_norm(ctx, t_buf_.hidden, *talker_layers_[0].input_ln_weight,
                  talker_cfg_.rms_norm_eps, t_buf_.normed, prefill_len, H_talker);

    int rotary_dim_talker = talker_cfg_.head_dim;
    if (rotary_dim_talker & 1) --rotary_dim_talker;

    for (int i = 0; i < talker_cfg_.num_hidden_layers; i++) {
        auto& L = talker_layers_[i];
        
        L.qkv_proj.forward(ctx, t_buf_.normed, t_buf_.qkv, prefill_len);
        ops::split_qkv(ctx, t_buf_.qkv, t_buf_.q, t_buf_.k, t_buf_.v, prefill_len, QD_talker, KVD_talker);

        ops::head_rms_norm(ctx, t_buf_.q, *L.q_norm_weight, talker_cfg_.rms_norm_eps, prefill_len, talker_cfg_.num_attention_heads, talker_cfg_.head_dim);
        ops::head_rms_norm(ctx, t_buf_.k, *L.k_norm_weight, talker_cfg_.rms_norm_eps, prefill_len, talker_cfg_.num_key_value_heads, talker_cfg_.head_dim);

        ops::apply_rope_partial(ctx, t_buf_.q, t_buf_.k, prefill_len, 0,
                                talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads,
                                talker_cfg_.head_dim, rotary_dim_talker, talker_cfg_.rope_theta);

        ops::copy_to_cache(ctx, t_buf_.k, talker_kv_cache_.k_cache(i), prefill_len, 0,
                           talker_cfg_.num_key_value_heads, talker_cfg_.head_dim, talker_kv_cache_.max_length());
        ops::copy_to_cache(ctx, t_buf_.v, talker_kv_cache_.v_cache(i), prefill_len, 0,
                           talker_cfg_.num_key_value_heads, talker_cfg_.head_dim, talker_kv_cache_.max_length());

        ops::attention_prefill(ctx, t_buf_.q, t_buf_.k, t_buf_.v,
                               t_buf_.attn_out, t_buf_.scores, prefill_len,
                               talker_cfg_.num_attention_heads, talker_cfg_.num_key_value_heads, talker_cfg_.head_dim);

        L.o_proj.forward(ctx, t_buf_.attn_out, t_buf_.gate, prefill_len);
        
        ops::fused_add_rms_norm(ctx, t_buf_.hidden, t_buf_.gate, *L.post_attn_ln_weight, talker_cfg_.rms_norm_eps, t_buf_.normed, prefill_len, H_talker);

        L.gate_up_proj.forward(ctx, t_buf_.normed, t_buf_.gate_up, prefill_len);
        ops::split_gate_up(ctx, t_buf_.gate_up, t_buf_.gate, t_buf_.up, prefill_len, FF_talker);
        ops::swiglu(ctx, t_buf_.gate, t_buf_.up, t_buf_.gate, prefill_len * FF_talker);

        L.down_proj.forward(ctx, t_buf_.gate, t_buf_.attn_out, prefill_len);
        
        Tensor* next_input_ln = (i < talker_cfg_.num_hidden_layers - 1) ? talker_layers_[i + 1].input_ln_weight : talker_final_norm_weight_;
        ops::fused_add_rms_norm(ctx, t_buf_.hidden, t_buf_.attn_out, *next_input_ln, talker_cfg_.rms_norm_eps, t_buf_.normed, prefill_len, H_talker);
        
    }

    // Final prefill step to get logits from the last position
    Tensor final_hidden = Tensor::allocate(ctx, {1, H_talker});
    {
        bf16* src = t_buf_.hidden.data_as<bf16>();
        bf16* dst = final_hidden.data_as<bf16>();
        ctx.queue().memcpy(dst, src + (prefill_len - 1) * H_talker, H_talker * sizeof(bf16)).wait();
    }
    
    // Final Norm
    Tensor final_normed = Tensor::allocate(ctx, {1, H_talker});
    ops::rms_norm(ctx, final_hidden, *talker_final_norm_weight_, talker_cfg_.rms_norm_eps, final_normed, 1, H_talker);

    // Logits
    talker_codec_head_.forward(ctx, final_normed, t_buf_.logits, 1);

    // 采样得到首码 (codebook 0)
    int first_token = ops::sample_with_config(ctx, t_buf_.logits, talker_cfg_.vocab_size, gen_config, {});
    talker_kv_cache_.advance(prefill_len);
    current_talker_len_ = prefill_len;

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

    int max_tokens = tts_gen.max_new_tokens;
    out_codes.reserve(max_tokens * 16);

    while (gen_step < max_tokens) {
        if (token == eos_id) {
            break;
        }

        // 收集这帧 codebook 0
        std::vector<int> frame_codes(16, 0);
        frame_codes[0] = token;

        // 获取首码 embedding
        Tensor first_code_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(first_code_dev.data(), &token, sizeof(int));
        Tensor last_id_hidden = Tensor::allocate(ctx, {1, H_talker});
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, first_code_dev.data_as<int>(), 1, last_id_hidden, H_talker);

        // 拼接 past_hidden_talker 与 last_id_hidden -> pred_input [2, H_talker]
        {
            bf16* dst = pred_input.data_as<bf16>();
            bf16* src_past = past_hidden_talker.data_as<bf16>();
            bf16* src_last = last_id_hidden.data_as<bf16>();
            ctx.queue().memcpy(dst, src_past, H_talker * sizeof(bf16)).wait();
            ctx.queue().memcpy(dst + H_talker, src_last, H_talker * sizeof(bf16)).wait();
        }

        // 输入到 Code Predictor
        if (has_predictor_projection_) {
            predictor_projection_linear_.forward_bias(ctx, pred_input, *predictor_projection_bias_, p_buf_.pred_input_proj, 2);
        } else {
            ctx.queue().memcpy(p_buf_.pred_input_proj.data(), pred_input.data(), 2 * H_pred * sizeof(bf16)).wait();
        }

        // ==========================================
        // 运行 Code Predictor 自回归生成其余 15 个 codes
        // ==========================================
        predictor_kv_cache_.reset();

        // --- Predictor Prefill (seq_len = 2) ---
        ctx.queue().memcpy(p_buf_.hidden.data(), p_buf_.pred_input_proj.data(), 2 * H_pred * sizeof(bf16)).wait();
        ops::rms_norm(ctx, p_buf_.hidden, *predictor_layers_[0].input_ln_weight,
                      predictor_cfg_.rms_norm_eps, p_buf_.normed, 2, H_pred);

        int rotary_dim_pred = predictor_cfg_.head_dim;
        if (rotary_dim_pred & 1) --rotary_dim_pred;

        for (int i = 0; i < predictor_cfg_.num_hidden_layers; i++) {
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
        Tensor final_hidden_pred = Tensor::allocate(ctx, {1, H_pred});
        ctx.queue().memcpy(final_hidden_pred.data(), p_buf_.hidden.data_as<bf16>() + 1 * H_pred, H_pred * sizeof(bf16)).wait();
        
        Tensor final_normed_pred = Tensor::allocate(ctx, {1, H_pred});
        ops::rms_norm(ctx, final_hidden_pred, *predictor_final_norm_weight_, predictor_cfg_.rms_norm_eps, final_normed_pred, 1, H_pred);

        predictor_lm_heads_[0].forward(ctx, final_normed_pred, p_buf_.logits, 1);
        int tok = ops::sample_with_config(ctx, p_buf_.logits, predictor_cfg_.vocab_size, gen_config, {});
        frame_codes[1] = tok;

        predictor_kv_cache_.advance(2);

        // --- Predictor Decode loop (step 2 to 15, i.e., cb_idx = 1 to 14) ---
        for (int cb_idx = 1; cb_idx < 15; cb_idx++) {
            Tensor tok_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
            ctx.memcpy_h2d(tok_dev.data(), &tok, sizeof(int));
            Tensor emb_h = Tensor::allocate(ctx, {1, H_talker});
            ops::embedding_lookup(ctx, *predictor_embed_weights_[cb_idx - 1], tok_dev.data_as<int>(), 1, emb_h, H_talker);

            Tensor emb_pred = Tensor::allocate(ctx, {1, H_pred});
            if (has_predictor_projection_) {
                predictor_projection_linear_.forward_bias(ctx, emb_h, *predictor_projection_bias_, emb_pred, 1);
            } else {
                ctx.queue().memcpy(emb_pred.data(), emb_h.data(), H_pred * sizeof(bf16)).wait();
            }

            // Copy to single slot in p_buf_.hidden at position cb_idx + 1 (since prefill took slots 0, 1)
            ctx.queue().memcpy(p_buf_.hidden.data_as<bf16>() + (cb_idx + 1) * H_pred, emb_pred.data(), H_pred * sizeof(bf16)).wait();
            
            // Norm
            Tensor step_normed = Tensor::allocate(ctx, {1, H_pred});
            ops::rms_norm(ctx, emb_pred, *predictor_layers_[0].input_ln_weight,
                          predictor_cfg_.rms_norm_eps, step_normed, 1, H_pred);

            // Forward layers with decode path (seq_len = 1)
            for (int i = 0; i < predictor_cfg_.num_hidden_layers; i++) {
                auto& L = predictor_layers_[i];
                L.qkv_proj.forward(ctx, step_normed, p_buf_.qkv, 1);

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
                ops::fused_add_rms_norm(ctx, emb_pred, p_buf_.gate, *L.post_attn_ln_weight, predictor_cfg_.rms_norm_eps, step_normed, 1, H_pred);

                L.gate_up_proj.forward(ctx, step_normed, p_buf_.gate_up, 1);
                ops::fused_gate_up_swiglu(ctx, p_buf_.gate_up, p_buf_.gate, FF_pred);

                L.down_proj.forward(ctx, p_buf_.gate, p_buf_.attn_out, 1);
                Tensor* next_input_ln = (i < predictor_cfg_.num_hidden_layers - 1) ? predictor_layers_[i + 1].input_ln_weight : predictor_final_norm_weight_;
                ops::fused_add_rms_norm(ctx, emb_pred, p_buf_.attn_out, *next_input_ln, predictor_cfg_.rms_norm_eps, step_normed, 1, H_pred);
            }
            ctx.queue().memcpy(p_buf_.hidden.data_as<bf16>() + (cb_idx + 1) * H_pred, emb_pred.data(), H_pred * sizeof(bf16)).wait();

            // Compute logits and sample tok
            ops::rms_norm(ctx, emb_pred, *predictor_final_norm_weight_, predictor_cfg_.rms_norm_eps, final_normed_pred, 1, H_pred);
            predictor_lm_heads_[cb_idx].forward(ctx, final_normed_pred, p_buf_.logits, 1);
            
            tok = ops::sample_with_config(ctx, p_buf_.logits, predictor_cfg_.vocab_size, tts_gen, {});
            frame_codes[cb_idx + 1] = tok;

            predictor_kv_cache_.advance(1);
        }

        // 保存这帧的 16 个 codes
        for (int c : frame_codes) {
            out_codes.push_back(static_cast<int32_t>(c));
        }
        out_n_frames++;

        // ==========================================
        // 运行 Talker Decode (seq_len = 1) 并生成下一个首码
        // ==========================================
        
        // 查找 16 个 codes 的 embeddings
        Tensor frame_tokens_dev = Tensor::allocate(ctx, {16}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(frame_tokens_dev.data(), frame_codes.data(), 16 * sizeof(int));

        Tensor sum_emb = Tensor::allocate(ctx, {1, H_talker});
        
        // 临时分配用于计算和
        Tensor single_emb = Tensor::allocate(ctx, {1, H_talker});

        // 查找 codebook 0
        int c0 = frame_codes[0];
        Tensor c0_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
        ctx.memcpy_h2d(c0_dev.data(), &c0, sizeof(int));
        ops::embedding_lookup(ctx, *talker_codec_embed_weight_, c0_dev.data_as<int>(), 1, sum_emb, H_talker);

        // 查找 predictor 对应的 15 个 embeddings 并累加
        for (int i = 0; i < 15; i++) {
            int ci = frame_codes[i + 1];
            Tensor ci_dev = Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
            ctx.memcpy_h2d(ci_dev.data(), &ci, sizeof(int));
            ops::embedding_lookup(ctx, *predictor_embed_weights_[i], ci_dev.data_as<int>(), 1, single_emb, H_talker);
            
            // 累加：sum_emb += single_emb
            bf16* sum_ptr = sum_emb.data_as<bf16>();
            bf16* sgl_ptr = single_emb.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> idx) {
                sum_ptr[idx[0]] = sum_ptr[idx[0]] + sgl_ptr[idx[0]];
            });
            ctx.queue().wait();
        }

        // 加上 trailing_text_hidden[gen_step] 或 tts_pad_embed（对齐 ggml）
        Tensor add_vec = Tensor::allocate(ctx, {1, H_talker});
        if (gen_step < trailing_len) {
            // 使用预计算的 trailing text hidden state（正文 token 的 ResizeMLP 投影）
            ctx.queue().memcpy(add_vec.data(),
                trailing_text_hidden.data() + gen_step * H_talker,
                H_talker * sizeof(bf16)).wait();
        } else {
            // 超出 trailing text 长度后回落到 tts_pad_embed
            ctx.queue().memcpy(add_vec.data(), tts_pad_embed.data(), H_talker * sizeof(bf16)).wait();
        }

        bf16* sum_ptr = sum_emb.data_as<bf16>();
        bf16* add_ptr = add_vec.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(H_talker), [=](sycl::id<1> idx) {
            sum_ptr[idx[0]] = sum_ptr[idx[0]] + add_ptr[idx[0]];
        });
        ctx.queue().wait();

        // 运行 Talker Decode Step (seq_len = 1)
        Tensor step_normed_talker = Tensor::allocate(ctx, {1, H_talker});
        ops::rms_norm(ctx, sum_emb, *talker_layers_[0].input_ln_weight,
                      talker_cfg_.rms_norm_eps, step_normed_talker, 1, H_talker);

        int current_pos = current_talker_len_;

        for (int i = 0; i < talker_cfg_.num_hidden_layers; i++) {
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
        Tensor final_normed_talker = Tensor::allocate(ctx, {1, H_talker});
        ops::rms_norm(ctx, sum_emb, *talker_final_norm_weight_, talker_cfg_.rms_norm_eps, final_normed_talker, 1, H_talker);

        // 保存新的 past_hidden_talker (保存归一化后的值以对齐 Python)
        ctx.queue().memcpy(past_hidden_talker.data(), final_normed_talker.data(), H_talker * sizeof(bf16)).wait();

        talker_codec_head_.forward(ctx, final_normed_talker, t_buf_.logits, 1);
        token = ops::sample_with_config(ctx, t_buf_.logits, talker_cfg_.vocab_size, tts_gen, generated_cb0_tokens);

        if (token != eos_id) {
            generated_cb0_tokens.push_back(token);
        }
        talker_kv_cache_.advance(1);
        current_talker_len_++;
        gen_step++;
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
    mimi_loaded_ = true;
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

    if (codes.size() < static_cast<size_t>(n_frames * 16)) {
        AILA_LOG_ERROR("[MimiDecoder] Error: input codes size does not match n_frames * 16.");
        return false;
    }

    const char* env_debug = std::getenv("AILA_MIMI_DEBUG");
    bool debug_print = (env_debug && std::string(env_debug) == "1");

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

    // 0. 将 codes 复制 to GPU 设备端
    Tensor codes_dev = Tensor::allocate(ctx, {n_frames, 16}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(codes_dev.data(), codes.data(), n_frames * 16 * sizeof(int32_t));
    ctx.synchronize();

    // ==========================================
    // 1. VQ 解量化查表与投影
    // ==========================================
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
    ctx.synchronize();

    // 线性投影 MatMul (把 [512, 256, 1] 形状的权重 reshape 成 2D 的 [512, 256])
    Tensor first_proj_w = mimi_weights_.get("decoder.quantizer.rvq_first.output_proj.weight").reshape_view({512, 256});
    Tensor rest_proj_w = mimi_weights_.get("decoder.quantizer.rvq_rest.output_proj.weight").reshape_view({512, 256});

    Linear first_proj, rest_proj;
    first_proj.init(ctx, first_proj_w, 256, 512, false);
    rest_proj.init(ctx, rest_proj_w, 256, 512, false);

    Tensor proj_first = Tensor::allocate(ctx, {n_frames, 512});
    Tensor proj_rest = Tensor::allocate(ctx, {n_frames, 512});

    first_proj.forward(ctx, temp_first, proj_first, n_frames);
    rest_proj.forward(ctx, temp_rest, proj_rest, n_frames);
    ctx.synchronize();

    // 向量相加得到 latent
    Tensor latent = Tensor::allocate(ctx, {n_frames, 512});
    ops::copy_tensor(ctx, proj_first, latent, n_frames * 512);
    ops::residual_add(ctx, latent, proj_rest, n_frames * 512);
    ctx.synchronize();

    // ==========================================
    // 2. Pre-conv 卷积 [3, 512, 1024]
    // ==========================================
    Tensor pre_conv_out = Tensor::allocate(ctx, {n_frames, 1024});
    Tensor& pre_conv_w = mimi_weights_.get("decoder.pre_conv.conv.weight");
    Tensor& pre_conv_b = mimi_weights_.get("decoder.pre_conv.conv.bias");
    ops::causal_conv1d(ctx, latent, pre_conv_w, pre_conv_b, pre_conv_out, 1, 512, 1024, n_frames, 3, 1);
    ctx.synchronize();

    // ==========================================
    // 3. Pre-transformer (8 Layers Causal Attention)
    // ==========================================
    Tensor pre_tfm_in = Tensor::allocate(ctx, {n_frames, 512});
    Linear pre_tfm_in_proj;
    pre_tfm_in_proj.init(ctx, mimi_weights_.get("decoder.pre_transformer.input_proj.weight"), 1024, 512, false);
    Tensor& pre_tfm_in_proj_b = mimi_weights_.get("decoder.pre_transformer.input_proj.bias");
    pre_tfm_in_proj.forward_bias(ctx, pre_conv_out, pre_tfm_in_proj_b, pre_tfm_in, n_frames);
    ctx.synchronize();

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

    AILA_LOG_INFO("[MimiDebug] Step 6: Pre-transformer 8 layers loop");
    Tensor x = Tensor::allocate(ctx, {n_frames, 512});
    ops::copy_tensor(ctx, pre_tfm_in, x, n_frames * 512);

    for (int l = 0; l < 8; ++l) {
        std::string layer_prefix = "decoder.pre_transformer.layers." + std::to_string(l) + ".";

        Tensor residual = Tensor::allocate(ctx, {n_frames, 512});
        ops::copy_tensor(ctx, x, residual, n_frames * 512);
        ctx.synchronize();

        // input rms norm
        Tensor normed = Tensor::allocate(ctx, {n_frames, 512});
        Tensor& input_ln_w = mimi_weights_.get(layer_prefix + "input_layernorm.weight");
        ops::rms_norm(ctx, x, input_ln_w, 1e-5f, normed, n_frames, 512);
        ctx.synchronize();

        // Q/K/V Linear (无 bias)
        Linear q_proj, k_proj, v_proj;
        q_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.q_proj.weight"), 512, 1024, false);
        k_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.k_proj.weight"), 512, 1024, false);
        v_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.v_proj.weight"), 512, 1024, false);

        Tensor q = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor k = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor v = Tensor::allocate(ctx, {n_frames, 1024});

        q_proj.forward(ctx, normed, q, n_frames);
        k_proj.forward(ctx, normed, k, n_frames);
        v_proj.forward(ctx, normed, v, n_frames);
        ctx.synchronize();

        // Apply RoPE positions (num_heads=16, head_dim=64)
        ops::apply_rope(ctx, q, k, n_frames, 0, 16, 16, 64, 10000.0f);
        ctx.synchronize();

        // Attention Prefill (num_heads=16, head_dim=64 -> output_dim=1024)
        Tensor attn_out = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor scores_buf = Tensor::allocate(ctx, {16, n_frames, n_frames}, dnnl::memory::data_type::f32);
        ops::attention_prefill(ctx, q, k, v, attn_out, scores_buf, n_frames, 16, 16, 64);
        ctx.synchronize();

        // Out proj
        Linear o_proj;
        o_proj.init(ctx, mimi_weights_.get(layer_prefix + "self_attn.o_proj.weight"), 1024, 512, false);
        Tensor proj_out = Tensor::allocate(ctx, {n_frames, 512});
        o_proj.forward(ctx, attn_out, proj_out, n_frames);
        ctx.synchronize();

        // Attention layer scale
        Tensor& attn_scale = mimi_weights_.get(layer_prefix + "self_attn_layer_scale.scale");
        apply_layer_scale_gpu(proj_out, attn_scale, n_frames, 512);
        ctx.synchronize();

        // Add to residual
        ops::residual_add(ctx, residual, proj_out, n_frames * 512);
        ops::copy_tensor(ctx, residual, x, n_frames * 512);
        ctx.synchronize();

        // MLP
        Tensor mlp_residual = Tensor::allocate(ctx, {n_frames, 512});
        ops::copy_tensor(ctx, x, mlp_residual, n_frames * 512);
        ctx.synchronize();

        Tensor normed_post = Tensor::allocate(ctx, {n_frames, 512});
        Tensor& post_attn_ln_w = mimi_weights_.get(layer_prefix + "post_attention_layernorm.weight");
        ops::rms_norm(ctx, x, post_attn_ln_w, 1e-5f, normed_post, n_frames, 512);
        ctx.synchronize();

        Linear gate_proj, up_proj, down_proj;
        gate_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.gate_proj.weight"), 512, 1024, false);
        up_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.up_proj.weight"), 512, 1024, false);

        Tensor gate_out = Tensor::allocate(ctx, {n_frames, 1024});
        Tensor up_out = Tensor::allocate(ctx, {n_frames, 1024});

        gate_proj.forward(ctx, normed_post, gate_out, n_frames);
        up_proj.forward(ctx, normed_post, up_out, n_frames);
        ctx.synchronize();

        // SwiGLU activation: gate = silu(gate) * up
        ops::swiglu(ctx, gate_out, up_out, gate_out, n_frames * 1024);
        ctx.synchronize();

        down_proj.init(ctx, mimi_weights_.get(layer_prefix + "mlp.down_proj.weight"), 1024, 512, false);
        Tensor down_out = Tensor::allocate(ctx, {n_frames, 512});
        down_proj.forward(ctx, gate_out, down_out, n_frames);
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
    Linear pre_tfm_out_proj;
    pre_tfm_out_proj.init(ctx, mimi_weights_.get("decoder.pre_transformer.output_proj.weight"), 512, 1024, false);
    Tensor& pre_tfm_out_proj_b = mimi_weights_.get("decoder.pre_transformer.output_proj.bias");
    pre_tfm_out_proj.forward_bias(ctx, final_normed, pre_tfm_out_proj_b, pre_tfm_out, n_frames);
    ctx.synchronize();

    // ==========================================
    // 4. 两层 ConvNeXt 上采样 (总共上采样 4 倍)
    // ==========================================
    Tensor upsample_in = Tensor::allocate(ctx, {n_frames, 1024});
    ops::copy_tensor(ctx, pre_tfm_out, upsample_in, n_frames * 1024);
    ctx.synchronize();
    int L = n_frames;

    for (int i = 0; i < 2; ++i) {
        std::string up_prefix = "decoder.upsample." + std::to_string(i) + ".";

        // 转置卷积上采样 2 倍: kernel_size=2, stride=2, output_channels=1024
        Tensor conv_t_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& conv_w = mimi_weights_.get(up_prefix + "0.conv.weight");
        Tensor& conv_b = mimi_weights_.get(up_prefix + "0.conv.bias");
        ops::causal_conv_transpose1d(ctx, upsample_in, conv_w, conv_b, conv_t_out, 1, 1024, 1024, L, 2, 2);
        ctx.synchronize();

        // ConvNeXt block
        Tensor dw_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& dw_w = mimi_weights_.get(up_prefix + "1.dwconv.conv.weight");
        Tensor& dw_b = mimi_weights_.get(up_prefix + "1.dwconv.conv.bias");
        ops::causal_conv1d_dw(ctx, conv_t_out, dw_w, dw_b, dw_out, 1, 1024, 2 * L, 7, 1);
        ctx.synchronize();

        // LayerNorm
        Tensor normed_dw = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& norm_w = mimi_weights_.get(up_prefix + "1.norm.weight");
        Tensor& norm_b = mimi_weights_.get(up_prefix + "1.norm.bias");
        ops::layer_norm(ctx, dw_out, norm_w, norm_b, 1e-6f, normed_dw, 2 * L, 1024);
        ctx.synchronize();

        // Linear pwconv1
        Linear pw1;
        pw1.init(ctx, mimi_weights_.get(up_prefix + "1.pwconv1.weight"), 1024, 4096, false);
        Tensor pw1_out = Tensor::allocate(ctx, {2 * L, 4096});
        Tensor& pw1_b = mimi_weights_.get(up_prefix + "1.pwconv1.bias");
        pw1.forward_bias(ctx, normed_dw, pw1_b, pw1_out, 2 * L);
        ctx.synchronize();

        // GELU
        ops::gelu_tanh_inplace(ctx, pw1_out, 2 * L * 4096);
        ctx.synchronize();

        // Linear pwconv2
        Linear pw2;
        pw2.init(ctx, mimi_weights_.get(up_prefix + "1.pwconv2.weight"), 4096, 1024, false);
        Tensor pw2_out = Tensor::allocate(ctx, {2 * L, 1024});
        Tensor& pw2_b = mimi_weights_.get(up_prefix + "1.pwconv2.bias");
        pw2.forward_bias(ctx, pw1_out, pw2_b, pw2_out, 2 * L);
        ctx.synchronize();

        // Scale by gamma and add to residual
        Tensor& gamma = mimi_weights_.get(up_prefix + "1.gamma");
        apply_layer_scale_gpu(pw2_out, gamma, 2 * L, 1024);
        ctx.synchronize();

        ops::residual_add(ctx, conv_t_out, pw2_out, 2 * L * 1024);
        ctx.synchronize();

        L = 2 * L;
        upsample_in = Tensor::allocate(ctx, {L, 1024});
        ops::copy_tensor(ctx, conv_t_out, upsample_in, L * 1024);
        ctx.synchronize();
    }

    // ==========================================
    // 5. Decoder Blocks (4 Blocks, 总共上采样 480 倍)
    // ==========================================
    // 先通过 dec0 一维卷积 [7, 1024, 1536]
    Tensor dec0_out = Tensor::allocate(ctx, {L, 1536});
    Tensor& dec0_w = mimi_weights_.get("decoder.decoder.0.conv.weight");
    Tensor& dec0_b = mimi_weights_.get("decoder.decoder.0.conv.bias");
    ops::causal_conv1d(ctx, upsample_in, dec0_w, dec0_b, dec0_out, 1, 1024, 1536, L, 7, 1);
    ctx.synchronize();

    Tensor dec_in = Tensor::allocate(ctx, {L, 1536});
    ops::copy_tensor(ctx, dec0_out, dec_in, L * 1536);
    ctx.synchronize();

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
        ops::snake_beta(ctx, dec_in, snake_a, snake_b, dec_in, L * in_d, in_d, L);
        ctx.synchronize();

        // 2. Transposed Convolution
        Tensor conv_t_out = Tensor::allocate(ctx, {L * stride, out_d});
        Tensor& conv_t_w = mimi_weights_.get(dec_prefix + "1.conv.weight");
        Tensor& conv_t_b = mimi_weights_.get(dec_prefix + "1.conv.bias");
        ops::causal_conv_transpose1d(ctx, dec_in, conv_t_w, conv_t_b, conv_t_out, 1, in_d, out_d, L, kernel, stride);
        ctx.synchronize();

        // 3. 3 Residual blocks with dilations 1, 3, 9
        int dilations[3] = {1, 3, 9};
        Tensor xx = Tensor::allocate(ctx, {L * stride, out_d});
        ops::copy_tensor(ctx, conv_t_out, xx, L * stride * out_d);
        ctx.synchronize();

        for (int r = 0; r < 3; ++r) {
            std::string res_prefix = dec_prefix + std::to_string(r + 2) + ".";

            Tensor res_in = Tensor::allocate(ctx, {L * stride, out_d});
            ops::copy_tensor(ctx, xx, res_in, L * stride * out_d);
            ctx.synchronize();

            // act1
            Tensor xx_act1 = Tensor::allocate(ctx, {L * stride, out_d});
            Tensor& act1_a = mimi_weights_.get(res_prefix + "act1.alpha");
            Tensor& act1_b = mimi_weights_.get(res_prefix + "act1.beta");
            ops::snake_beta(ctx, xx, act1_a, act1_b, xx_act1, L * stride * out_d, out_d, L * stride);
            ctx.synchronize();

            // conv1 [7, out_d, out_d] dilation
            Tensor conv1_out = Tensor::allocate(ctx, {L * stride, out_d});
            Tensor& conv1_w = mimi_weights_.get(res_prefix + "conv1.conv.weight");
            Tensor& conv1_b = mimi_weights_.get(res_prefix + "conv1.conv.bias");
            ops::causal_conv1d(ctx, xx_act1, conv1_w, conv1_b, conv1_out, 1, out_d, out_d, L * stride, 7, dilations[r]);
            ctx.synchronize();

            // act2
            Tensor& act2_a = mimi_weights_.get(res_prefix + "act2.alpha");
            Tensor& act2_b = mimi_weights_.get(res_prefix + "act2.beta");
            ops::snake_beta(ctx, conv1_out, act2_a, act2_b, conv1_out, L * stride * out_d, out_d, L * stride);
            ctx.synchronize();

            // conv2 [1, out_d, out_d]
            Tensor conv2_out = Tensor::allocate(ctx, {L * stride, out_d});
            Tensor& conv2_w = mimi_weights_.get(res_prefix + "conv2.conv.weight");
            Tensor& conv2_b = mimi_weights_.get(res_prefix + "conv2.conv.bias");
            ops::causal_conv1d(ctx, conv1_out, conv2_w, conv2_b, conv2_out, 1, out_d, out_d, L * stride, 1, 1);
            ctx.synchronize();

            // Add to residual
            ops::residual_add(ctx, res_in, conv2_out, L * stride * out_d);
            ops::copy_tensor(ctx, res_in, xx, L * stride * out_d);
            ctx.synchronize();
        }

        L = L * stride;
        dec_in = Tensor::allocate(ctx, {L, out_d});
        ops::copy_tensor(ctx, xx, dec_in, L * out_d);
        ctx.synchronize();
    }

    // ==========================================
    // 6. 最终 SnakeBeta 和映射 (channels=3 -> 1)
    // ==========================================
    // 最终 SnakeBeta (输入维度是最后的 out_dims[3] 即 96)
    Tensor& dec5_a = mimi_weights_.get("decoder.decoder.5.alpha");
    Tensor& dec5_b = mimi_weights_.get("decoder.decoder.5.beta");
    ops::snake_beta(ctx, dec_in, dec5_a, dec5_b, dec_in, L * 96, 96, L);
    ctx.synchronize();

    // dec6一维卷积 [7, 96, 1]
    Tensor dec6_out = Tensor::allocate(ctx, {L, 1});
    Tensor& dec6_w = mimi_weights_.get("decoder.decoder.6.conv.weight");
    Tensor& dec6_b = mimi_weights_.get("decoder.decoder.6.conv.bias");
    ops::causal_conv1d(ctx, dec_in, dec6_w, dec6_b, dec6_out, 1, 96, 1, L, 7, 1);
    ctx.synchronize();

    // tanh 修正为 clamp 激活并拷贝到 Host
    out_samples.resize(L);
    float* host_ptr = out_samples.data();
    auto* dev_ptr = dec6_out.data_as<bf16>();

    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(L), [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            float val = static_cast<float>(dev_ptr[i]);
            // PyTorch clamp(min=-1, max=1)
            if (val < -1.0f) val = -1.0f;
            else if (val > 1.0f) val = 1.0f;
            dev_ptr[i] = bf16(val);
        });
    }).wait();

    // Now copy back as bf16 and convert to float on Host
    std::vector<bf16> cpu_bf16(L);
    ctx.queue().memcpy(cpu_bf16.data(), dev_ptr, L * sizeof(bf16)).wait();

    for (int i = 0; i < L; ++i) {
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

    {
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

    AILA_LOG_INFO("[MimiDebug] Successful! Decoded into %d samples.", L);
    return true;
}


