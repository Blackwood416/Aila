#include "DiffusionGemmaBnb4Backend.hpp"

#include "../utils/SafeTensors.hpp"
#include "../utils/EnvUtils.hpp"
#include "core/Context.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace {

bool load_weight_ref(Context& ctx, ModelWeights& weights,
                     const std::string& name, Bnb4BitWeightRef& out,
                     int in_features, std::string* error_message) {
    if (!weights.has(name)) {
        if (error_message) {
            *error_message = "DiffusionGemmaBnb4Backend: missing weight: " + name;
        }
        return false;
    }
    if (!LoadBnb4BitWeightRef(ctx, weights, name, out, error_message,
                              in_features)) {
        return false;
    }
    return true;
}

void router_topk_from_bf16_scores(
    const std::vector<sycl::ext::oneapi::bfloat16>& score_values,
    const std::vector<sycl::ext::oneapi::bfloat16>& per_expert_scale,
    int tokens, int num_experts, int top_k, std::vector<int>& expert_ids,
    std::vector<float>& expert_weights) {
    expert_ids.assign(static_cast<size_t>(tokens) * top_k, 0);
    expert_weights.assign(static_cast<size_t>(tokens) * top_k, 0.0f);
    for (int token = 0; token < tokens; ++token) {
        const auto* row = score_values.data() +
                          static_cast<size_t>(token) * num_experts;
        float max_score = -1.0e30f;
        for (int expert = 0; expert < num_experts; ++expert) {
            max_score = std::max(max_score, static_cast<float>(row[expert]));
        }
        std::vector<std::pair<float, int>> ranked(
            static_cast<size_t>(num_experts));
        float probability_sum = 0.0f;
        for (int expert = 0; expert < num_experts; ++expert) {
            const float value =
                std::exp(static_cast<float>(row[expert]) - max_score);
            ranked[static_cast<size_t>(expert)] = {value, expert};
            probability_sum += value;
        }
        const float inv_probability_sum = 1.0f / probability_sum;
        for (auto& [value, expert] : ranked) {
            (void)expert;
            value *= inv_probability_sum;
        }
        std::partial_sort(
            ranked.begin(), ranked.begin() + top_k, ranked.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.first > rhs.first;
            });
        float selected_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            selected_sum += ranked[static_cast<size_t>(k)].first;
        }
        for (int k = 0; k < top_k; ++k) {
            const auto [probability, expert] =
                ranked[static_cast<size_t>(k)];
            const size_t output_index =
                static_cast<size_t>(token) * top_k + k;
            expert_ids[output_index] = expert;
            expert_weights[output_index] =
                probability / selected_sum *
                static_cast<float>(
                    per_expert_scale[static_cast<size_t>(expert)]);
        }
    }
}

} // namespace

bool DiffusionGemmaBnb4Backend::load(Context& ctx, ModelWeights& weights,
                                     const ModelSpec& spec, int max_seq_len,
                                     std::string* error_message) {
    if (spec.family != ModelFamily::DiffusionGemma) {
        if (error_message) {
            *error_message = "DiffusionGemmaBnb4Backend: invalid model family";
        }
        return false;
    }
    if (!spec.is_bitsandbytes_4bit()) {
        if (error_message) {
            *error_message =
                "DiffusionGemmaBnb4Backend: model is not a bitsandbytes 4-bit checkpoint";
        }
        return false;
    }

    spec_ = spec;
    cfg_ = spec.diffusion_gemma;
    const auto& tc = cfg_.text;
    if (static_cast<int>(tc.layer_types.size()) != tc.num_hidden_layers) {
        if (error_message) {
            *error_message =
                "DiffusionGemmaBnb4Backend: layer_types does not match num_hidden_layers";
        }
        return false;
    }

    // A770 16GB budget: cap the KV cache to 384 by default; users may raise
    // it via --max-seq at their own risk.
    max_seq_len_ = max_seq_len > 0 ? max_seq_len : 384;
    max_seq_len_ = std::min(max_seq_len_, 384);
    if (max_seq_len > 384) {
        AILA_LOG_WARN(
            "[DiffusionGemma] Clamping max_seq_len %d -> 384 to fit A770 VRAM",
            max_seq_len);
    }

    // DiffusionGemmaTextScaledWordEmbedding stores the scale as fp32 but
    // explicitly casts it to the bf16 embedding dtype before multiplying.
    // Preserve that observable boundary instead of multiplying by the exact
    // fp32 square root.
    embed_scale_ = static_cast<float>(
        sycl::ext::oneapi::bfloat16(
            std::sqrt(static_cast<float>(tc.hidden_size))));

    const std::string embed_name =
        "model.encoder.language_model.embed_tokens.weight";
    if (!weights.has(embed_name)) {
        if (error_message) {
            *error_message = "DiffusionGemmaBnb4Backend: missing " + embed_name;
        }
        return false;
    }
    embed_weight_ = &weights.get(embed_name);
    if (embed_weight_->dtype() != dnnl::memory::data_type::bf16) {
        if (error_message) {
            *error_message = "DiffusionGemmaBnb4Backend: embed_tokens must be bf16";
        }
        return false;
    }
    final_norm_weight_ = &weights.get("model.encoder.language_model.norm.weight");

    Bnb4BitWeightRef sc_gate_ref;
    Bnb4BitWeightRef sc_up_ref;
    Bnb4BitWeightRef sc_down_ref;
    if (!load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.gate_proj.weight",
                         sc_gate_ref, tc.hidden_size, error_message) ||
        !load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.up_proj.weight",
                         sc_up_ref, tc.hidden_size, error_message) ||
        !load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.down_proj.weight",
                         sc_down_ref, tc.intermediate_size, error_message) ||
        !sc_gate_.init(ctx, sc_gate_ref, error_message) ||
        !sc_up_.init(ctx, sc_up_ref, error_message) ||
        !sc_down_.init(ctx, sc_down_ref, error_message)) {
        return false;
    }
    sc_pre_norm_ = &weights.get("model.decoder.self_conditioning.pre_norm.weight");

    layers_.clear();
    runtimes_.clear();
    moes_.clear();
    layers_.reserve(static_cast<size_t>(tc.num_hidden_layers));
    runtimes_.reserve(static_cast<size_t>(tc.num_hidden_layers));
    moes_.reserve(static_cast<size_t>(tc.num_hidden_layers));

    for (int layer_idx = 0; layer_idx < tc.num_hidden_layers; ++layer_idx) {
        const std::string prefix =
            "model.encoder.language_model.layers." + std::to_string(layer_idx) + ".";
        const std::string attn = prefix + "self_attn.";
        const std::string mlp = prefix + "mlp.";
        const std::string experts = prefix + "experts.";
        const std::string router = prefix + "router.";

        DiffusionGemmaLayerWeights layer;
        const int head_dim = tc.head_dim_for_layer(layer_idx);
        const int num_heads = tc.num_attention_heads;
        const int num_kv_heads = tc.kv_heads_for_layer(layer_idx);
        const bool sliding = tc.is_sliding_layer(layer_idx);
        const int64_t expected_q = static_cast<int64_t>(num_heads) * head_dim;
        const int64_t expected_kv = static_cast<int64_t>(num_kv_heads) * head_dim;
        const int64_t expected_moe_gate_up =
            static_cast<int64_t>(tc.num_experts) * 2 * tc.moe_intermediate_size;
        const int64_t expected_moe_down =
            static_cast<int64_t>(tc.num_experts) * tc.hidden_size;

        if (!load_weight_ref(ctx, weights, attn + "q_proj.weight",
                             layer.q_proj_ref, tc.hidden_size, error_message) ||
            !load_weight_ref(ctx, weights, attn + "k_proj.weight",
                             layer.k_proj_ref, tc.hidden_size, error_message) ||
            (sliding && !load_weight_ref(ctx, weights, attn + "v_proj.weight",
                                         layer.v_proj_ref, tc.hidden_size,
                                         error_message)) ||
            !load_weight_ref(ctx, weights, attn + "o_proj.weight",
                             layer.o_proj_ref, static_cast<int>(expected_q),
                             error_message) ||
            !load_weight_ref(ctx, weights, mlp + "gate_proj.weight",
                             layer.mlp_gate_up_ref, tc.hidden_size,
                             error_message) ||
            !load_weight_ref(ctx, weights, mlp + "up_proj.weight",
                             layer.mlp_up_ref, tc.hidden_size,
                             error_message) ||
            !load_weight_ref(ctx, weights, mlp + "down_proj.weight",
                             layer.mlp_down_ref, tc.intermediate_size,
                             error_message) ||
            !load_weight_ref(ctx, weights, experts + "gate_up_proj",
                             layer.moe_gate_up_ref, tc.hidden_size,
                             error_message) ||
            !load_weight_ref(ctx, weights, experts + "down_proj",
                             layer.moe_down_ref, tc.moe_intermediate_size,
                             error_message)) {
            return false;
        }

        if (layer.q_proj_ref.logical_out_features() != expected_q ||
            layer.k_proj_ref.logical_out_features() != expected_kv ||
            (sliding &&
             layer.v_proj_ref.logical_out_features() != expected_kv) ||
            layer.o_proj_ref.logical_out_features() != tc.hidden_size ||
            layer.mlp_gate_up_ref.logical_out_features() != tc.intermediate_size ||
            layer.mlp_up_ref.logical_out_features() != tc.intermediate_size ||
            layer.mlp_down_ref.logical_out_features() != tc.hidden_size ||
            layer.moe_gate_up_ref.logical_out_features() != expected_moe_gate_up ||
            layer.moe_down_ref.logical_out_features() != expected_moe_down) {
            if (error_message) {
                *error_message =
                    "DiffusionGemmaBnb4Backend: layer " +
                    std::to_string(layer_idx) + " weight shape mismatch";
            }
            return false;
        }

        if (!layer.q_proj.init(ctx, layer.q_proj_ref, error_message) ||
            !layer.k_proj.init(ctx, layer.k_proj_ref, error_message) ||
            (sliding && !layer.v_proj.init(ctx, layer.v_proj_ref, error_message)) ||
            !layer.o_proj.init(ctx, layer.o_proj_ref, error_message) ||
            !layer.mlp_gate.init(ctx, layer.mlp_gate_up_ref, error_message) ||
            !layer.mlp_up.init(ctx, layer.mlp_up_ref, error_message) ||
            !layer.mlp_down.init(ctx, layer.mlp_down_ref, error_message)) {
            return false;
        }

        auto moe = std::make_unique<Bnb4GroupedMoE>();
        if (!moe->init(ctx, layer.moe_gate_up_ref, layer.moe_down_ref,
                       tc.hidden_size, tc.moe_intermediate_size, tc.num_experts,
                       error_message)) {
            return false;
        }

        layer.input_ln = &weights.get(prefix + "input_layernorm.weight");
        layer.post_attn_ln = &weights.get(prefix + "post_attention_layernorm.weight");
        layer.pre_ff_ln = &weights.get(prefix + "pre_feedforward_layernorm.weight");
        layer.post_ff_ln = &weights.get(prefix + "post_feedforward_layernorm.weight");
        layer.pre_ff_ln_2 = &weights.get(prefix + "pre_feedforward_layernorm_2.weight");
        layer.post_ff_ln_1 = &weights.get(prefix + "post_feedforward_layernorm_1.weight");
        layer.post_ff_ln_2 = &weights.get(prefix + "post_feedforward_layernorm_2.weight");
        layer.q_norm = &weights.get(attn + "q_norm.weight");
        layer.k_norm = &weights.get(attn + "k_norm.weight");
        layer.router_proj = &weights.get(router + "proj.weight");
        layer.router_scale = &weights.get(router + "scale");
        layer.per_expert_scale = &weights.get(router + "per_expert_scale");
        const std::string layer_scalar_name = prefix + "layer_scalar";
        if (weights.has(layer_scalar_name)) {
            Tensor& scalar_tensor = weights.get(layer_scalar_name);
            if (scalar_tensor.dtype() != dnnl::memory::data_type::bf16 ||
                scalar_tensor.numel() != 1) {
                if (error_message) {
                    *error_message =
                        "DiffusionGemmaBnb4Backend: invalid layer_scalar: " +
                        layer_scalar_name;
                }
                return false;
            }
            bf16 scalar_value;
            ctx.memcpy_d2h(&scalar_value, scalar_tensor.data(), sizeof(bf16));
            layer.layer_scalar = static_cast<float>(scalar_value);
        }
        layer.router_linear.init(ctx, *layer.router_proj, tc.hidden_size,
                                 tc.num_experts, false, false);

        LayerRuntime runtime;
        runtime.k_cache = Tensor::allocate(
            ctx, {num_kv_heads, max_seq_len_, head_dim},
            dnnl::memory::data_type::bf16);
        runtime.v_cache = Tensor::allocate(
            ctx, {num_kv_heads, max_seq_len_, head_dim},
            dnnl::memory::data_type::bf16);

        layers_.push_back(std::move(layer));
        runtimes_.push_back(std::move(runtime));
        moes_.push_back(std::move(moe));
    }

    current_len_ = 0;
    AILA_LOG_INFO(
        "[DiffusionGemma] Loaded %d layers (hidden=%d heads=%d kv=%d window=%d "
        "experts=%d top_k=%d canvas=%d max_seq=%d)",
        tc.num_hidden_layers, tc.hidden_size, tc.num_attention_heads,
        tc.num_key_value_heads, tc.sliding_window, tc.num_experts,
        tc.top_k_experts, cfg_.canvas_length, max_seq_len_);
    return true;
}

Tensor& DiffusionGemmaBnb4Backend::forward(Context& ctx,
                                           const int* token_ids_device,
                                           int seq_len) {
    (void)ctx;
    (void)token_ids_device;
    (void)seq_len;
    throw std::runtime_error(
        "DiffusionGemmaBnb4Backend::forward is not used; the block-diffusion "
        "generation path must be used instead");
}

void DiffusionGemmaBnb4Backend::reset() {
    current_len_ = 0;
}

bool DiffusionGemmaBnb4Backend::truncate_kv_cache(int new_len) {
    (void)new_len;
    return false;  // full reset required
}

bool DiffusionGemmaBnb4Backend::diffusion_encode(Context& ctx,
                                                 const int* token_ids_device,
                                                 int seq_len,
                                                 std::string* error_message) {
    if (seq_len <= 0 || current_len_ + seq_len > max_seq_len_) {
        if (error_message) {
            *error_message = "DiffusionGemmaBnb4Backend: context overflow";
        }
        return false;
    }
    const auto& tc = cfg_.text;
    const int start_pos = current_len_;
    ensure_buffers(ctx, seq_len, start_pos + seq_len);

    ops::embedding_lookup(ctx, *embed_weight_, token_ids_device, seq_len,
                          buf_.hidden, tc.hidden_size);
    ops::scale_inplace(ctx, buf_.hidden, embed_scale_,
                       seq_len * tc.hidden_size);

    for (int layer_idx = 0; layer_idx < tc.num_hidden_layers; ++layer_idx) {
        run_layer_forward(ctx, layers_[static_cast<size_t>(layer_idx)],
                          runtimes_[static_cast<size_t>(layer_idx)],
                          buf_.hidden, seq_len, start_pos, false);
    }
    // Ensure all encoder kernels have completed before the caller reuses or
    // reallocates the runtime buffers (in-order queue + USM free is otherwise
    // not safe against pending writes).
    ctx.synchronize();
    current_len_ += seq_len;
    return true;
}

Tensor& DiffusionGemmaBnb4Backend::diffusion_decode_canvas(
    Context& ctx, const int* canvas_ids_device,
    const sycl::ext::oneapi::bfloat16* sc_embeds_device, int canvas_len) {
    const auto& tc = cfg_.text;
    const int start_pos = current_len_;
    const int total_len = start_pos + canvas_len;
    ensure_buffers(ctx, canvas_len, total_len);
    ensure_logits_buffers(ctx);
    const bool profile = aila::env::read_flag("AILA_PROFILE_DG", false);
    profile_totals_ = {};
    const bool debug_probes =
        aila::env::read_flag("AILA_DG_DEBUG_PROBES", false);
    const std::string dump_dir =
        aila::env::read_string("AILA_DG_DUMP_DIR", "");
    const bool dump_decoder =
        aila::env::read_flag("AILA_DG_DUMP_DECODER", false) &&
        !dump_dir.empty();
    const auto dump_decoder_tensor = [&](const char* tag, Tensor& tensor) {
        if (!dump_decoder) return;
        ctx.synchronize();
        std::vector<uint8_t> bytes(static_cast<size_t>(tensor.numel()) *
                                   tensor.element_size());
        ctx.memcpy_d2h(bytes.data(), tensor.data(), bytes.size());
        std::ofstream out(dump_dir + "/" + tag + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    };

    ops::embedding_lookup(ctx, *embed_weight_, canvas_ids_device, canvas_len,
                          buf_.hidden, tc.hidden_size);
    if (debug_probes) {
        std::vector<bf16> probe(static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                       probe.size() * sizeof(bf16));
        bool bad = false;
        int bad_idx = -1;
        for (size_t i = 0; i < probe.size(); ++i) {
            uint16_t raw = 0;
            std::memcpy(&raw, &probe[i], sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u) {
                bad = true;
                bad_idx = static_cast<int>(i);
                break;
            }
        }
        AILA_LOG_INFO("[DG-DEBUG] decode embed bad=%d idx=%d", bad ? 1 : 0, bad_idx);
    }
    ops::scale_inplace(ctx, buf_.hidden, embed_scale_,
                       canvas_len * tc.hidden_size);
    dump_decoder_tensor("decoder_token_embeds", buf_.hidden);
    if (debug_probes) {
        std::vector<bf16> probe(static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                       probe.size() * sizeof(bf16));
        bool bad = false;
        int bad_idx = -1;
        for (size_t i = 0; i < probe.size(); ++i) {
            uint16_t raw = 0;
            std::memcpy(&raw, &probe[i], sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u) {
                bad = true;
                bad_idx = static_cast<int>(i);
                break;
            }
        }
        AILA_LOG_INFO("[DG-DEBUG] decode scaled bad=%d idx=%d", bad ? 1 : 0, bad_idx);
    }

    // Self-conditioning: pre_norm(soft) -> gate/up -> gelu_tanh -> down ->
    // add to inputs -> post_norm (no learned weight).
    {
        const int hidden = tc.hidden_size;
        if (sc_embeds_device == nullptr) {
            ctx.queue().memset(buf_.sc_normed.data(), 0,
                               static_cast<size_t>(canvas_len) * hidden *
                                   sizeof(sycl::ext::oneapi::bfloat16));
        } else {
            ctx.queue().memcpy(buf_.sc_normed.data(), sc_embeds_device,
                               static_cast<size_t>(canvas_len) * hidden *
                                   sizeof(sycl::ext::oneapi::bfloat16));
        }
        dump_decoder_tensor("decoder_soft_embeds", buf_.sc_normed);
        ops::rms_norm(ctx, buf_.sc_normed, *sc_pre_norm_, tc.rms_norm_eps,
                      buf_.sc_normed, canvas_len, hidden);
        dump_decoder_tensor("decoder_sc_normed", buf_.sc_normed);
        if (debug_probes) {
            std::vector<bf16> probe(
                static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.sc_normed.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            uint16_t max_exp = 0;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                max_exp = std::max(max_exp, static_cast<uint16_t>(raw & 0x7F80u));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            uint16_t row128col0 = 0;
            std::memcpy(&row128col0,
                        &probe[static_cast<size_t>(128) * hidden + 0],
                        sizeof(row128col0));
            AILA_LOG_INFO(
                "[DG-DEBUG] sc_normed bad=%d idx=%d max_exp=0x%04X r128c0=0x%04X",
                bad ? 1 : 0, bad_idx, max_exp, row128col0);
        }
        sc_gate_.forward(ctx, buf_.sc_scratch, buf_.sc_normed, buf_.sc_gate,
                         canvas_len);
        dump_decoder_tensor("decoder_sc_gate", buf_.sc_gate);
        sc_up_.forward(ctx, buf_.sc_scratch, buf_.sc_normed, buf_.sc_up,
                       canvas_len);
        dump_decoder_tensor("decoder_sc_up", buf_.sc_up);
        ops::gelu_tanh_gate_mul(ctx, buf_.sc_gate, buf_.sc_up, buf_.sc_proj,
                                canvas_len * tc.intermediate_size);
        dump_decoder_tensor("decoder_sc_proj", buf_.sc_proj);
        sc_down_.forward(ctx, buf_.sc_scratch, buf_.sc_proj, buf_.sc_down,
                         canvas_len);
        dump_decoder_tensor("decoder_sc_down", buf_.sc_down);
        if (debug_probes) {
            std::vector<bf16> probe(
                static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.sc_down.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            AILA_LOG_INFO("[DG-DEBUG] sc_down bad=%d idx=%d", bad ? 1 : 0,
                          bad_idx);
        }
        if (debug_probes) {
            std::vector<bf16> probe(
                static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            AILA_LOG_INFO("[DG-DEBUG] hidden pre-residual bad=%d idx=%d", bad ? 1 : 0,
                          bad_idx);
        }
        ops::residual_add(ctx, buf_.hidden, buf_.sc_down,
                          canvas_len * hidden);
        if (debug_probes) {
            std::vector<bf16> probe(
                static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            AILA_LOG_INFO("[DG-DEBUG] hidden post-residual bad=%d idx=%d", bad ? 1 : 0,
                          bad_idx);
        }
        ops::rms_norm_unweighted(ctx, buf_.hidden, tc.rms_norm_eps,
                                 buf_.hidden, canvas_len, hidden);
        dump_decoder_tensor("decoder_hidden_after_sc", buf_.hidden);
        if (debug_probes) {
            std::vector<bf16> probe(
                static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            AILA_LOG_INFO("[DG-DEBUG] hidden post-sc-norm bad=%d idx=%d", bad ? 1 : 0,
                          bad_idx);
        }
        if (debug_probes) {
            std::vector<bf16> probe(static_cast<size_t>(canvas_len) * hidden);
            ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                           probe.size() * sizeof(bf16));
            bool bad = false;
            int bad_idx = -1;
            for (size_t i = 0; i < probe.size(); ++i) {
                uint16_t raw = 0;
                std::memcpy(&raw, &probe[i], sizeof(raw));
                if ((raw & 0x7F80u) == 0x7F80u) {
                    bad = true;
                    bad_idx = static_cast<int>(i);
                    break;
                }
            }
            AILA_LOG_INFO("[DG-DEBUG] post-sc hidden bad=%d idx=%d", bad ? 1 : 0,
                          bad_idx);
        }
    }

    for (int layer_idx = 0; layer_idx < tc.num_hidden_layers; ++layer_idx) {
        run_layer_forward(ctx, layers_[static_cast<size_t>(layer_idx)],
                          runtimes_[static_cast<size_t>(layer_idx)],
                          buf_.hidden, canvas_len, start_pos, true);
    }
    if (debug_probes) {
        std::vector<bf16> probe(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(probe.data(), buf_.hidden.data(),
                       probe.size() * sizeof(bf16));
        float mn = 1e30f, mx = -1e30f;
        bool bad = false;
        for (bf16 x : probe) {
            const float f = static_cast<float>(x);
            mn = std::min(mn, f);
            mx = std::max(mx, f);
            uint16_t raw = 0;
            std::memcpy(&raw, &x, sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u) bad = true;
        }
        AILA_LOG_INFO("[DG-DEBUG] post-layers hidden min=%.4f max=%.4f bad=%d", mn, mx,
                      bad ? 1 : 0);
    }
    if (debug_probes) {
        std::vector<bf16> probe_hidden(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(probe_hidden.data(), buf_.hidden.data(),
                       probe_hidden.size() * sizeof(bf16));
        bool bad = false;
        for (bf16 x : probe_hidden) {
            const float f = static_cast<float>(x);
            uint16_t raw = 0;
            std::memcpy(&raw, &x, sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u) {
                bad = true;
                break;
            }
        }
        AILA_LOG_INFO("[DG-DEBUG] pre-final-norm hidden bad=%d", bad ? 1 : 0);
    }
    ops::rms_norm(ctx, buf_.hidden, *final_norm_weight_, tc.rms_norm_eps,
                  buf_.normed, canvas_len, tc.hidden_size);
    dump_decoder_tensor("decoder_final_normed", buf_.normed);
    if (debug_probes) {
        std::vector<bf16> probe_hidden(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(probe_hidden.data(), buf_.hidden.data(),
                       probe_hidden.size() * sizeof(bf16));
        bool bad = false;
        for (bf16 x : probe_hidden) {
            const float f = static_cast<float>(x);
            uint16_t raw = 0;
            std::memcpy(&raw, &x, sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u) {
                bad = true;
                break;
            }
        }
        AILA_LOG_INFO("[DG-DEBUG] post-final-norm hidden bad=%d", bad ? 1 : 0);
        AILA_LOG_INFO("[DG-DEBUG] addr hidden=%p normed=%p first=%04X %04X %04X %04X",
                      buf_.hidden.data(), buf_.normed.data(),
                      static_cast<uint16_t>(probe_hidden[0]),
                      static_cast<uint16_t>(probe_hidden[1]),
                      static_cast<uint16_t>(probe_hidden[2]),
                      static_cast<uint16_t>(probe_hidden[3]));
    }

    if (debug_probes) {
        std::vector<bf16> normed_host(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        std::vector<bf16> hidden_host(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(normed_host.data(), buf_.normed.data(),
                       normed_host.size() * sizeof(bf16));
        ctx.memcpy_d2h(hidden_host.data(), buf_.hidden.data(),
                       hidden_host.size() * sizeof(bf16));
        float mn = 1e30f, mx = -1e30f;
        int bad_index = -1;
        for (size_t i = 0; i < normed_host.size(); ++i) {
            const bf16 x = normed_host[i];
            const float f = static_cast<float>(x);
            mn = std::min(mn, f);
            mx = std::max(mx, f);
            uint16_t raw = 0;
            std::memcpy(&raw, &x, sizeof(raw));
            if ((raw & 0x7F80u) == 0x7F80u && bad_index < 0) {
                bad_index = static_cast<int>(i);
            }
        }
        if (bad_index >= 0) {
            const int idx = bad_index;
            const float hv = static_cast<float>(hidden_host[static_cast<size_t>(idx)]);
            const float nv = static_cast<float>(normed_host[static_cast<size_t>(idx)]);
            union {
                float f;
                uint32_t u;
            } hb, nb;
            hb.f = hv;
            nb.f = nv;
            AILA_LOG_INFO(
                "[DG-DEBUG] normed bad idx=%d hidden=%.6f bits=0x%08X "
                "normed=%.6f bits=0x%08X",
                idx, hv, hb.u, nv, nb.u);
        } else {
            AILA_LOG_INFO("[DG-DEBUG] normed clean min=%.4f max=%.4f", mn, mx);
        }
    }
    if (debug_probes) {
        std::vector<bf16> hidden_host(
            static_cast<size_t>(canvas_len) * tc.hidden_size);
        ctx.memcpy_d2h(hidden_host.data(), buf_.hidden.data(),
                       hidden_host.size() * sizeof(bf16));
        float mn = 1e30f, mx = -1e30f;
        for (bf16 x : hidden_host) {
            const float f = static_cast<float>(x);
            mn = std::min(mn, f);
            mx = std::max(mx, f);
        }
        AILA_LOG_INFO("[DG-DEBUG] pre-norm hidden min=%.4f max=%.4f", mn, mx);
    }

    buf_.lm_head.forward(ctx, buf_.normed, buf_.logits_bf16, canvas_len);
    if (dump_decoder) {
        ctx.synchronize();
        std::vector<bf16> logits_row(static_cast<size_t>(tc.vocab_size));
        ctx.memcpy_d2h(logits_row.data(), buf_.logits_bf16.data(),
                       logits_row.size() * sizeof(bf16));
        std::ofstream out(dump_dir + "/decoder_logits_bf16_row0.bin",
                          std::ios::binary);
        out.write(reinterpret_cast<const char*>(logits_row.data()),
                  static_cast<std::streamsize>(logits_row.size() *
                                               sizeof(bf16)));
    }
    if (debug_probes) {
        std::vector<bf16> logits_host(4096);
        ctx.memcpy_d2h(logits_host.data(), buf_.logits_bf16.data(),
                       logits_host.size() * sizeof(bf16));
        float mn = 1e30f, mx = -1e30f;
        int nan_count = 0;
        for (bf16 x : logits_host) {
            const float f = static_cast<float>(x);
            mn = std::min(mn, f);
            mx = std::max(mx, f);
            if (std::isnan(f)) ++nan_count;
        }
        AILA_LOG_INFO("[DG-DEBUG] raw logits min=%.4f max=%.4f nan=%d", mn, mx,
                      nan_count);
    }
    {
        const int64_t n =
            static_cast<int64_t>(canvas_len) * tc.vocab_size;
        bf16* src = static_cast<bf16*>(buf_.logits_bf16.data());
        float* dst = static_cast<float*>(buf_.logits_f32.data());
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(n)),
            [=](sycl::id<1> idx) {
                dst[idx[0]] = static_cast<float>(src[idx[0]]);
            });
    }
    ops::logit_softcap_inplace(
        ctx, buf_.logits_f32, tc.final_logit_softcapping,
        canvas_len * tc.vocab_size);
    if (profile) {
        ctx.synchronize();
        AILA_LOG_INFO(
            "[DG Profile] attention=%.2f ms dense_mlp=%.2f ms "
            "router=%.2f ms moe=%.2f ms post=%.2f ms",
            profile_totals_.attention_ms, profile_totals_.dense_mlp_ms,
            profile_totals_.router_ms, profile_totals_.moe_ms,
            profile_totals_.post_ms);
        AILA_LOG_INFO(
            "[DG Profile Detail] qkv_proj=%.2f ms attention_core=%.2f ms "
            "o_proj=%.2f ms dense_gate_up=%.2f ms "
            "dense_activation=%.2f ms dense_down=%.2f ms",
            profile_totals_.qkv_proj_ms,
            profile_totals_.attention_core_ms,
            profile_totals_.o_proj_ms,
            profile_totals_.dense_gate_up_ms,
            profile_totals_.dense_activation_ms,
            profile_totals_.dense_down_ms);
    }
    return buf_.logits_f32;
}

Tensor& DiffusionGemmaBnb4Backend::diffusion_compute_soft_embeds(
    Context& ctx, Tensor& logits_f32, int canvas_len) {
    const auto& tc = cfg_.text;
    const int hidden = tc.hidden_size;
    const int vocab = tc.vocab_size;
    ensure_buffers(ctx, canvas_len, current_len_ + canvas_len);
    ensure_logits_buffers(ctx);

    const float* logits_ptr = static_cast<const float*>(logits_f32.data());
    bf16* probs_ptr = static_cast<bf16*>(buf_.probs_bf16.data());
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(canvas_len)),
        [=](sycl::id<1> idx) {
            const int row = static_cast<int>(idx[0]);
            const float* src = logits_ptr + static_cast<size_t>(row) * vocab;
            bf16* dst = probs_ptr + static_cast<size_t>(row) * vocab;
            float max_logit = -1.0e30f;
            for (int c = 0; c < vocab; ++c) {
                const float value = static_cast<float>(bf16(src[c]));
                max_logit = std::max(max_logit, value);
            }
            float sum = 0.0f;
            for (int c = 0; c < vocab; ++c) {
                const float value = static_cast<float>(bf16(src[c]));
                sum += std::exp(value - max_logit);
            }
            const float inv = 1.0f / sum;
            for (int c = 0; c < vocab; ++c) {
                const float value = static_cast<float>(bf16(src[c]));
                dst[c] = bf16(std::exp(value - max_logit) * inv);
            }
        });

    buf_.soft_embed.forward(ctx, buf_.probs_bf16, buf_.soft_embeds,
                            canvas_len);
    ops::scale_inplace(ctx, buf_.soft_embeds, embed_scale_,
                       canvas_len * hidden);
    const std::string dump_dir =
        aila::env::read_string("AILA_DG_DUMP_DIR", "");
    if (aila::env::read_flag("AILA_DG_DUMP_DECODER", false) &&
        !dump_dir.empty()) {
        ctx.synchronize();
        std::vector<float> logits_row(static_cast<size_t>(vocab));
        std::vector<bf16> probs_row(static_cast<size_t>(vocab));
        std::vector<bf16> soft_row(static_cast<size_t>(hidden));
        ctx.memcpy_d2h(logits_row.data(), logits_f32.data(),
                       logits_row.size() * sizeof(float));
        ctx.memcpy_d2h(probs_row.data(), buf_.probs_bf16.data(),
                       probs_row.size() * sizeof(bf16));
        ctx.memcpy_d2h(soft_row.data(), buf_.soft_embeds.data(),
                       soft_row.size() * sizeof(bf16));
        const auto write_bytes = [&](const char* name, const void* data,
                                     size_t bytes) {
            std::ofstream out(dump_dir + "/" + name, std::ios::binary);
            out.write(static_cast<const char*>(data),
                      static_cast<std::streamsize>(bytes));
        };
        write_bytes("soft_logits_row0.bin", logits_row.data(),
                    logits_row.size() * sizeof(float));
        write_bytes("soft_probs_row0.bin", probs_row.data(),
                    probs_row.size() * sizeof(bf16));
        write_bytes("soft_embed_row0.bin", soft_row.data(),
                    soft_row.size() * sizeof(bf16));
    }
    return buf_.soft_embeds;
}

void DiffusionGemmaBnb4Backend::ensure_buffers(Context& ctx, int q_len,
                                               int kv_len) {
    const auto& tc = cfg_.text;
    const int hidden = tc.hidden_size;
    const int canvas_len = cfg_.canvas_length;
    if (buf_.hidden.valid() && q_len <= buf_q_cap_ && kv_len <= buf_kv_cap_) {
        return;
    }
    // Pending kernels may still reference buffers that are about to be freed.
    ctx.synchronize();
    if (q_len > buf_q_cap_) buf_q_cap_ = q_len;
    if (kv_len > buf_kv_cap_) buf_kv_cap_ = kv_len;
    const int max_q = buf_q_cap_;
    const int max_kv = buf_kv_cap_;

    buf_.hidden = Tensor::allocate(ctx, {max_q, hidden},
                                   dnnl::memory::data_type::bf16);
    buf_.normed = Tensor::allocate(ctx, {max_q, hidden},
                                   dnnl::memory::data_type::bf16);
    buf_.attn_o = Tensor::allocate(ctx, {max_q, hidden},
                                   dnnl::memory::data_type::bf16);

    buf_.mlp_gate = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.mlp_up = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.mlp_proj = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.mlp_out = Tensor::allocate(ctx, {max_q, hidden},
                                    dnnl::memory::data_type::bf16);
    buf_.ff1 = Tensor::allocate(ctx, {max_q, hidden},
                                dnnl::memory::data_type::bf16);
    buf_.ff2 = Tensor::allocate(ctx, {max_q, hidden},
                                dnnl::memory::data_type::bf16);
    buf_.moe_device_out = Tensor::allocate(ctx, {max_q, hidden},
                                           dnnl::memory::data_type::bf16);
    buf_.moe_gpu_ref = Tensor::allocate(ctx, {max_q, hidden},
                                        dnnl::memory::data_type::bf16);
    // Shared batched MoE transfer buffer: capacity for every routed row
    // (tokens x top_k) of one decoder pass, reused across layers.
    buf_.moe_transfer = Tensor::allocate(
        ctx, {static_cast<int64_t>(max_q) * tc.top_k_experts, hidden},
        dnnl::memory::data_type::bf16);
    buf_.moe_token_ids = Tensor::allocate(
        ctx, {static_cast<int64_t>(max_q) * tc.top_k_experts},
        dnnl::memory::data_type::s32);
    buf_.router_scores = Tensor::allocate(
        ctx, {max_q, tc.num_experts}, dnnl::memory::data_type::bf16);

    buf_.sc_normed = Tensor::allocate(ctx, {max_q, hidden},
                                      dnnl::memory::data_type::bf16);
    buf_.sc_gate = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.sc_up = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.sc_proj = Tensor::allocate(
        ctx, {max_q, tc.intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.sc_down = Tensor::allocate(ctx, {max_q, hidden},
                                    dnnl::memory::data_type::bf16);
    buf_.ones = Tensor::allocate(ctx, {tc.global_head_dim},
                                 dnnl::memory::data_type::bf16);
    std::vector<sycl::ext::oneapi::bfloat16> ones(
        static_cast<size_t>(tc.global_head_dim),
        sycl::ext::oneapi::bfloat16(1.0f));
    ctx.memcpy_h2d(buf_.ones.data(), ones.data(),
                   ones.size() * sizeof(sycl::ext::oneapi::bfloat16));

    AILA_LOG_DEBUG(
        "[DiffusionGemma] Buffers allocated: current=%.1f MB peak=%.1f MB",
        ctx.current_allocated_bytes() / (1024.0 * 1024.0),
        ctx.peak_allocated_bytes() / (1024.0 * 1024.0));
}

void DiffusionGemmaBnb4Backend::ensure_logits_buffers(Context& ctx) {
    if (buf_.logits_f32.valid()) return;
    ctx.synchronize();
    const auto& tc = cfg_.text;
    const int canvas_len = cfg_.canvas_length;
    buf_.logits_bf16 = Tensor::allocate(
        ctx, {canvas_len, tc.vocab_size}, dnnl::memory::data_type::bf16);
    buf_.logits_f32 = Tensor::allocate(
        ctx, {canvas_len, tc.vocab_size}, dnnl::memory::data_type::f32);
    buf_.soft_embeds = Tensor::allocate(ctx, {canvas_len, tc.hidden_size},
                                        dnnl::memory::data_type::bf16);
    buf_.probs_bf16 = Tensor::allocate(
        ctx, {canvas_len, tc.vocab_size}, dnnl::memory::data_type::bf16);
    buf_.lm_head.init(ctx, *embed_weight_, tc.hidden_size, tc.vocab_size);
    // Embedding is already [vocab, hidden], exactly the [in, out] layout for
    // probs @ embedding. Do not build the 1.5 GB decode-GEMV transpose because
    // this path is canvas-only (M=256).
    buf_.soft_embed.init(ctx, *embed_weight_, tc.vocab_size, tc.hidden_size,
                         true, false);
    AILA_LOG_DEBUG(
        "[DiffusionGemma] Logits buffers allocated: current=%.1f MB",
        ctx.current_allocated_bytes() / (1024.0 * 1024.0));
}

void DiffusionGemmaBnb4Backend::fill_attention_mask(
    Context& ctx, Tensor& mask, int q_len, int kv_len, int cached_len,
    bool bidirectional, bool sliding, int window) {
    float* mask_ptr = static_cast<float*>(mask.data());
    ctx.queue().parallel_for(
        sycl::range<2>(q_len, kv_len),
        [=](sycl::id<2> idx) {
            const int q = static_cast<int>(idx[0]);
            const int t = static_cast<int>(idx[1]);
            const int absolute_q = cached_len + q;
            bool allowed = true;
            if (!bidirectional && t > absolute_q) {
                allowed = false;
            }
            // The diffusion decoder is a DiT block: HF deliberately gives
            // both layer patterns a fully bidirectional mask over the encoder
            // prefix and current canvas. Sliding-window masking only applies
            // while constructing the causal encoder cache.
            if (!bidirectional && sliding && t < absolute_q - window + 1) {
                allowed = false;
            }
            mask_ptr[static_cast<size_t>(q) * kv_len + t] =
                allowed ? 0.0f : -1.0e30f;
        });
}

void DiffusionGemmaBnb4Backend::concat_kv(
    Context& ctx, Tensor& dst, const Tensor& cache, const Tensor& new_kv,
    int seq_len, int start_pos, int kv_heads, int head_dim,
    int max_seq_len) {
    bf16* dst_ptr = static_cast<bf16*>(dst.data());
    const bf16* cache_ptr = static_cast<const bf16*>(cache.data());
    const bf16* new_ptr = static_cast<const bf16*>(new_kv.data());
    const int total = start_pos + seq_len;
    const int kv_dim = kv_heads * head_dim;
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(total) * kv_dim),
        [=](sycl::id<1> idx) {
            const int d = static_cast<int>(idx[0]) % head_dim;
            const int h = (static_cast<int>(idx[0]) / head_dim) % kv_heads;
            const int t = static_cast<int>(idx[0]) / (head_dim * kv_heads);
            const float value =
                t < start_pos
                    ? static_cast<float>(
                          cache_ptr[static_cast<size_t>(h) * max_seq_len *
                                    head_dim +
                                    static_cast<size_t>(t) * head_dim + d])
                    : static_cast<float>(
                          new_ptr[static_cast<size_t>(t - start_pos) *
                                      (kv_heads * head_dim) +
                                  static_cast<size_t>(h) * head_dim + d]);
            dst_ptr[idx[0]] = bf16(value);
        });
}

void DiffusionGemmaBnb4Backend::run_layer_forward(
    Context& ctx, DiffusionGemmaLayerWeights& layer, LayerRuntime& runtime,
    Tensor& hidden, int seq_len, int start_pos, bool bidirectional) {
    const auto& tc = cfg_.text;
    const int hidden_size = tc.hidden_size;
    const int layer_idx =
        static_cast<int>(&layer - layers_.data());
    const bool profile = bidirectional &&
                         aila::env::read_flag("AILA_PROFILE_DG", false);
    const bool trace_moe =
        aila::env::read_flag("AILA_DG_MOE_TRACE", false);
    const auto profile_start = std::chrono::steady_clock::now();
    auto profile_mark = profile_start;
    const auto record_phase = [&](double& target_ms) {
        if (!profile) return;
        ctx.synchronize();
        const auto now = std::chrono::steady_clock::now();
        target_ms +=
            std::chrono::duration<double, std::milli>(now - profile_mark)
                .count();
        profile_mark = now;
    };
    const auto trace_point = [&](const char* label) {
        if (trace_moe) {
            std::fprintf(stderr, "[MoE trace] layer=%d %s\n", layer_idx,
                         label);
        }
    };
    trace_point("begin");
    AILA_LOG_DEBUG(
        "[DiffusionGemma] Layer %d forward (seq=%d start=%d bidi=%d) "
        "mem=%.1f MB",
        layer_idx, seq_len, start_pos, bidirectional ? 1 : 0,
        ctx.current_allocated_bytes() / (1024.0 * 1024.0));
    const bool sliding = tc.is_sliding_layer(layer_idx);
    const int head_dim = tc.head_dim_for_layer(layer_idx);
    const int num_heads = tc.num_attention_heads;
    const int num_kv_heads = tc.kv_heads_for_layer(layer_idx);
    const int rotary_dim = sliding
                               ? head_dim
                               : static_cast<int>(head_dim *
                                                  tc.rope_full.partial_rotary_factor);
    const float theta = sliding ? tc.rope_sliding.rope_theta
                                : tc.rope_full.rope_theta;
    const int q_dim = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int kv_len = start_pos + seq_len;

    // Exact-size per-layer scratch. The NF4 GEMM and masked attention assume
    // contiguous row-major layouts; padded shared buffers would misalign rows.
    Tensor q = Tensor::allocate(ctx, {seq_len, q_dim},
                                dnnl::memory::data_type::bf16);
    Tensor k = Tensor::allocate(ctx, {seq_len, kv_dim},
                                dnnl::memory::data_type::bf16);
    Tensor v = Tensor::allocate(ctx, {seq_len, kv_dim},
                                dnnl::memory::data_type::bf16);
    Tensor attn_out = Tensor::allocate(ctx, {seq_len, q_dim},
                                       dnnl::memory::data_type::bf16);
    Tensor k_full = Tensor::allocate(ctx, {kv_len, kv_dim},
                                     dnnl::memory::data_type::bf16);
    Tensor v_full = Tensor::allocate(ctx, {kv_len, kv_dim},
                                     dnnl::memory::data_type::bf16);
    Tensor mask = Tensor::allocate(ctx, {seq_len, kv_len},
                                   dnnl::memory::data_type::f32);
    Tensor scores = Tensor::allocate(
        ctx, {num_heads, seq_len, kv_len}, dnnl::memory::data_type::f32);
    const std::string dump_dir = aila::env::read_string("AILA_DG_DUMP_DIR", "");
    const int dump_layer_idx =
        aila::env::read_int_raw("AILA_DG_DUMP_LAYER", 0);
    const bool dump_bidirectional =
        aila::env::read_flag("AILA_DG_DUMP_DECODER", false);
    const bool dump_layer0 = layer_idx == dump_layer_idx &&
                             bidirectional == dump_bidirectional &&
                             !dump_dir.empty();
    const auto dump_tensor = [&](const char* tag, Tensor& t) {
        if (!dump_layer0) return;
        ctx.synchronize();
        std::vector<uint8_t> bytes(static_cast<size_t>(t.numel()) *
                                   t.element_size());
        ctx.memcpy_d2h(bytes.data(), t.data(), bytes.size());
        std::ofstream out(dump_dir + "/" + tag + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    };
    if (dump_layer0) {
        std::ofstream meta(dump_dir + "/layer0.txt");
        meta << "layer=" << layer_idx << " seq=" << seq_len
             << " hidden=" << hidden_size
             << " head_dim=" << head_dim << " heads=" << num_heads
             << " kv_heads=" << num_kv_heads << " sliding=" << sliding
             << " start=" << start_pos
             << " bidirectional=" << (bidirectional ? 1 : 0) << "\n";
        dump_tensor("hidden_in", hidden);
    }

    ops::rms_norm(ctx, hidden, *layer.input_ln, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    if (dump_layer0) dump_tensor("normed", buf_.normed);
    if (dump_layer0) {
        std::vector<uint16_t> nr3(4);
        ctx.memcpy_d2h(nr3.data(),
                       static_cast<const uint8_t*>(buf_.normed.data()) +
                           3 * hidden_size * 2,
                       8);
        AILA_LOG_INFO("[DG-DEBUG] backend normed r3=%04X %04X %04X %04X",
                      nr3[0], nr3[1], nr3[2], nr3[3]);
    }
    layer.q_proj.forward(ctx, buf_.qkv_scratch, buf_.normed, q, seq_len);
    if (dump_layer0) dump_tensor("q_proj", q);
    layer.k_proj.forward(ctx, buf_.qkv_scratch, buf_.normed, k, seq_len);
    if (dump_layer0) dump_tensor("k_proj", k);
    if (sliding) {
        layer.v_proj.forward(ctx, buf_.qkv_scratch, buf_.normed, v, seq_len);
    } else {
        // k_eq_v: V = K (before k_norm).
        ctx.queue().memcpy(v.data(), k.data(),
                           static_cast<size_t>(seq_len) * kv_dim *
                               sizeof(bf16));
    }
    record_phase(profile_totals_.qkv_proj_ms);
    trace_point("qkv done");

    ops::head_rms_norm(ctx, q, *layer.q_norm, tc.rms_norm_eps, seq_len,
                       num_heads, head_dim);
    ops::head_rms_norm(ctx, k, *layer.k_norm, tc.rms_norm_eps, seq_len,
                       num_kv_heads, head_dim);
    ops::head_rms_norm(ctx, v, buf_.ones, tc.rms_norm_eps, seq_len,
                       num_kv_heads, head_dim);
    if (dump_layer0) {
        dump_tensor("q_norm", q);
        dump_tensor("k_norm", k);
    }
    ops::apply_rope_partial(ctx, q, k, seq_len, start_pos, num_heads,
                            num_kv_heads, head_dim, rotary_dim, theta,
                            false, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0,
                            !sliding);
    if (dump_layer0) {
        dump_tensor("q_rope", q);
        dump_tensor("k_rope", k);
        dump_tensor("v_norm", v);
    }

    if (!bidirectional) {
        ops::copy_to_cache(ctx, k, runtime.k_cache, seq_len, start_pos,
                           num_kv_heads, head_dim, max_seq_len_);
        ops::copy_to_cache(ctx, v, runtime.v_cache, seq_len, start_pos,
                           num_kv_heads, head_dim, max_seq_len_);
    }

    concat_kv(ctx, k_full, runtime.k_cache, k, seq_len, start_pos,
              num_kv_heads, head_dim, max_seq_len_);
    concat_kv(ctx, v_full, runtime.v_cache, v, seq_len, start_pos,
              num_kv_heads, head_dim, max_seq_len_);
    fill_attention_mask(ctx, mask, seq_len, kv_len, start_pos,
                        bidirectional, sliding, tc.sliding_window);
    if (dump_layer0) {
        dump_tensor("k_full", k_full);
        dump_tensor("v_full", v_full);
        dump_tensor("attention_mask", mask);
    }
    ops::attention_prefill_masked(ctx, q, k_full, v_full, mask, attn_out,
                                  scores, seq_len, kv_len, num_heads,
                                  num_kv_heads, head_dim);
    record_phase(profile_totals_.attention_core_ms);
    trace_point("attn core done");
    if (dump_layer0) dump_tensor("attention_scores", scores);
    if (dump_layer0) dump_tensor("attn_out", attn_out);
    layer.o_proj.forward(ctx, buf_.o_scratch, attn_out, buf_.attn_o, seq_len);
    if (dump_layer0) dump_tensor("attn_o", buf_.attn_o);
    ops::rms_norm(ctx, buf_.attn_o, *layer.post_attn_ln, tc.rms_norm_eps,
                  buf_.attn_o, seq_len, hidden_size);
    if (dump_layer0) dump_tensor("post_attn_norm", buf_.attn_o);
    ops::residual_add(ctx, hidden, buf_.attn_o, seq_len * hidden_size);
    if (dump_layer0) dump_tensor("hidden_after_attn", hidden);
    record_phase(profile_totals_.o_proj_ms);
    trace_point("o done");
    if (profile) {
        profile_totals_.attention_ms = profile_totals_.qkv_proj_ms +
                                       profile_totals_.attention_core_ms +
                                       profile_totals_.o_proj_ms;
    }

    // Dense MLP path.
    ops::rms_norm(ctx, hidden, *layer.pre_ff_ln, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    if (dump_layer0) dump_tensor("pre_ff_norm", buf_.normed);
    layer.mlp_gate.forward(ctx, buf_.mlp_scratch, buf_.normed, buf_.mlp_gate,
                           seq_len);
    if (dump_layer0) dump_tensor("mlp_gate", buf_.mlp_gate);
    layer.mlp_up.forward(ctx, buf_.mlp_scratch, buf_.normed, buf_.mlp_up,
                         seq_len);
    if (dump_layer0) dump_tensor("mlp_up", buf_.mlp_up);
    record_phase(profile_totals_.dense_gate_up_ms);
    trace_point("dense gu done");
    ops::gelu_tanh_gate_mul(ctx, buf_.mlp_gate, buf_.mlp_up, buf_.mlp_proj,
                            seq_len * tc.intermediate_size);
    if (dump_layer0) dump_tensor("mlp_proj", buf_.mlp_proj);
    record_phase(profile_totals_.dense_activation_ms);
    trace_point("dense act done");
    layer.mlp_down.forward(ctx, buf_.mlp_scratch, buf_.mlp_proj, buf_.mlp_out,
                           seq_len);
    if (dump_layer0) dump_tensor("mlp_out", buf_.mlp_out);
    ops::rms_norm(ctx, buf_.mlp_out, *layer.post_ff_ln_1, tc.rms_norm_eps,
                  buf_.ff1, seq_len, hidden_size);
    if (dump_layer0) dump_tensor("ff1", buf_.ff1);
    record_phase(profile_totals_.dense_down_ms);
    trace_point("dense down done");
    if (profile) {
        profile_totals_.dense_mlp_ms = profile_totals_.dense_gate_up_ms +
                                       profile_totals_.dense_activation_ms +
                                       profile_totals_.dense_down_ms;
    }

    // MoE path (parallel to MLP, routed from the pre-MLP residual).
    // Keep the router projection on the trusted bf16 oneDNN path.  A scalar
    // host dot product uses a different accumulation order and changed about
    // 10-15% of near-tied top-k expert assignments in real layer traces.
    ops::rms_norm_unweighted(ctx, hidden, tc.rms_norm_eps, buf_.normed,
                             seq_len, hidden_size);
    bf16* routed_ptr = static_cast<bf16*>(buf_.normed.data());
    const bf16* router_scale_ptr =
        static_cast<const bf16*>(layer.router_scale->data());
    const bf16 root_size =
        bf16(1.0f / std::sqrt(static_cast<float>(hidden_size)));
    ctx.queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(seq_len) * hidden_size),
        [=](sycl::id<1> idx) {
            const size_t flat = idx[0];
            const int column = static_cast<int>(flat % hidden_size);
            const bf16 scaled = bf16(static_cast<float>(routed_ptr[flat]) *
                                     static_cast<float>(
                                         router_scale_ptr[column]));
            routed_ptr[flat] =
                bf16(static_cast<float>(scaled) * static_cast<float>(root_size));
        });
    layer.router_linear.forward(ctx, buf_.normed, buf_.router_scores, seq_len);
    std::vector<bf16> router_score_bf16(
        static_cast<size_t>(seq_len) * tc.num_experts);
    ctx.memcpy_d2h(router_score_bf16.data(), buf_.router_scores.data(),
                   router_score_bf16.size() * sizeof(bf16));
    std::vector<bf16> per_expert_scale_bf16(static_cast<size_t>(tc.num_experts));
    ctx.memcpy_d2h(per_expert_scale_bf16.data(), layer.per_expert_scale->data(),
                   per_expert_scale_bf16.size() * sizeof(bf16));
    std::vector<int> expert_ids;
    std::vector<float> expert_weights;
    router_topk_from_bf16_scores(
        router_score_bf16, per_expert_scale_bf16, seq_len, tc.num_experts,
        tc.top_k_experts, expert_ids, expert_weights);
    record_phase(profile_totals_.router_ms);
    trace_point("router done");
    if (dump_layer0) {
        std::ofstream ids_out(dump_dir + "/router_ids.bin", std::ios::binary);
        ids_out.write(reinterpret_cast<const char*>(expert_ids.data()),
                      static_cast<std::streamsize>(expert_ids.size() * sizeof(int)));
        std::ofstream weights_out(dump_dir + "/router_weights.bin", std::ios::binary);
        weights_out.write(reinterpret_cast<const char*>(expert_weights.data()),
                          static_cast<std::streamsize>(expert_weights.size() * sizeof(float)));
    }

    ops::rms_norm(ctx, hidden, *layer.pre_ff_ln_2, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    if (dump_layer0) dump_tensor("pre_ff2_norm", buf_.normed);
    static const bool moe_gpu =
        aila::env::read_flag("AILA_DG_MOE_GPU", false);
    static const bool moe_batched =
        aila::env::read_flag("AILA_DG_MOE_BATCHED", true);
    static const bool moe_harness =
        aila::env::read_flag("AILA_DG_MOE_HARNESS", false);
    auto* moe = moes_[static_cast<size_t>(layer_idx)].get();
    const auto compare_moe_outputs =
        [&](const char* label, Tensor& candidate) {
            std::vector<uint8_t> host_bytes(
                static_cast<size_t>(seq_len) * hidden_size * sizeof(bf16));
            std::vector<uint8_t> candidate_bytes(host_bytes.size());
            ctx.memcpy_d2h(host_bytes.data(), buf_.moe_device_out.data(),
                           host_bytes.size());
            ctx.memcpy_d2h(candidate_bytes.data(), candidate.data(),
                           candidate_bytes.size());
            ctx.synchronize();
            size_t mismatches = 0;
            size_t first = host_bytes.size();
            for (size_t i = 0; i < host_bytes.size(); i += 2) {
                if (host_bytes[i] != candidate_bytes[i] ||
                    host_bytes[i + 1] != candidate_bytes[i + 1]) {
                    ++mismatches;
                    first = std::min(first, i);
                }
            }
            static long long harness_total_mismatches = 0;
            static long long harness_total_elements = 0;
            harness_total_mismatches += static_cast<long long>(mismatches);
            harness_total_elements +=
                static_cast<long long>(host_bytes.size() / 2);
            if (mismatches != 0) {
                const size_t element = first / 2;
                uint16_t host_raw = 0;
                uint16_t candidate_raw = 0;
                std::memcpy(&host_raw, &host_bytes[first], 2);
                std::memcpy(&candidate_raw, &candidate_bytes[first], 2);
                std::fprintf(stderr,
                    "[MoE harness] %s layer=%d seq=%d MISMATCH=%zu first=%zu "
                    "(m=%d h=%d) host=%u cand=%u total=%lld/%lld\n",
                    label, layer_idx, seq_len, mismatches, element,
                    static_cast<int>(element / hidden_size),
                    static_cast<int>(element % hidden_size), host_raw,
                    candidate_raw, harness_total_mismatches,
                    harness_total_elements);
            }
        };
    if (moe_harness) {
        trace_point("harness begin");
        // Reference: committed host-driven path.
        std::vector<bf16> moe_input(static_cast<size_t>(seq_len) * hidden_size);
        ctx.memcpy_d2h(moe_input.data(), buf_.normed.data(),
                       moe_input.size() * sizeof(bf16));
        std::vector<bf16> moe_output;
        moe->forward(ctx, moe_input, expert_ids, expert_weights, moe_output,
                     seq_len, tc.top_k_experts);
        record_phase(profile_totals_.moe_ms);
        ctx.memcpy_h2d(buf_.moe_device_out.data(), moe_output.data(),
                       moe_output.size() * sizeof(bf16));
        // Candidates run after the host reference on the same in-order queue.
        if (moe_batched) {
            std::vector<bf16> batched_output;
            moe->forward_batched(ctx, buf_.normed, buf_.moe_transfer,
                                 buf_.moe_token_ids, expert_ids,
                                 expert_weights, batched_output, seq_len,
                                 tc.top_k_experts);
            ctx.memcpy_h2d(buf_.moe_gpu_ref.data(), batched_output.data(),
                           batched_output.size() * sizeof(bf16));
            compare_moe_outputs("batched", buf_.moe_gpu_ref);
        }
        if (moe_gpu) {
            // Candidate: GPU-resident path (WIP).
            moe->forward_gpu(ctx, buf_.normed, buf_.moe_gpu_ref, expert_ids,
                             expert_weights, seq_len, tc.top_k_experts);
            compare_moe_outputs("gpu", buf_.moe_gpu_ref);
        }
    } else if (moe_batched) {
        trace_point("batched begin");
        std::vector<bf16> moe_output;
        moe->forward_batched(ctx, buf_.normed, buf_.moe_transfer,
                             buf_.moe_token_ids, expert_ids, expert_weights,
                             moe_output, seq_len, tc.top_k_experts);
        record_phase(profile_totals_.moe_ms);
        ctx.memcpy_h2d(buf_.moe_device_out.data(), moe_output.data(),
                       moe_output.size() * sizeof(bf16));
        trace_point("batched done");
    } else if (moe_gpu) {
        moe->forward_gpu(ctx, buf_.normed, buf_.moe_device_out, expert_ids,
                         expert_weights, seq_len, tc.top_k_experts);
        record_phase(profile_totals_.moe_ms);
    } else {
        std::vector<bf16> moe_input(static_cast<size_t>(seq_len) * hidden_size);
        ctx.memcpy_d2h(moe_input.data(), buf_.normed.data(),
                       moe_input.size() * sizeof(bf16));
        std::vector<bf16> moe_output;
        moe->forward(ctx, moe_input, expert_ids, expert_weights, moe_output,
                     seq_len, tc.top_k_experts);
        record_phase(profile_totals_.moe_ms);
        ctx.memcpy_h2d(buf_.moe_device_out.data(), moe_output.data(),
                       moe_output.size() * sizeof(bf16));
    }
    trace_point("post moe");
    if (dump_layer0) dump_tensor("moe_raw", buf_.moe_device_out);
    ops::rms_norm(ctx, buf_.moe_device_out, *layer.post_ff_ln_2, tc.rms_norm_eps,
                  buf_.ff2, seq_len, hidden_size);
    if (dump_layer0) dump_tensor("ff2", buf_.ff2);

    ops::residual_add(ctx, buf_.ff1, buf_.ff2, seq_len * hidden_size);
    if (dump_layer0) dump_tensor("ff_sum", buf_.ff1);
    ops::rms_norm(ctx, buf_.ff1, *layer.post_ff_ln, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    if (dump_layer0) dump_tensor("post_ff_norm", buf_.normed);
    ops::residual_add(ctx, hidden, buf_.normed, seq_len * hidden_size);
    ops::scale_inplace(ctx, hidden, layer.layer_scalar,
                       seq_len * hidden_size);
    if (dump_layer0) dump_tensor("hidden_out", hidden);
    // Local scratch tensors are about to go out of scope; make sure every
    // asynchronous kernel touching them has completed before they are freed.
    ctx.synchronize();
    if (profile) {
        const auto now = std::chrono::steady_clock::now();
        profile_totals_.post_ms +=
            std::chrono::duration<double, std::milli>(now - profile_mark)
                .count();
    }
}

DiffusionGemmaBnb4Backend::~DiffusionGemmaBnb4Backend() = default;

