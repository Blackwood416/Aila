#include "DiffusionGemmaBnb4Backend.hpp"

#include "../utils/SafeTensors.hpp"
#include "core/Context.hpp"
#include "profile/Profiling.hpp"

#include <stdexcept>

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

} // namespace

DiffusionGemmaBnb4Backend::~DiffusionGemmaBnb4Backend() = default;

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
    max_seq_len_ = max_seq_len > 0 ? max_seq_len : 4096;
    if (max_seq_len_ > cfg_.text.max_position_embeddings) {
        max_seq_len_ = cfg_.text.max_position_embeddings;
    }

    const auto& tc = cfg_.text;
    if (static_cast<int>(tc.layer_types.size()) != tc.num_hidden_layers) {
        if (error_message) {
            *error_message =
                "DiffusionGemmaBnb4Backend: layer_types does not match num_hidden_layers";
        }
        return false;
    }

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
    if (!load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.gate_proj.weight",
                         sc_gate_, tc.intermediate_size, error_message) ||
        !load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.up_proj.weight",
                         sc_up_, tc.intermediate_size, error_message) ||
        !load_weight_ref(ctx, weights,
                         "model.decoder.self_conditioning.down_proj.weight",
                         sc_down_, tc.intermediate_size, error_message)) {
        return false;
    }
    sc_pre_norm_ = &weights.get("model.decoder.self_conditioning.pre_norm.weight");

    layers_.clear();
    layers_.reserve(static_cast<size_t>(tc.num_hidden_layers));
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
            static_cast<int64_t>(tc.num_experts) * tc.moe_intermediate_size;
        const int64_t expected_mlp = tc.intermediate_size;

        if (!load_weight_ref(ctx, weights, attn + "q_proj.weight", layer.q_proj,
                             tc.hidden_size, error_message) ||
            !load_weight_ref(ctx, weights, attn + "k_proj.weight", layer.k_proj,
                             tc.hidden_size, error_message) ||
            (sliding && !load_weight_ref(ctx, weights, attn + "v_proj.weight",
                                         layer.v_proj, tc.hidden_size,
                                         error_message)) ||
            !load_weight_ref(ctx, weights, attn + "o_proj.weight", layer.o_proj,
                             static_cast<int>(expected_q), error_message) ||
            !load_weight_ref(ctx, weights, mlp + "gate_proj.weight",
                             layer.mlp_gate_up, tc.hidden_size, error_message) ||
            !load_weight_ref(ctx, weights, mlp + "down_proj.weight",
                             layer.mlp_down, tc.intermediate_size, error_message) ||
            !load_weight_ref(ctx, weights, experts + "gate_up_proj",
                             layer.moe_gate_up, tc.hidden_size, error_message) ||
            !load_weight_ref(ctx, weights, experts + "down_proj",
                             layer.moe_down, tc.hidden_size, error_message)) {
            return false;
        }

        if (layer.q_proj.logical_out_features() != expected_q ||
            layer.k_proj.logical_out_features() != expected_kv ||
            (sliding && layer.v_proj.logical_out_features() != expected_kv) ||
            layer.o_proj.logical_out_features() != expected_q ||
            layer.mlp_gate_up.logical_out_features() != expected_mlp ||
            layer.mlp_down.logical_out_features() != tc.hidden_size ||
            layer.moe_gate_up.logical_out_features() != expected_moe_gate_up ||
            layer.moe_down.logical_out_features() != expected_moe_down) {
            if (error_message) {
                *error_message =
                    "DiffusionGemmaBnb4Backend: layer " +
                    std::to_string(layer_idx) + " weight shape mismatch";
            }
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

        if (layer.q_norm->numel() != head_dim ||
            layer.k_norm->numel() != head_dim ||
            layer.router_proj->numel() !=
                static_cast<int64_t>(tc.num_experts) * tc.hidden_size ||
            layer.router_scale->numel() != tc.hidden_size ||
            layer.per_expert_scale->numel() != tc.num_experts) {
            if (error_message) {
                *error_message =
                    "DiffusionGemmaBnb4Backend: layer " +
                    std::to_string(layer_idx) + " auxiliary tensor shape mismatch";
            }
            return false;
        }

        layers_.push_back(std::move(layer));
    }

    current_len_ = 0;
    AILA_LOG_INFO(
        "[DiffusionGemma] Loaded %d layers (hidden=%d heads=%d kv=%d window=%d "
        "experts=%d top_k=%d canvas=%d)",
        tc.num_hidden_layers, tc.hidden_size, tc.num_attention_heads,
        tc.num_key_value_heads, tc.sliding_window, tc.num_experts,
        tc.top_k_experts, cfg_.canvas_length);
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
    return false;  // full reset required until KV state is implemented
}
