#include "DiffusionGemmaBnb4Backend.hpp"

#include "../utils/SafeTensors.hpp"
#include "../utils/EnvUtils.hpp"
#include "core/Context.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <cmath>
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

    embed_scale_ = std::sqrt(static_cast<float>(tc.hidden_size));

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

    ops::embedding_lookup(ctx, *embed_weight_, canvas_ids_device, canvas_len,
                          buf_.hidden, tc.hidden_size);
    {
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
    {
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
        ops::rms_norm(ctx, buf_.sc_normed, *sc_pre_norm_, tc.rms_norm_eps,
                      buf_.sc_normed, canvas_len, hidden);
        {
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
        sc_up_.forward(ctx, buf_.sc_scratch, buf_.sc_normed, buf_.sc_up,
                       canvas_len);
        ops::gelu_tanh_gate_mul(ctx, buf_.sc_gate, buf_.sc_up, buf_.sc_proj,
                                canvas_len * tc.intermediate_size);
        sc_down_.forward(ctx, buf_.sc_scratch, buf_.sc_proj, buf_.sc_down,
                         canvas_len);
        {
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
        {
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
        {
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
        {
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
        {
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
    {
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
    {
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
    {
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

    {
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
    {
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
    {
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
                max_logit = std::max(max_logit, src[c]);
            }
            float sum = 0.0f;
            for (int c = 0; c < vocab; ++c) {
                sum += std::exp(src[c] - max_logit);
            }
            const float inv = 1.0f / sum;
            for (int c = 0; c < vocab; ++c) {
                dst[c] = bf16(std::exp(src[c] - max_logit) * inv);
            }
        });

    static std::unordered_map<int, Linear> soft_embed_prims;
    auto& lin = soft_embed_prims[canvas_len];
    if (!lin.weight()) {
        lin.init(ctx, *embed_weight_, vocab, hidden);
    }
    lin.forward(ctx, buf_.probs_bf16, buf_.soft_embeds, canvas_len);
    ops::scale_inplace(ctx, buf_.soft_embeds, embed_scale_,
                       canvas_len * hidden);
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

    AILA_LOG_INFO(
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
    AILA_LOG_INFO(
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
            if (sliding && t < absolute_q - window + 1) {
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
    AILA_LOG_INFO(
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
    const bool dump_layer0 = layer_idx == 0 && !bidirectional && !dump_dir.empty();
    const auto dump_tensor = [&](const char* tag, Tensor& t) {
        if (!dump_layer0) return;
        ctx.synchronize();
        std::vector<uint8_t> bytes(static_cast<size_t>(t.numel()) * 2);
        ctx.memcpy_d2h(bytes.data(), t.data(), bytes.size());
        std::ofstream out(dump_dir + "/" + tag + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
    };
    if (dump_layer0) {
        std::ofstream meta(dump_dir + "/layer0.txt");
        meta << "seq=" << seq_len << " hidden=" << hidden_size
             << " head_dim=" << head_dim << " heads=" << num_heads
             << " kv_heads=" << num_kv_heads << " sliding=" << sliding
             << " start=" << start_pos << "\n";
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

    ops::head_rms_norm(ctx, q, *layer.q_norm, tc.rms_norm_eps, seq_len,
                       num_heads, head_dim);
    ops::head_rms_norm(ctx, k, *layer.k_norm, tc.rms_norm_eps, seq_len,
                       num_kv_heads, head_dim);
    ops::head_rms_norm(ctx, v, buf_.ones, tc.rms_norm_eps, seq_len,
                       num_kv_heads, head_dim);
    ops::apply_rope_partial(ctx, q, k, seq_len, start_pos, num_heads,
                            num_kv_heads, head_dim, rotary_dim, theta);
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
    ops::attention_prefill_masked(ctx, q, k_full, v_full, mask, attn_out,
                                  scores, seq_len, kv_len, num_heads,
                                  num_kv_heads, head_dim);
    if (dump_layer0) dump_tensor("attn_out", attn_out);
    layer.o_proj.forward(ctx, buf_.o_scratch, attn_out, buf_.attn_o, seq_len);
    if (dump_layer0) dump_tensor("attn_o", buf_.attn_o);
    ops::rms_norm(ctx, buf_.attn_o, *layer.post_attn_ln, tc.rms_norm_eps,
                  buf_.attn_o, seq_len, hidden_size);
    ops::residual_add(ctx, hidden, buf_.attn_o, seq_len * hidden_size);

    // Dense MLP path.
    ops::rms_norm(ctx, hidden, *layer.pre_ff_ln, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    layer.mlp_gate.forward(ctx, buf_.mlp_scratch, buf_.normed, buf_.mlp_gate,
                           seq_len);
    layer.mlp_up.forward(ctx, buf_.mlp_scratch, buf_.normed, buf_.mlp_up,
                         seq_len);
    ops::gelu_tanh_gate_mul(ctx, buf_.mlp_gate, buf_.mlp_up, buf_.mlp_proj,
                            seq_len * tc.intermediate_size);
    layer.mlp_down.forward(ctx, buf_.mlp_scratch, buf_.mlp_proj, buf_.mlp_out,
                           seq_len);
    ops::rms_norm(ctx, buf_.mlp_out, *layer.post_ff_ln_1, tc.rms_norm_eps,
                  buf_.ff1, seq_len, hidden_size);

    // MoE path (parallel to MLP, routed from the pre-MLP residual).
    std::vector<bf16> hidden_host(static_cast<size_t>(seq_len) * hidden_size);
    ctx.memcpy_d2h(hidden_host.data(), hidden.data(),
                   hidden_host.size() * sizeof(bf16));
    std::vector<float> router_hidden(static_cast<size_t>(seq_len) * hidden_size);
    for (size_t i = 0; i < hidden_host.size(); ++i) {
        router_hidden[i] = static_cast<float>(hidden_host[i]);
    }
    std::vector<bf16> router_scale_bf16(static_cast<size_t>(hidden_size));
    ctx.memcpy_d2h(router_scale_bf16.data(), layer.router_scale->data(),
                   router_scale_bf16.size() * sizeof(bf16));
    std::vector<bf16> router_proj_bf16(
        static_cast<size_t>(tc.num_experts) * hidden_size);
    ctx.memcpy_d2h(router_proj_bf16.data(), layer.router_proj->data(),
                   router_proj_bf16.size() * sizeof(bf16));
    std::vector<bf16> per_expert_scale_bf16(static_cast<size_t>(tc.num_experts));
    ctx.memcpy_d2h(per_expert_scale_bf16.data(), layer.per_expert_scale->data(),
                   per_expert_scale_bf16.size() * sizeof(bf16));
    std::vector<float> router_scale(static_cast<size_t>(hidden_size));
    std::vector<float> router_proj(
        static_cast<size_t>(tc.num_experts) * hidden_size);
    std::vector<float> per_expert_scale(static_cast<size_t>(tc.num_experts));
    for (size_t i = 0; i < router_scale.size(); ++i) {
        router_scale[i] = static_cast<float>(router_scale_bf16[i]);
    }
    for (size_t i = 0; i < router_proj.size(); ++i) {
        router_proj[i] = static_cast<float>(router_proj_bf16[i]);
    }
    for (size_t i = 0; i < per_expert_scale.size(); ++i) {
        per_expert_scale[i] = static_cast<float>(per_expert_scale_bf16[i]);
    }
    std::vector<int> expert_ids;
    std::vector<float> expert_weights;
    ops::gemma4_router_host(
        router_hidden.data(), router_scale.data(), router_proj.data(),
        per_expert_scale.data(), seq_len, hidden_size, tc.num_experts,
        tc.top_k_experts, expert_ids, expert_weights);

    ops::rms_norm(ctx, hidden, *layer.pre_ff_ln_2, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    std::vector<bf16> moe_input(static_cast<size_t>(seq_len) * hidden_size);
    ctx.memcpy_d2h(moe_input.data(), buf_.normed.data(),
                   moe_input.size() * sizeof(bf16));
    std::vector<bf16> moe_output;
    if (aila::env::read_flag("AILA_DG_SKIP_MOE", false)) {
        moe_output.assign(moe_input.size(), bf16(0.0f));
    } else {
        moes_[static_cast<size_t>(layer_idx)]->forward(
            ctx, moe_input, expert_ids, expert_weights, moe_output, seq_len,
            tc.top_k_experts);
    }
    ctx.memcpy_h2d(buf_.moe_device_out.data(), moe_output.data(),
                   moe_output.size() * sizeof(bf16));
    ops::rms_norm(ctx, buf_.moe_device_out, *layer.post_ff_ln_2, tc.rms_norm_eps,
                  buf_.ff2, seq_len, hidden_size);

    ops::residual_add(ctx, buf_.ff1, buf_.ff2, seq_len * hidden_size);
    ops::rms_norm(ctx, buf_.ff1, *layer.post_ff_ln, tc.rms_norm_eps, buf_.normed,
                  seq_len, hidden_size);
    ops::residual_add(ctx, hidden, buf_.normed, seq_len * hidden_size);
    // Local scratch tensors are about to go out of scope; make sure every
    // asynchronous kernel touching them has completed before they are freed.
    ctx.synchronize();
}

DiffusionGemmaBnb4Backend::~DiffusionGemmaBnb4Backend() = default;

