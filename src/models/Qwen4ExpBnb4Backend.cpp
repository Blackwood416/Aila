#include "Qwen4ExpBnb4Backend.hpp"
#include "../utils/EnvUtils.hpp"
#include "../profile/Profiling.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

template <typename T, typename Compare>
void stable_sort_with_scratch(T* arr, size_t n, T* scratch, Compare comp) {
    if (n <= 1) return;
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t i = 0; i < n; i += 2 * width) {
            size_t mid = std::min(i + width, n);
            size_t end = std::min(i + 2 * width, n);
            size_t p = i, q = mid, k = i;
            while (p < mid && q < end) {
                if (comp(arr[q], arr[p])) {
                    scratch[k++] = arr[q++];
                } else {
                    scratch[k++] = arr[p++];
                }
            }
            while (p < mid) scratch[k++] = arr[p++];
            while (q < end) scratch[k++] = arr[q++];
        }
        std::copy(scratch, scratch + n, arr);
    }
}

void apply_silu_div4(Context& ctx, Tensor& buf, size_t count) {
    if ((count % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        vec8* ptr8 = reinterpret_cast<vec8*>(buf.data());
        size_t count8 = count / 8;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count8), [=](sycl::id<1> idx) {
                vec8 v_in = ptr8[idx];
                vec8 v_out;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float v = static_cast<float>(v_in[k]) * 0.25f;
                    float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    v_out[k] = bf16(v * s);
                }
                ptr8[idx] = v_out;
            });
        });
        return;
    }
    bf16* ptr = static_cast<bf16*>(buf.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(ptr[idx]) * 0.25f;
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            ptr[idx] = bf16(v * s);
        });
    });
}

void apply_sigmoid(Context& ctx, Tensor& buf, size_t count) {
    if ((count % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        vec8* ptr8 = reinterpret_cast<vec8*>(buf.data());
        size_t count8 = count / 8;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count8), [=](sycl::id<1> idx) {
                vec8 v_in = ptr8[idx];
                vec8 v_out;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float v = static_cast<float>(v_in[k]);
                    float s = 1.0f / (1.0f + sycl::native::exp(-v));
                    v_out[k] = bf16(s);
                }
                ptr8[idx] = v_out;
            });
        });
        return;
    }
    bf16* ptr = static_cast<bf16*>(buf.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(ptr[idx]);
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            ptr[idx] = bf16(s);
        });
    });
}

void apply_two_sigmoid_div4(Context& ctx, Tensor& in_buf, Tensor& out_f32, size_t count) {
    const bf16* in_ptr = static_cast<const bf16*>(in_buf.data());
    float* out_ptr = static_cast<float*>(out_f32.data());
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
            float v = static_cast<float>(in_ptr[idx]) * 0.25f;
            float s = 1.0f / (1.0f + sycl::native::exp(-v));
            out_ptr[idx] = 2.0f * s;
        });
    });
}

void add_shared_expert_to_y(Context& ctx, Tensor& y, Tensor& shared_out, Tensor& shared_gate, int rows, int hidden_size) {
    bf16* y_ptr = static_cast<bf16*>(y.data());
    const bf16* s_ptr = static_cast<const bf16*>(shared_out.data());
    const bf16* g_ptr = static_cast<const bf16*>(shared_gate.data());

    if ((hidden_size % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = hidden_size / 8;
        constexpr int wg = 64;
        const size_t num_wgs_per_row = (dim8 + wg - 1) / wg;

        if (rows == 1) {
            ctx.queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_wgs_per_row * wg), sycl::range<1>(wg)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                        const size_t d8 = item.get_global_id(0);
                        if (d8 < static_cast<size_t>(dim8)) {
                            const float g_val = static_cast<float>(g_ptr[0]);
                            const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
                            const size_t elem_offset = d8 * 8;
                            const vec8 s_vec = *reinterpret_cast<const vec8*>(s_ptr + elem_offset);
                            vec8 y_vec = *reinterpret_cast<const vec8*>(y_ptr + elem_offset);
#pragma unroll
                            for (int v = 0; v < 8; ++v) {
                                float s_val = static_cast<float>(s_vec[v]);
                                float y_val = static_cast<float>(y_vec[v]);
                                y_vec[v] = bf16(y_val + gate_act * s_val);
                            }
                            *reinterpret_cast<vec8*>(y_ptr + elem_offset) = y_vec;
                        }
                    });
            });
            return;
        }

        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(rows), num_wgs_per_row * wg),
                                  sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t r = item.get_group(0);
                    const size_t d8 = item.get_global_id(1);
                    if (d8 < static_cast<size_t>(dim8)) {
                        const float g_val = static_cast<float>(g_ptr[r]);
                        const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
                        const size_t elem_offset = r * hidden_size + d8 * 8;
                        const vec8 s_vec = *reinterpret_cast<const vec8*>(s_ptr + elem_offset);
                        vec8 y_vec = *reinterpret_cast<const vec8*>(y_ptr + elem_offset);
#pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            float s_val = static_cast<float>(s_vec[v]);
                            float y_val = static_cast<float>(y_vec[v]);
                            y_vec[v] = bf16(y_val + gate_act * s_val);
                        }
                        *reinterpret_cast<vec8*>(y_ptr + elem_offset) = y_vec;
                    }
                });
        });
        return;
    }

    const size_t total = static_cast<size_t>(rows) * hidden_size;
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
            size_t r = idx[0] / hidden_size;
            float g_val = static_cast<float>(g_ptr[r]);
            float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
            float s_val = static_cast<float>(s_ptr[idx[0]]);
            float y_val = static_cast<float>(y_ptr[idx[0]]);
            y_ptr[idx[0]] = bf16(y_val + gate_act * s_val);
        });
    });
}

void dump_hidden_row(Context& ctx, const std::string& path, const Tensor& hidden4, int row_idx, int hidden_dim = 10240) {
    std::vector<bf16> host_buf(hidden_dim);
    const bf16* src = static_cast<const bf16*>(hidden4.data()) + static_cast<size_t>(row_idx) * hidden_dim;
    ctx.memcpy_d2h(host_buf.data(), src, hidden_dim * sizeof(bf16));
    std::ofstream os(path, std::ios::binary);
    if (os) {
        os.write(reinterpret_cast<const char*>(host_buf.data()), hidden_dim * sizeof(bf16));
    }
}

} // namespace

// ============================================================================
// Session
// ============================================================================

Qwen4ExpBnb4Backend::Session::Session(
    Context& ctx, int capacity,
    std::shared_ptr<const SessionOwner> owner,
    bool boundedSnapshots, std::size_t maxSnapshots)
    : context_(&ctx), owner_(std::move(owner)),
      capacity_(capacity), boundedSnapshots_(boundedSnapshots),
      maxSnapshots_(maxSnapshots) {
    layerCaches_.resize(48);
    for (int i = 0; i < 48; ++i) {
        bool is_linear = ((i + 1) % 4 != 0);
        auto& cache = layerCaches_[i];
        if (is_linear) {
            cache.linear_state = Tensor::allocate(ctx, {48, 128, 128}, dnnl::memory::data_type::f32);
            ctx.queue().memset(cache.linear_state.data(), 0, cache.linear_state.size_bytes());
            cache.host_linear_state.assign(48 * 128 * 128, 0.0f);

            cache.linear_conv_state = Tensor::allocate(ctx, {3, 10240}, dnnl::memory::data_type::f32);
            ctx.queue().memset(cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
            cache.host_linear_conv_state.assign(3 * 10240, 0.0f);
            cache.linear_conv_head = 0;
            cache.device_state_dirty = false;
        } else {
            cache.k = Tensor::allocate(ctx, {2, capacity, 256}, dnnl::memory::data_type::bf16);
            cache.v = Tensor::allocate(ctx, {2, capacity, 256}, dnnl::memory::data_type::bf16);
        }
    }
}

Qwen4ExpBnb4Backend::Session::~Session() {
    if (context_) {
        context_->synchronize();
    }
}

// ============================================================================
// Backend Construction & Lifecycle
// ============================================================================

Qwen4ExpBnb4Backend::Qwen4ExpBnb4Backend()
    : sessionOwner_(std::make_shared<SessionOwner>()) {}

Qwen4ExpBnb4Backend::~Qwen4ExpBnb4Backend() {
    free_pinned_host_buffers();
}

void Qwen4ExpBnb4Backend::free_pinned_host_buffers() {
    if (!context_) return;
    context_->synchronize();
    if (router_host_ids_) context_->free_host(router_host_ids_);
    if (router_host_weights_) context_->free_host(router_host_weights_);
    router_host_ids_ = nullptr;
    router_host_weights_ = nullptr;
    if (pred_host_ids_) context_->free_host(pred_host_ids_);
    if (pred_host_weights_) context_->free_host(pred_host_weights_);
    pred_host_ids_ = nullptr;
    pred_host_weights_ = nullptr;
    if (embed_host_ids_) context_->free_host(embed_host_ids_);
    if (embed_host_rows_) context_->free_host(embed_host_rows_);
    embed_host_ids_ = nullptr;
    embed_host_rows_ = nullptr;
    if (batch_slots_host_) context_->free_host(batch_slots_host_);
    if (batch_slots_dev_) context_->free_device(batch_slots_dev_);
    if (batch_weights_dev_) context_->free_device(batch_weights_dev_);
    batch_slots_host_ = nullptr;
    batch_slots_dev_ = nullptr;
    batch_weights_dev_ = nullptr;
}

// ============================================================================
// Load
// ============================================================================

bool Qwen4ExpBnb4Backend::load(
    Context& ctx, ModelWeights& weights, const ModelSpec& spec,
    int max_seq_len, std::string* error_message) {
    context_ = &ctx;
    spec_ = spec;
    cfg_ = spec.qwen4exp_text;
    hidden_size_ = cfg_.hidden_size;
    max_seq_len_ = std::min(max_seq_len, 2048);
    if (max_seq_len > 2048) {
        AILA_LOG_WARN("[Qwen4Exp] Clamping max_seq_len=%d to 2048 (M1 limit; QSA indexer in M2 will support full context)", max_seq_len);
    }

    full_q_heads_ = cfg_.num_attention_heads;
    full_kv_heads_ = cfg_.num_key_value_heads;
    full_head_dim_ = cfg_.head_dim;
    full_q_dim_ = full_q_heads_ * full_head_dim_;
    full_kv_dim_ = full_kv_heads_ * full_head_dim_;
    full_q_proj_dim_ = 2 * full_q_dim_;
    full_fused_qkv_dim_ = full_q_proj_dim_ + 2 * full_kv_dim_;

    linear_q_heads_ = cfg_.linear_num_key_heads;
    linear_kv_heads_ = cfg_.linear_num_value_heads;
    linear_head_dim_ = cfg_.linear_key_head_dim;
    linear_q_dim_ = linear_q_heads_ * linear_head_dim_;
    linear_kv_dim_ = linear_kv_heads_ * cfg_.linear_value_head_dim;
    linear_qkv_dim_ = 2 * linear_q_dim_ + linear_kv_dim_;
    linear_z_dim_ = linear_kv_dim_;
    linear_all_dim_ = linear_qkv_dim_ + linear_z_dim_ + 2 * linear_kv_heads_;

    auto upload_bnb = [&](const std::string& weight_name) {
        if (weights.has_host(weight_name)) weights.upload(weight_name, ctx);
        if (weights.has_host(weight_name + ".absmax")) weights.upload(weight_name + ".absmax", ctx);
        if (weights.has_host(weight_name + ".quant_map")) weights.upload(weight_name + ".quant_map", ctx);
        if (weights.has_host(weight_name + ".nested_absmax")) weights.upload(weight_name + ".nested_absmax", ctx);
        if (weights.has_host(weight_name + ".nested_quant_map")) weights.upload(weight_name + ".nested_quant_map", ctx);
        if (weights.has_host(weight_name + ".quant_state.bitsandbytes__nf4")) weights.upload(weight_name + ".quant_state.bitsandbytes__nf4", ctx);
    };

    auto erase_bnb = [&](const std::string& weight_name) {
        if (weights.has(weight_name)) weights.erase(weight_name);
        if (weights.has(weight_name + ".absmax")) weights.erase(weight_name + ".absmax");
        if (weights.has(weight_name + ".nested_absmax")) weights.erase(weight_name + ".nested_absmax");
        if (weights.has(weight_name + ".nested_quant_map")) weights.erase(weight_name + ".nested_quant_map");
        if (weights.has(weight_name + ".quant_state.bitsandbytes__nf4")) weights.erase(weight_name + ".quant_state.bitsandbytes__nf4");
    };

    auto init_quant_linear = [&](const std::string& name, Bnb4BitLinear& linear) -> bool {
        upload_bnb(name);
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

    auto ensure_f32 = [&](const std::string& name) -> Tensor* {
        if (!weights.has(name)) {
            weights.upload(name, ctx);
        }
        Tensor& src = weights.get(name);
        if (src.dtype() == dnnl::memory::data_type::f32) {
            return &src;
        }
        size_t n = static_cast<size_t>(src.numel());
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

    auto plus_one_norm = [&](const std::string& name) -> Tensor* {
        if (!weights.has(name)) {
            weights.upload(name, ctx);
        }
        Tensor& src = weights.get(name);
        size_t n = static_cast<size_t>(src.numel());
        std::vector<bf16> host(n);
        ctx.memcpy_d2h(host.data(), src.data(), n * sizeof(bf16));
        std::vector<float> host_f32(n);
        for (size_t i = 0; i < n; ++i) {
            host[i] = bf16(static_cast<float>(host[i]) + 1.0f);
        }
        Tensor dst = Tensor::allocate(ctx, src.shape(), dnnl::memory::data_type::bf16);
        ctx.memcpy_h2d(dst.data(), host.data(), n * sizeof(bf16));
        weights.replace(name, std::move(dst));
        return &weights.get(name);
    };

    AILA_LOG_INFO("[Qwen4Exp] Uploading model embed_tokens and lm_head...");
    const std::string embed_name = "model.language_model.embed_tokens.weight";
    static const bool s_host_embed = aila::env::read_flag("AILA_Q4E_HOST_EMBED", true);
    embed_host_ = s_host_embed ? weights.host_view(embed_name) : HostTensorView{};
    if (!(embed_host_.valid() && embed_host_.dtype == DT_BF16 && embed_host_.shape.size() == 2 &&
          embed_host_.shape[1] == hidden_size_)) {
        embed_host_ = {};
        embed_weight_ = &weights.upload(embed_name, ctx);
    }

    Tensor& raw_lm_head = weights.upload("lm_head.weight", ctx);
    lm_head_weight_transposed_ = Tensor::allocate(ctx, {hidden_size_, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
    ops::transpose(ctx, raw_lm_head, lm_head_weight_transposed_);
    ctx.synchronize();
    weights.erase("lm_head.weight");
    lm_head_.init(ctx, lm_head_weight_transposed_, hidden_size_, cfg_.vocab_size, /*preprocessed=*/true, /*prepare_gemv=*/false);

    // Hyper-Connection Mixer
    AILA_LOG_INFO("[Qwen4Exp] Loading hyper_connection_mixer...");
    auto load_hc = [&](const std::string& hc_prefix, Qwen4ExpHyperConnectionLayer& hc, bool with_inject) {
        weights.upload(hc_prefix + "hc_norm.weight", ctx);
        hc.hc_norm_weight = ensure_f32(hc_prefix + "hc_norm.weight");
        hc.mix_down_w = &weights.upload(hc_prefix + "input_mix_weight_down.weight", ctx);
        hc.input_mix_weight_down.init(ctx, *hc.mix_down_w, 10240, 320, false);
        hc.mix_up_w = &weights.upload(hc_prefix + "input_mix_weight_up.weight", ctx);
        hc.input_mix_weight_up.init(ctx, *hc.mix_up_w, 320, 10240, false);
        if (with_inject) {
            hc.inject_w = &weights.upload(hc_prefix + "block_inject_weight.weight", ctx);
            hc.block_inject_weight.init(ctx, *hc.inject_w, 10240, 4, false);
        }
    };
    load_hc("model.language_model.hyper_connection_mixer.", mixer_, false);

    // Layers
    AILA_LOG_INFO("[Qwen4Exp] Loading %d layers...", cfg_.num_hidden_layers);
    layers_.resize(cfg_.num_hidden_layers);
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
        auto& layer = layers_[i];
        std::string prefix = "model.language_model.layers." + std::to_string(i) + ".";

        bool is_linear = true;
        if (i < static_cast<int>(cfg_.layer_types.size())) {
            is_linear = (cfg_.layer_types[i] == "linear_attention");
        } else {
            is_linear = ((i + 1) % 4 != 0);
        }
        layer.is_linear = is_linear;

        load_hc(prefix + "attn_hyper_connection.", layer.attn_hc, true);
        load_hc(prefix + "mlp_hyper_connection.", layer.mlp_hc, true);

        // Attention
        if (is_linear) {
            layer.geom = ops::DeltaNetGeometry::create(
                linear_q_heads_, linear_kv_heads_, linear_head_dim_, cfg_.linear_value_head_dim,
                cfg_.linear_conv_kernel_dim, cfg_.rms_norm_eps, ops::GateActivation::Sigmoid);

            if (!init_quant_linear(prefix + "linear_attn.out_proj.weight", layer.linear_o_proj)) {
                return false;
            }

            Bnb4BitLinear linear_qkv_proj, linear_z_proj, linear_a_proj, linear_b_proj;
            if (!init_quant_linear(prefix + "linear_attn.in_proj_qkv.weight", linear_qkv_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_z.weight", linear_z_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_a.weight", linear_a_proj) ||
                !init_quant_linear(prefix + "linear_attn.in_proj_b.weight", linear_b_proj)) {
                return false;
            }
            std::string fused_error;
            if (!layer.linear_all_proj.init_fused_rows(ctx, linear_qkv_proj, linear_z_proj, linear_a_proj, linear_b_proj, &fused_error)) {
                if (error_message) *error_message = fused_error;
                return false;
            }
            erase_bnb(prefix + "linear_attn.in_proj_qkv.weight");
            erase_bnb(prefix + "linear_attn.in_proj_z.weight");
            erase_bnb(prefix + "linear_attn.in_proj_a.weight");
            erase_bnb(prefix + "linear_attn.in_proj_b.weight");

            weights.upload(prefix + "linear_attn.norm.weight", ctx);
            layer.linear_norm_weight = ensure_f32(prefix + "linear_attn.norm.weight");
            weights.upload(prefix + "linear_attn.A_log", ctx);
            layer.linear_A_log = ensure_f32(prefix + "linear_attn.A_log");
            layer.linear_dt_bias = &weights.upload(prefix + "linear_attn.dt_bias", ctx);
            layer.linear_conv1d_weight = &weights.upload(prefix + "linear_attn.conv1d.weight", ctx);
        } else {
            if (!init_quant_linear(prefix + "self_attn.o_proj.weight", layer.o_proj)) {
                return false;
            }
            Bnb4BitLinear q_proj, k_proj, v_proj;
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
            erase_bnb(prefix + "self_attn.q_proj.weight");
            erase_bnb(prefix + "self_attn.k_proj.weight");
            erase_bnb(prefix + "self_attn.v_proj.weight");

            weights.upload(prefix + "self_attn.q_norm.weight", ctx);
            layer.q_norm_weight = plus_one_norm(prefix + "self_attn.q_norm.weight");
            weights.upload(prefix + "self_attn.k_norm.weight", ctx);
            layer.k_norm_weight = plus_one_norm(prefix + "self_attn.k_norm.weight");
        }

        // MoE Router & Shared Expert
        layer.router_w = &weights.upload(prefix + "mlp.gate.weight", ctx);
        layer.router_gate.init(ctx, *layer.router_w, hidden_size_, cfg_.num_experts, false);

        layer.shared_expert_gate_w = &weights.upload(prefix + "mlp.shared_expert_gate.weight", ctx);
        layer.shared_expert_gate.init(ctx, *layer.shared_expert_gate_w, hidden_size_, 1, false);

        Bnb4BitLinear shared_gate_proj, shared_up_proj;
        if (!init_quant_linear(prefix + "mlp.shared_expert.gate_proj.weight", shared_gate_proj) ||
            !init_quant_linear(prefix + "mlp.shared_expert.up_proj.weight", shared_up_proj) ||
            !init_quant_linear(prefix + "mlp.shared_expert.down_proj.weight", layer.shared_down_proj)) {
            return false;
        }
        std::string fused_gate_up_error;
        if (!layer.shared_gate_up_proj.init_fused_rows(ctx, shared_gate_proj, shared_up_proj, &fused_gate_up_error)) {
            if (error_message) *error_message = fused_gate_up_error;
            return false;
        }
        erase_bnb(prefix + "mlp.shared_expert.gate_proj.weight");
        erase_bnb(prefix + "mlp.shared_expert.up_proj.weight");
    }

    // PLE (Layer 1)
    AILA_LOG_INFO("[Qwen4Exp] Initializing PLE (Layer 1)...");
    if (!ple_.init(ctx, weights, 1, max_seq_len_, cfg_.rms_norm_eps, error_message)) {
        return false;
    }

    // Expert Cache: AILA_Q4E_EXPERT_CACHE_MB, or by default the VRAM left after static weights
    // minus headroom for session state, prefill workspace and the desktop. On a 16 GB A770
    // driving a display, ~1 GB free after warmup already slowed every kernel ~2x (WDDM paging);
    // ~2 GB free did not.
    size_t expert_cache_bytes = 0;
    const int expert_cache_mb = aila::env::read_int("AILA_Q4E_EXPERT_CACHE_MB", 0);
    if (expert_cache_mb > 0) {
        expert_cache_bytes = static_cast<size_t>(expert_cache_mb) << 20;
    } else {
        size_t vram = ctx.queue().get_device().get_info<sycl::info::device::global_mem_size>();
        if (ctx.device_allocation_limit() > 0) vram = std::min(vram, ctx.device_allocation_limit());
        const size_t full_layers = static_cast<size_t>(
            std::count_if(layers_.begin(), layers_.end(), [](const Qwen4ExpLayer& l) { return !l.is_linear; }));
        const size_t kv_bytes = full_layers * 2 * cfg_.num_key_value_heads * cfg_.head_dim * sizeof(bf16) *
                                static_cast<size_t>(max_seq_len_);
        const size_t reserve =
            (static_cast<size_t>(std::max(0, aila::env::read_int("AILA_Q4E_VRAM_RESERVE_MB", 2560))) << 20) + kv_bytes;
        const size_t min_bytes = static_cast<size_t>(1024) << 20;
        const size_t used = ctx.current_allocated_bytes() + reserve;
        expert_cache_bytes = vram > used + min_bytes ? vram - used : min_bytes;
    }
    AILA_LOG_INFO("[Qwen4Exp] Initializing Expert Cache (%.0f MB)...", expert_cache_bytes / (1024.0 * 1024.0));
    if (!expert_cache_.init(ctx, weights, cfg_.num_hidden_layers, expert_cache_bytes,
                            cfg_.num_experts, hidden_size_, cfg_.moe_intermediate_size, 64, error_message)) {
        return false;
    }

    ensure_runtime_buffers(ctx, 1);
    // Batched prefill acquire needs every unique expert of a layer resident at once.
    if (expert_cache_.num_slots() >= cfg_.num_experts + cfg_.num_experts_per_tok) {
        prefill_expert_slot_.assign(cfg_.num_experts, -1);
        prefill_expert_uses_.assign(cfg_.num_experts, 0);
        prefill_unique_.reserve(cfg_.num_experts);
        prefill_unique_slots_.reserve(cfg_.num_experts);
    } else {
        AILA_LOG_WARN("[Qwen4Exp] Expert cache has %d slots (< %d): prefill acquires experts per token",
                      expert_cache_.num_slots(), cfg_.num_experts + cfg_.num_experts_per_tok);
    }
    prefill_pred_uses_.assign(cfg_.num_experts, 0);
    prefill_pred_unique_.reserve(cfg_.num_experts);
    sort_scratch_.resize(cfg_.num_experts);
    AILA_LOG_INFO("[Qwen4Exp] Backend successfully loaded (48 layers, max_seq_len=%d)", max_seq_len_);
    return true;
}

// ============================================================================
// Runtime Buffers
// ============================================================================

void Qwen4ExpBnb4Backend::ensure_runtime_buffers(Context& ctx, int seq_len) {
    if (seq_len <= runtime_seq_capacity_) return;
    int cap = std::max(seq_len, runtime_seq_capacity_ == 0 ? std::min(max_seq_len_, 64) : runtime_seq_capacity_ * 2);
    cap = std::min(cap, max_seq_len_);
    if (cap < seq_len) cap = seq_len;

    buf_.hidden4 = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_N = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_down = Tensor::allocate(ctx, {cap, 320}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_up = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_buf = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_f32 = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::f32);
    buf_.x = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);
    buf_.y = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    // Full Attention buffers
    buf_.full_qkv = Tensor::allocate(ctx, {cap, full_fused_qkv_dim_}, dnnl::memory::data_type::bf16);
    buf_.q = Tensor::allocate(ctx, {cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.z = Tensor::allocate(ctx, {cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.k = Tensor::allocate(ctx, {cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.v = Tensor::allocate(ctx, {cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.attn_out = Tensor::allocate(ctx, {cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.decode_scores = Tensor::allocate(ctx, {full_q_heads_, max_seq_len_}, dnnl::memory::data_type::f32);
    constexpr int kDecodeExactTile = 128;
    const int partial_stride = (full_head_dim_ == 256) ? 272 : 144;
    const int max_tiles = (max_seq_len_ + kDecodeExactTile - 1) / kDecodeExactTile;
    buf_.decode_attn_partials = Tensor::allocate(
        ctx,
        {(int64_t)full_q_heads_, (int64_t)max_tiles, (int64_t)partial_stride},
        dnnl::memory::data_type::f32);
    buf_.scores = Tensor::allocate(ctx, {full_q_heads_, cap, cap}, dnnl::memory::data_type::f32);
    buf_.incr_scores = Tensor::allocate(ctx, {full_q_heads_, cap, max_seq_len_}, dnnl::memory::data_type::f32);

    // Linear Attention buffers
    buf_.linear_all = Tensor::allocate(ctx, {cap, linear_all_dim_}, dnnl::memory::data_type::bf16);

    // MoE buffers
    buf_.router_scores = Tensor::allocate(ctx, {cap, 512}, dnnl::memory::data_type::bf16);
    pred_scores_ = Tensor::allocate(ctx, {cap, cfg_.num_experts}, dnnl::memory::data_type::bf16);
    free_pinned_host_buffers();
    router_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
    router_host_weights_ = static_cast<float*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    pred_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
    pred_host_weights_ = static_cast<float*>(ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    if (embed_host_.valid()) {
        embed_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * sizeof(int)));
        embed_host_rows_ = static_cast<uint8_t*>(ctx.alloc_host(static_cast<size_t>(cap) * hidden_size_ * sizeof(bf16)));
    }
    batch_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
    batch_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
    batch_weights_dev_ = static_cast<float*>(
        ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    buf_.moe_h = Tensor::allocate(ctx, {cap, cfg_.num_experts_per_tok, cfg_.moe_intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_buf = Tensor::allocate(ctx, {cap, 1}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_up_buf = Tensor::allocate(ctx, {cap, 1280}, dnnl::memory::data_type::bf16);
    buf_.shared_inter_buf = Tensor::allocate(ctx, {cap, 640}, dnnl::memory::data_type::bf16);
    buf_.shared_out_buf = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    if (!buf_.logits.valid()) {
        buf_.logits = Tensor::allocate(ctx, {1, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
    }

    runtime_seq_capacity_ = cap;
}

// ============================================================================
// Forward Implementation
// ============================================================================

void Qwen4ExpBnb4Backend::EmbedTokens(Context& ctx, const int* tokenIdsDevice, int count) {
    if (!embed_host_.valid()) {
        ops::embedding_lookup(ctx, *embed_weight_, tokenIdsDevice, count, buf_.x, hidden_size_);
        return;
    }
    // The blocking readback also drains the previous h2d, so the pinned rows can be reused.
    ctx.memcpy_d2h(embed_host_ids_, tokenIdsDevice, static_cast<size_t>(count) * sizeof(int));
    const size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(bf16);
    const int64_t rows = embed_host_.shape[0];
    for (int t = 0; t < count; ++t) {
        const int id = embed_host_ids_[t];
        if (id < 0 || id >= rows) {
            throw std::runtime_error("Qwen4ExpBnb4Backend: token id " + std::to_string(id) + " outside the embedding table");
        }
        std::memcpy(embed_host_rows_ + static_cast<size_t>(t) * row_bytes,
                    embed_host_.ptr + static_cast<size_t>(id) * row_bytes, row_bytes);
    }
    ctx.memcpy_h2d_async(buf_.x.data(), embed_host_rows_, static_cast<size_t>(count) * row_bytes);
}

void Qwen4ExpBnb4Backend::DenseLinear(
    Context& ctx, Linear& linear, Tensor* raw_weight,
    Tensor& input, Tensor& output, int count, int out_features, int in_features) {
    static const bool s_native_gemv = aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    if (count == 1 && s_native_gemv && raw_weight) {
        ops::bf16_gemv_bf16(ctx, *raw_weight, input.data_as<bf16>(), output.data_as<bf16>(),
                            out_features, in_features);
        return;
    }
    linear.forward(ctx, input, output, count);
}

void Qwen4ExpBnb4Backend::RunHyperConnection(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject) {
    const int hc_dim = 4 * hidden_size_;
    const int lowrank = cfg_.hc_lowrank;
    ops::grouped_rms_norm(ctx, static_cast<const bf16*>(buf_.hidden4.data()),
                          static_cast<const float*>(hc.hc_norm_weight->data()),
                          cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
                          count, 4, hidden_size_, true);

    static const bool s_fused_hc = aila::env::read_flag("AILA_Q4E_HC_FUSED", true);
    if (count == 1 && s_fused_hc && hc.mix_down_w && hc.mix_up_w && (!with_inject || hc.inject_w)) {
        ops::hc_decode_down_inject(ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                                   hc.mix_down_w->data_as<bf16>(),
                                   with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
                                   static_cast<bf16*>(buf_.hc_mix_down.data()),
                                   static_cast<float*>(buf_.hc_inject_f32.data()),
                                   lowrank, hidden_size_, 4);
        ops::hc_decode_up_mix(ctx, static_cast<const bf16*>(buf_.hc_mix_down.data()),
                              hc.mix_up_w->data_as<bf16>(),
                              static_cast<const bf16*>(buf_.hc_N.data()),
                              static_cast<bf16*>(buf_.x.data()),
                              lowrank, hidden_size_, 4);
        return;
    }

    DenseLinear(ctx, hc.input_mix_weight_down, hc.mix_down_w, buf_.hc_N, buf_.hc_mix_down,
                count, lowrank, hc_dim);
    apply_silu_div4(ctx, buf_.hc_mix_down, static_cast<size_t>(count) * lowrank);

    DenseLinear(ctx, hc.input_mix_weight_up, hc.mix_up_w, buf_.hc_mix_down, buf_.hc_mix_up,
                count, hc_dim, lowrank);
    apply_sigmoid(ctx, buf_.hc_mix_up, static_cast<size_t>(count) * hc_dim);

    ops::hc_mix_reduce(ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                       static_cast<const bf16*>(buf_.hc_mix_up.data()),
                       static_cast<bf16*>(buf_.x.data()),
                       count, hidden_size_, 4);

    if (with_inject) {
        DenseLinear(ctx, hc.block_inject_weight, hc.inject_w, buf_.hc_N, buf_.hc_inject_buf,
                    count, 4, hc_dim);
        apply_two_sigmoid_div4(ctx, buf_.hc_inject_buf, buf_.hc_inject_f32, static_cast<size_t>(count) * 4);
    }
}

Tensor& Qwen4ExpBnb4Backend::ForwardImpl(
    Context& ctx, Session& session,
    const int* tokenIdsDevice, int count, bool lastTokenOnly) {
    ValidateSession(session);
    if (count <= 0) return buf_.logits;

    int start_pos = session.currentLength_;
    int cached_len = start_pos + count;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: context length " +
                                 std::to_string(cached_len) + " exceeds max_seq_len " +
                                 std::to_string(max_seq_len_));
    }

    ensure_runtime_buffers(ctx, count);

    // 1. Embedding lookup + optional vision embedding override
    EmbedTokens(ctx, tokenIdsDevice, count);

    if (!session.embeddingOverridePositions_.empty() &&
        session.embeddingOverrideHiddenSize_ == hidden_size_) {
        bf16* hidden_ptr = static_cast<bf16*>(buf_.x.data());
        for (size_t i = 0; i < session.embeddingOverridePositions_.size(); ++i) {
            int pos = session.embeddingOverridePositions_[i];
            if (pos < start_pos || pos >= start_pos + count) continue;
            int local_row = pos - start_pos;
            const bf16* src = session.embeddingOverrideValues_.data() + i * static_cast<size_t>(hidden_size_);
            bf16* dst = hidden_ptr + static_cast<size_t>(local_row) * static_cast<size_t>(hidden_size_);
            ctx.queue().memcpy(dst, src, static_cast<size_t>(hidden_size_) * sizeof(bf16));
        }
    }

    // Repeat embedding 4x into hidden4: [count, 10240]
    const bf16* embed_ptr = static_cast<const bf16*>(buf_.x.data());
    bf16* hidden4_ptr = static_cast<bf16*>(buf_.hidden4.data());
    const int h_size = hidden_size_;

    if (count == 1 && (h_size % 8 == 0)) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = h_size / 8;
        constexpr int wg = 256;
        const size_t num_wgs = (dim8 + wg - 1) / wg;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_wgs * wg), sycl::range<1>(wg)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    const int i = static_cast<int>(item.get_global_id(0));
                    if (i < dim8) {
                        const vec8 val = *reinterpret_cast<const vec8*>(embed_ptr + i * 8);
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            *reinterpret_cast<vec8*>(hidden4_ptr + s * h_size + i * 8) = val;
                        }
                    }
                });
        });
    } else if (h_size % 8 == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = h_size / 8;
        constexpr int wg = 256;
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(count, wg), sycl::range<2>(1, wg)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    const size_t t = item.get_group(0);
                    const int lid = static_cast<int>(item.get_local_id(1));
                    const size_t t_embed_base = t * h_size;
                    const size_t t_hidden_base = t * 10240;
                    for (int i = lid; i < dim8; i += wg) {
                        const vec8 val = *reinterpret_cast<const vec8*>(embed_ptr + t_embed_base + i * 8);
#pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            *reinterpret_cast<vec8*>(hidden4_ptr + t_hidden_base + s * h_size + i * 8) = val;
                        }
                    }
                });
        });
    } else {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(count) * h_size), [=](sycl::id<1> idx) {
                size_t t = idx[0] / h_size;
                size_t d = idx[0] % h_size;
                bf16 val = embed_ptr[idx[0]];
                for (int s = 0; s < 4; ++s) {
                    hidden4_ptr[t * 10240 + s * h_size + d] = val;
                }
            });
        });
    }

    const int full_rotary_dim = 64;
    constexpr int kMaxTopK = 16; // moe_decode_* kernels capture at most 16 slots
    const int top_k = cfg_.num_experts_per_tok;
    if (top_k > kMaxTopK) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: num_experts_per_tok exceeds 16");
    }
    static const std::string s_dump_dir = aila::env::read_string("AILA_DEBUG_Q4E_DUMP_DIR", "");
    static const bool s_profile = aila::env::read_flag("AILA_PROFILE_Q4E_DECODE", false);
    using prof_clock = std::chrono::steady_clock;
    auto prof_ms = [](prof_clock::time_point a, prof_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto prof_t0 = prof_clock::now();
    double prof_router_sync_ms = 0.0;
    double prof_acquire_ms = 0.0;
    const ExpertCacheStats prof_stats0 = expert_cache_.stats();

    // AILA_PROFILE_Q4E_STAGES serializes after every stage; use for relative cost only.
    static const bool s_stages = aila::env::read_flag("AILA_PROFILE_Q4E_STAGES", false);
    enum { kStagePle, kStageHc, kStageLinearAttn, kStageFullAttn, kStageRouter,
           kStageExperts, kStageShared, kStageHead, kStageCount };
    double stage_ms[kStageCount] = {};
    auto stage_t = prof_clock::now();
    static const bool s_predict_stats = aila::env::read_flag("AILA_Q4E_PREDICT_STATS", false);
    // Decode-time speculative prefetch of layer L+1 experts (predicted from layer L's MoE input)
    // overlapping host fill + PCIe DMA with GPU compute of layer L+1's attention.
    static const bool s_prefetch = aila::env::read_flag("AILA_Q4E_PREFETCH", true);
    static const bool s_fuse_shared = aila::env::read_flag("AILA_Q4E_MOE_SHARED_FUSED", false);
    // Router top-k output is sorted by probability; only the most confident guesses are prefetched.
    static const int s_prefetch_topn = std::max(1, aila::env::read_int("AILA_Q4E_PREFETCH_TOPN", 10));
    static const int s_prefill_prefetch_max = aila::env::read_int("AILA_Q4E_PREFILL_PREFETCH_MAX", 48);
    // Note: pred_scores_, pred_host_ids_, and pred_host_weights_ are pre-allocated in ensure_runtime_buffers
    std::array<int, kMaxTopK> pred_ids{};
    bool have_pred = false;
    auto stage = [&](int s) {
        if (!s_stages) return;
        ctx.synchronize();
        const auto now = prof_clock::now();
        stage_ms[s] += prof_ms(stage_t, now);
        stage_t = now;
    };

    // 2. 48 Transformer Layers
    for (int L = 0; L < cfg_.num_hidden_layers; ++L) {
        auto& layer = layers_[L];
        auto& cache = session.layerCaches_[L];

        // 2.0. PLE on Layer 1 (Epilogue Residual Fused)
        if (L == 1) {
            const int* ids_ptr = embed_host_ids_;
            std::vector<int> fallback_host_ids;
            if (!embed_host_.valid()) {
                fallback_host_ids.resize(count);
                ctx.memcpy_d2h(fallback_host_ids.data(), tokenIdsDevice, count * sizeof(int));
                ids_ptr = fallback_host_ids.data();
            }
            ple_.forward(ctx, session.ple_state_, ids_ptr, count,
                         hidden4_ptr, nullptr, hidden4_ptr);
        }

        stage(kStagePle);

        // 2.1. Attn Hyper-Connection
        RunHyperConnection(ctx, layer.attn_hc, count, true);
        stage(kStageHc);

        // 2.2. Attention Computation
        if (layer.is_linear) {
            ops::DeltaNetWeights d_weights;
            d_weights.conv_weight = static_cast<const bf16*>(layer.linear_conv1d_weight->data());
            d_weights.norm_weight = static_cast<const float*>(layer.linear_norm_weight->data());
            d_weights.a_log = static_cast<const float*>(layer.linear_A_log->data());
            d_weights.dt_bias = static_cast<const bf16*>(layer.linear_dt_bias->data());

            ops::DeltaNetState d_state;
            d_state.state = static_cast<float*>(cache.linear_state.data());
            d_state.conv_state = static_cast<float*>(cache.linear_conv_state.data());
            d_state.conv_head = cache.linear_conv_head;
            d_state.device_state_dirty = cache.device_state_dirty;

            if (count == 1) {
                layer.linear_all_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.x, buf_.linear_all, 1);
                bf16* linear_all_ptr = static_cast<bf16*>(buf_.linear_all.data());
                const bf16* qkv_ptr = linear_all_ptr;
                const bf16* z_ptr = linear_all_ptr + linear_qkv_dim_;
                const bf16* a_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_;
                const bf16* b_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_ + linear_kv_heads_;

                ops::run_linear_delta_decode_gpu(ctx, layer.geom, d_weights, d_state,
                                                 qkv_ptr, z_ptr, a_ptr, b_ptr,
                                                 static_cast<bf16*>(buf_.attn_out.data()));

                layer.linear_o_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.attn_out, buf_.y, 1);
            } else {
                layer.linear_all_proj.forward(ctx, linear_scratch_, buf_.x, buf_.linear_all, count);
                ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, d_weights, d_state,
                                                         static_cast<const bf16*>(buf_.linear_all.data()),
                                                         static_cast<bf16*>(buf_.attn_out.data()), count);
                layer.linear_o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.y, count);
            }
            cache.linear_conv_head = d_state.conv_head;
            cache.device_state_dirty = d_state.device_state_dirty;
        } else {
            if (count == 1) {
                layer.qkv_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.x, buf_.full_qkv, 1);
                bf16* fused_ptr = static_cast<bf16*>(buf_.full_qkv.data());
                Tensor q_gate_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)full_q_proj_dim_});
                Tensor k_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_, {1, (int64_t)full_kv_dim_});
                Tensor v_view = Tensor::view(ctx, fused_ptr + full_q_proj_dim_ + full_kv_dim_, {1, (int64_t)full_kv_dim_});

                ops::decode_prepare_qgkv_packed_partial(
                    ctx, q_gate_view, k_view, v_view,
                    buf_.q, buf_.z,
                    *layer.q_norm_weight, *layer.k_norm_weight,
                    cache.k, cache.v, start_pos,
                    full_q_heads_, full_kv_heads_, full_head_dim_,
                    cfg_.rms_norm_eps, full_rotary_dim,
                    cfg_.rope.rope_theta,
                    cfg_.rope.mrope_interleaved,
                    (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr), (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr), (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                    session.mropePromptLength_, session.mropeTextPositionDelta_,
                    cfg_.rope.mrope_section[0],
                    cfg_.rope.mrope_section[1],
                    cfg_.rope.mrope_section[2]);

                ops::attention_decode(
                    ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.decode_scores,
                    full_q_heads_, full_kv_heads_, full_head_dim_, cached_len,
                    &buf_.decode_attn_partials, &buf_.z);

                layer.o_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.attn_out, buf_.y, 1);
            } else {
                layer.qkv_proj.forward(ctx, linear_scratch_, buf_.x, buf_.full_qkv, count);
                ops::split_qgkv(ctx, buf_.full_qkv, buf_.q, buf_.z, buf_.k, buf_.v, count,
                                full_q_heads_, full_kv_heads_, full_head_dim_);

                ops::head_rms_norm(ctx, buf_.q, *layer.q_norm_weight, cfg_.rms_norm_eps, count, full_q_heads_, full_head_dim_);
                ops::head_rms_norm(ctx, buf_.k, *layer.k_norm_weight, cfg_.rms_norm_eps, count, full_kv_heads_, full_head_dim_);

                ops::apply_rope_partial(ctx, buf_.q, buf_.k, count, start_pos,
                                        full_q_heads_, full_kv_heads_, full_head_dim_, full_rotary_dim,
                                        cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                                        (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr), (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr), (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                        session.mropePromptLength_, session.mropeTextPositionDelta_,
                                        cfg_.rope.mrope_section[0],
                                        cfg_.rope.mrope_section[1],
                                        cfg_.rope.mrope_section[2]);

                if (count > 1) {
                    ops::copy_to_cache(ctx, buf_.k, cache.k, count, start_pos, full_kv_heads_, full_head_dim_, session.Capacity());
                    ops::copy_to_cache(ctx, buf_.v, cache.v, count, start_pos, full_kv_heads_, full_head_dim_, session.Capacity());
                }

                if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores, count, full_q_heads_, full_kv_heads_, full_head_dim_, &buf_.z);
                } else {
                    ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores, count, start_pos, full_q_heads_, full_kv_heads_, full_head_dim_, session.Capacity(), &buf_.z);
                }

                layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.y, count);
            }
        }

        stage(layer.is_linear ? kStageLinearAttn : kStageFullAttn);

        // 2.3. Attn Residual Add
        ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                           static_cast<const bf16*>(buf_.y.data()),
                           static_cast<const float*>(buf_.hc_inject_f32.data()),
                           count, hidden_size_, 4);

        // 2.4. MLP Hyper-Connection
        RunHyperConnection(ctx, layer.mlp_hc, count, true);
        stage(kStageHc);

        // 2.5. MoE Computation
        DenseLinear(ctx, layer.router_gate, layer.router_w, buf_.x, buf_.router_scores,
                    count, cfg_.num_experts, hidden_size_);
        // Top-k writes straight into pinned host memory; the event wait is the only readback.
        sycl::event route_event = ops::router_softmax_topk_device(
            ctx, static_cast<const bf16*>(buf_.router_scores.data()), nullptr,
            router_host_ids_, router_host_weights_,
            count, cfg_.num_experts, top_k, true);
        const bool predict_next = (count == 1)
            ? ((s_prefetch || s_predict_stats) && (L + 1 < cfg_.num_hidden_layers))
            : (s_prefetch && (s_prefill_prefetch_max > 0) && (L + 1 < cfg_.num_hidden_layers));
        if (predict_next) {
            // Route this layer's MoE input through the next layer's router to guess its experts.
            DenseLinear(ctx, layers_[L + 1].router_gate, layers_[L + 1].router_w, buf_.x, pred_scores_,
                        count, cfg_.num_experts, hidden_size_);
            static const bool s_pred_ids_only = aila::env::read_flag("AILA_Q4E_PRED_IDS_ONLY", true);
            if (s_pred_ids_only) {
                route_event = ops::router_topk_ids_device(
                    ctx, static_cast<const bf16*>(pred_scores_.data()),
                    pred_host_ids_, count, cfg_.num_experts, top_k);
            } else {
                route_event = ops::router_softmax_topk_device(
                    ctx, static_cast<const bf16*>(pred_scores_.data()), nullptr,
                    pred_host_ids_, pred_host_weights_, count, cfg_.num_experts, top_k, true);
            }
        }

        // Shared expert only depends on x: submit it before waiting on route_event so the GPU runs it
        // concurrently while the host fills missed routed experts and sets up dispatch.
        DenseLinear(ctx, layer.shared_expert_gate, layer.shared_expert_gate_w, buf_.x, buf_.shared_gate_buf,
                    count, 1, hidden_size_);
        if (count == 1) {
            bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.shared_gate_up_proj, buf_.x, buf_.shared_inter_buf, cfg_.shared_expert_intermediate_size);
            if (!fused_ok) {
                layer.shared_gate_up_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.x, buf_.shared_gate_up_buf, 1);
                ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                               static_cast<bf16*>(buf_.shared_inter_buf.data()), 1, cfg_.shared_expert_intermediate_size);
            }
            layer.shared_down_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.shared_inter_buf, buf_.shared_out_buf, 1);
        } else {
            layer.shared_gate_up_proj.forward(ctx, linear_scratch_, buf_.x, buf_.shared_gate_up_buf, count);
            ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                           static_cast<bf16*>(buf_.shared_inter_buf.data()), count, cfg_.shared_expert_intermediate_size);
            layer.shared_down_proj.forward(ctx, linear_scratch_, buf_.shared_inter_buf, buf_.shared_out_buf, count);
        }

        const auto prof_router_t0 = prof_clock::now();
        route_event.wait();
        const auto prof_router_t1 = prof_clock::now();
        if (predict_next || have_pred) {
            if (have_pred && s_predict_stats && count == 1) {
                for (int k = 0; k < top_k; ++k) {
                    const int e = router_host_ids_[k];
                    const bool in_pred = std::find(pred_ids.begin(), pred_ids.begin() + top_k, e) != pred_ids.begin() + top_k;
                    pred_overlap_ += in_pred ? 1 : 0;
                    if (!expert_cache_.contains(L, e)) {
                        ++pred_miss_total_;
                        pred_miss_hit_ += in_pred ? 1 : 0;
                    }
                }
                pred_total_ += top_k;
            }
            have_pred = predict_next && (count == 1);
            if (predict_next && count == 1) std::copy(pred_host_ids_, pred_host_ids_ + top_k, pred_ids.begin());
        }
        prof_router_sync_ms += prof_ms(prof_router_t0, prof_router_t1);
        stage(kStageRouter);
        // Prefill: acquire the layer's unique experts in one batch (one streamed fill/DMA
        // instead of a blocking round trip per token), then map slots back per token.
        const auto prof_acq_batch_t0 = prof_clock::now();
        const bool batch_acquire = count > 1 && prefill_expert_slot_.size() == static_cast<size_t>(cfg_.num_experts);
        sycl::event batch_dma_event{};
        if (batch_acquire) {
            std::fill(prefill_expert_uses_.begin(), prefill_expert_uses_.end(), 0);
            prefill_unique_.clear();
            for (int i = 0; i < count * top_k; ++i) {
                const int e = router_host_ids_[i];
                if (prefill_expert_uses_[e]++ == 0) {
                    prefill_unique_.push_back(e);
                }
            }
            prefill_unique_slots_.resize(prefill_unique_.size());
            batch_dma_event = expert_cache_.acquire(L, prefill_unique_.data(), static_cast<int>(prefill_unique_.size()),
                                                    prefill_unique_slots_.data());
            for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                prefill_expert_slot_[prefill_unique_[u]] = prefill_unique_slots_[u];
            }
            // A long prompt touches several times the cache capacity, so plain LRU would leave
            // only the last layers resident for decode. Keep each layer's share of its most-used
            // experts hot and queue the rest for eviction, least used first.
            static const bool s_prefill_demote = aila::env::read_flag("AILA_Q4E_PREFILL_DEMOTE", true);
            const size_t keep = static_cast<size_t>(expert_cache_.num_slots() / cfg_.num_hidden_layers);
            if (s_prefill_demote && prefill_unique_.size() > keep) {
                stable_sort_with_scratch(
                    prefill_unique_.data(), prefill_unique_.size(), sort_scratch_.data(),
                    [&](int a, int b) { return prefill_expert_uses_[a] > prefill_expert_uses_[b]; });
                expert_cache_.demote(L, prefill_unique_.data() + keep,
                                     static_cast<int>(prefill_unique_.size() - keep));
            }
        }
        if (s_prefetch && predict_next && count > 1) {
            prefill_pred_unique_.clear();
            std::fill(prefill_pred_uses_.begin(), prefill_pred_uses_.end(), 0);
            const int total_preds = count * top_k;
            for (int i = 0; i < total_preds; ++i) {
                const int e = pred_host_ids_[i];
                if (e >= 0 && e < cfg_.num_experts && prefill_pred_uses_[e]++ == 0) {
                    prefill_pred_unique_.push_back(e);
                }
            }
            stable_sort_with_scratch(
                prefill_pred_unique_.data(), prefill_pred_unique_.size(), sort_scratch_.data(),
                [&](int a, int b) { return prefill_pred_uses_[a] > prefill_pred_uses_[b]; });
            const int prefetch_count = std::min(static_cast<int>(prefill_pred_unique_.size()), s_prefill_prefetch_max);
            expert_cache_.prefetch_async(L + 1, prefill_pred_unique_.data(), prefetch_count);
        }
        prof_acquire_ms += prof_ms(prof_acq_batch_t0, prof_clock::now());

        static const bool s_prefill_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);
        if (batch_acquire && s_prefill_moe_rows && count > 1) {
            for (int t = 0; t < count; ++t) {
                const int* t_ids = router_host_ids_ + t * top_k;
                ops::MoeExpertSlotDev* t_batch_slots = batch_slots_host_ + t * top_k;
                for (int k = 0; k < top_k; ++k) {
                    t_batch_slots[k] = expert_cache_.get_slot_dev(prefill_expert_slot_[t_ids[k]]);
                }
            }
            ctx.queue().memcpy(batch_slots_dev_, batch_slots_host_,
                               static_cast<size_t>(count) * top_k * sizeof(ops::MoeExpertSlotDev));
            ctx.queue().memcpy(batch_weights_dev_, router_host_weights_,
                               static_cast<size_t>(count) * top_k * sizeof(float));

            const bf16* x_ptr = static_cast<const bf16*>(buf_.x.data());
            bf16* h_ptr = static_cast<bf16*>(buf_.moe_h.data());
            bf16* y_ptr = static_cast<bf16*>(buf_.y.data());

            ops::moe_prefill_gate_up_rows(
                ctx, batch_slots_dev_, nullptr, x_ptr, h_ptr,
                count, top_k, hidden_size_, cfg_.moe_intermediate_size, 64,
                &batch_dma_event, expert_cache_.uses_nested_codes());

            ops::moe_prefill_down_rows(
                ctx, batch_slots_dev_, batch_weights_dev_, nullptr, h_ptr, y_ptr,
                count, top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) : nullptr,
                s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) : nullptr,
                nullptr, expert_cache_.uses_nested_codes());
        } else {
            for (int t = 0; t < count; ++t) {
                const int* t_ids = router_host_ids_ + t * top_k;
                const float* t_weights = router_host_weights_ + t * top_k;
                int slots[kMaxTopK];
                sycl::event dma_event{};
                if (batch_acquire) {
                    for (int k = 0; k < top_k; ++k) slots[k] = prefill_expert_slot_[t_ids[k]];
                    if (t == 0) dma_event = batch_dma_event;
                } else {
                    const auto prof_acq_t0 = prof_clock::now();
                    dma_event = expert_cache_.acquire(L, t_ids, top_k, slots);
                    prof_acquire_ms += prof_ms(prof_acq_t0, prof_clock::now());
                    if (s_prefetch && predict_next && count == 1) {
                        expert_cache_.prefetch_async(L + 1, pred_ids.data(), std::min(top_k, s_prefetch_topn));
                    }
                }

                ops::MoeExpertSlotDev dev_slots[kMaxTopK];
                expert_cache_.get_slots_dev(slots, top_k, dev_slots);

                const bf16* x_t = static_cast<const bf16*>(buf_.x.data()) + t * hidden_size_;
                bf16* y_t = static_cast<bf16*>(buf_.y.data()) + t * hidden_size_;

                ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, dev_slots, nullptr, x_t,
                    static_cast<bf16*>(buf_.moe_h.data()),
                    top_k, hidden_size_, cfg_.moe_intermediate_size, 64,
                    dma_event);

                ops::moe_decode_down_weighted_accumulate(
                    ctx, dev_slots, nullptr, t_weights,
                    static_cast<const bf16*>(buf_.moe_h.data()),
                    nullptr, y_t,
                    top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) +
                                        static_cast<size_t>(t) * hidden_size_ : nullptr,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) + t : nullptr);
            }
        }
        stage(kStageExperts);

        if (!s_fuse_shared) {
            add_shared_expert_to_y(ctx, buf_.y, buf_.shared_out_buf, buf_.shared_gate_buf,
                                   count, hidden_size_);
        }

        // 2.6. MLP Residual Add
        ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                           static_cast<const bf16*>(buf_.y.data()),
                           static_cast<const float*>(buf_.hc_inject_f32.data()),
                           count, hidden_size_, 4);
        stage(kStageShared);

        if (!s_dump_dir.empty()) {
            std::string dump_path = s_dump_dir + "/layer_" + std::to_string(L) + ".bin";
            dump_hidden_row(ctx, dump_path, buf_.hidden4, count - 1, 10240);
        }
    }

    // 3. Final Hyper-Connection Mixer
    RunHyperConnection(ctx, mixer_, count, false);

    if (!s_dump_dir.empty()) {
        std::string mixer_path = s_dump_dir + "/mixer.bin";
        dump_hidden_row(ctx, mixer_path, buf_.x, count - 1, 2560);
    }

    // 4. LM Head (Compute for last token)
    if (lastTokenOnly) {
        const bf16* last_x = static_cast<const bf16*>(buf_.x.data()) + static_cast<size_t>(count - 1) * hidden_size_;
        Tensor x_last_view = Tensor::view(ctx, const_cast<bf16*>(last_x), {1, hidden_size_});
        lm_head_.forward(ctx, x_last_view, buf_.logits, 1);
        lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {1, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
    } else {
        if (buf_.logits.shape(0) < count) {
            buf_.logits = Tensor::allocate(ctx, {count, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
        }
        lm_head_.forward(ctx, buf_.x, buf_.logits, count);
        lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {count, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
    }

    stage(kStageHead);

    if (!s_dump_dir.empty()) {
        std::string logits_path = s_dump_dir + "/logits.bin";
        dump_hidden_row(ctx, logits_path, buf_.logits, 0, cfg_.vocab_size);
    }

    expert_cache_.wait_prefetch();
    CaptureSnapshot(ctx, session, cached_len, count);
    session.currentLength_ = cached_len;

    if (s_profile) {
        ctx.synchronize();
        const ExpertCacheStats& s1 = expert_cache_.stats();
        const uint64_t hits = s1.hit_count - prof_stats0.hit_count;
        const uint64_t misses = s1.miss_count - prof_stats0.miss_count;
        AILA_LOG_INFO("[Q4E Profile] pos=%d count=%d total=%.2fms router_sync=%.2fms acquire=%.2fms "
                      "(fill=%.2fms dma_wait=%.2fms pf_wait=%.2fms pf=%llu/%llu) hit=%llu/%llu xfer=%.1fMB",
                      start_pos, count, prof_ms(prof_t0, prof_clock::now()),
                      prof_router_sync_ms, prof_acquire_ms,
                      (s1.fill_ns - prof_stats0.fill_ns) * 1e-6,
                      (s1.dma_wait_ns - prof_stats0.dma_wait_ns) * 1e-6,
                      (s1.prefetch_wait_ns - prof_stats0.prefetch_wait_ns) * 1e-6,
                      static_cast<unsigned long long>(s1.prefetch_used - prof_stats0.prefetch_used),
                      static_cast<unsigned long long>(s1.prefetch_issued - prof_stats0.prefetch_issued),
                      static_cast<unsigned long long>(hits),
                      static_cast<unsigned long long>(hits + misses),
                      (s1.total_transferred_bytes - prof_stats0.total_transferred_bytes) / (1024.0 * 1024.0));
        if (s_predict_stats && pred_total_ > 0) {
            AILA_LOG_INFO("[Q4E Predict] next-layer overlap=%.1f%% misses_predicted=%llu/%llu (%.1f%%)",
                          100.0 * pred_overlap_ / pred_total_,
                          static_cast<unsigned long long>(pred_miss_hit_),
                          static_cast<unsigned long long>(pred_miss_total_),
                          pred_miss_total_ ? 100.0 * pred_miss_hit_ / pred_miss_total_ : 0.0);
        }
        if (s_stages) {
            AILA_LOG_INFO("[Q4E Stages] ple=%.2f hc=%.2f linear_attn=%.2f full_attn=%.2f router=%.2f "
                          "experts=%.2f (acquire=%.2f) shared=%.2f head=%.2f ms",
                          stage_ms[kStagePle], stage_ms[kStageHc], stage_ms[kStageLinearAttn],
                          stage_ms[kStageFullAttn], stage_ms[kStageRouter], stage_ms[kStageExperts],
                          prof_acquire_ms, stage_ms[kStageShared], stage_ms[kStageHead]);
        }
    }
    return lastTokenLogits_;
}

// ============================================================================
// Snapshot & Rollback
// ============================================================================

void Qwen4ExpBnb4Backend::CaptureSnapshot(
    Context& ctx, Session& session, int nextLength, int tokenCount) {
    if (session.boundedSnapshots_ && session.maxSnapshots_ == 0) return;
    int step = aila::env::read_int("AILA_Q4E_PREFILL_STEP", 64);
    if (step <= 0) step = 64;
    if (tokenCount <= 1 && nextLength % step != 0) return;

    ctx.synchronize();
    auto& snapshots = session.snapshots_;
    if (session.boundedSnapshots_ && snapshots.find(nextLength) == snapshots.end() &&
        snapshots.size() >= session.maxSnapshots_) {
        const auto oldest = std::min_element(
            snapshots.begin(), snapshots.end(),
            [](const auto& left, const auto& right) { return left.first < right.first; });
        snapshots.erase(oldest);
    }
    auto& snapshot = snapshots[nextLength];
    try {
        snapshot.layers.resize(session.layerCaches_.size());
        sycl::event last_d2h_event{};
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) continue;
            auto& cache = session.layerCaches_[i];
            if (cache.device_state_dirty) {
                if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
                    last_d2h_event = ctx.memcpy_d2h_async(cache.host_linear_state.data(), cache.linear_state.data(),
                                                          cache.host_linear_state.size() * sizeof(float));
                }
                if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
                    last_d2h_event = ctx.memcpy_d2h_async(cache.host_linear_conv_state.data(), cache.linear_conv_state.data(),
                                                          cache.host_linear_conv_state.size() * sizeof(float));
                }
                cache.device_state_dirty = false;
            }
        }
        if (last_d2h_event != sycl::event{}) {
            last_d2h_event.wait_and_throw();
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) continue;
            auto& cache = session.layerCaches_[i];
            snapshot.layers[i].linear_state_host = cache.host_linear_state;
            snapshot.layers[i].linear_conv_state_host = cache.host_linear_conv_state;
            snapshot.layers[i].linear_conv_head = cache.linear_conv_head;
        }
        ple_.capture_snapshot(ctx, session.ple_state_, snapshot.ple_snapshot);
    } catch (...) {
        snapshots.erase(nextLength);
        throw;
    }
}

int Qwen4ExpBnb4Backend::Truncate(Session& session, int newLength) {
    ValidateSession(session);
    if (newLength < 0 || newLength > session.currentLength_) {
        return session.currentLength_;
    }
    session.context_->synchronize();
    if (newLength == 0) {
        Reset(session);
        return 0;
    }
    int target = -1;
    for (const auto& kv : session.snapshots_) {
        if (kv.first <= newLength && kv.first > target) {
            target = kv.first;
        }
    }
    if (target < 0) {
        Reset(session);
        return 0;
    }
    const auto& snapshot = session.snapshots_.at(target);
    sycl::event last_h2d_event{};
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (!layers_[i].is_linear) continue;
        auto& cache = session.layerCaches_[i];
        const auto& snap_layer = snapshot.layers[i];
        cache.host_linear_state = snap_layer.linear_state_host;
        cache.host_linear_conv_state = snap_layer.linear_conv_state_host;
        cache.linear_conv_head = snap_layer.linear_conv_head;
        if (!cache.host_linear_state.empty() && cache.linear_state.valid()) {
            last_h2d_event = session.context_->memcpy_h2d_async(cache.linear_state.data(), cache.host_linear_state.data(),
                                                                cache.host_linear_state.size() * sizeof(float));
        }
        if (!cache.host_linear_conv_state.empty() && cache.linear_conv_state.valid()) {
            last_h2d_event = session.context_->memcpy_h2d_async(cache.linear_conv_state.data(), cache.host_linear_conv_state.data(),
                                                                cache.host_linear_conv_state.size() * sizeof(float));
        }
        cache.device_state_dirty = false;
    }
    if (last_h2d_event != sycl::event{}) {
        last_h2d_event.wait_and_throw();
    }
    ple_.restore_snapshot(*session.context_, session.ple_state_, snapshot.ple_snapshot);

    for (auto it = session.snapshots_.begin(); it != session.snapshots_.end(); ) {
        if (it->first > target) it = session.snapshots_.erase(it);
        else ++it;
    }
    session.currentLength_ = target;
    return target;
}

void Qwen4ExpBnb4Backend::Reset(Session& session) {
    ValidateSession(session);
    session.context_->synchronize();
    session.snapshots_.clear();
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        if (layers_[i].is_linear) {
            if (cache.linear_state.valid()) {
                session.context_->queue().memset(
                    cache.linear_state.data(), 0, cache.linear_state.size_bytes());
            }
            if (cache.linear_conv_state.valid()) {
                session.context_->queue().memset(
                    cache.linear_conv_state.data(), 0, cache.linear_conv_state.size_bytes());
            }
            std::fill(cache.host_linear_state.begin(), cache.host_linear_state.end(), 0.0f);
            std::fill(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(), 0.0f);
            cache.device_state_dirty = false;
            cache.linear_conv_head = 0;
        }
    }
    ple_.init_state(*session.context_, session.ple_state_);
    session.currentLength_ = 0;
    session.context_->synchronize();
}

bool Qwen4ExpBnb4Backend::truncate_kv_cache(int new_len) {
    if (new_len >= get_current_context_len()) return true;
    if (new_len <= 0) {
        reset();
        return false;
    }
    return Truncate(DefaultSession(*context_), new_len) > 0;
}

void Qwen4ExpBnb4Backend::reset() {
    if (defaultSession_) Reset(*defaultSession_);
}

std::unique_ptr<Qwen4ExpBnb4Backend::Session> Qwen4ExpBnb4Backend::CreateSession(
    Context& ctx, std::size_t maxSnapshots) {
    auto session = std::unique_ptr<Session>(new Session(ctx, max_seq_len_, sessionOwner_, true, maxSnapshots));
    ple_.init_state(ctx, session->ple_state_);
    return session;
}

Tensor& Qwen4ExpBnb4Backend::Forward(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count) {
    return ForwardImpl(ctx, session, tokenIdsDevice, count, true);
}

Tensor& Qwen4ExpBnb4Backend::ForwardLegacyLogits(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count) {
    return ForwardImpl(ctx, session, tokenIdsDevice, count, false);
}

Tensor& Qwen4ExpBnb4Backend::forward(
    Context& ctx, const int* token_ids_device, int seq_len) {
    return Forward(ctx, DefaultSession(ctx), token_ids_device, seq_len);
}

Qwen4ExpBnb4Backend::Session& Qwen4ExpBnb4Backend::DefaultSession(Context& ctx) {
    if (!defaultSession_) {
        defaultSession_ = CreateSession(ctx);
    }
    return *defaultSession_;
}

void Qwen4ExpBnb4Backend::ValidateSession(const Session& session) const {
    if (session.owner_ != sessionOwner_) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: session belongs to another backend");
    }
}

void Qwen4ExpBnb4Backend::set_embedding_overrides(
    const std::vector<int>& positions,
    const std::vector<sycl::ext::oneapi::bfloat16>& embeddings,
    int hidden_size) {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    session.embeddingOverridePositions_ = positions;
    session.embeddingOverrideValues_ = embeddings;
    session.embeddingOverrideHiddenSize_ = hidden_size;
}

void Qwen4ExpBnb4Backend::clear_embedding_overrides() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    session.embeddingOverridePositions_.clear();
    session.embeddingOverrideValues_.clear();
    session.embeddingOverrideHiddenSize_ = 0;
}

void Qwen4ExpBnb4Backend::set_mrope_positions(
    Context& ctx,
    const std::vector<int>& pos_t,
    const std::vector<int>& pos_h,
    const std::vector<int>& pos_w,
    int text_pos_delta) {
    auto& session = DefaultSession(ctx);
    if (pos_t.empty() || pos_h.empty() || pos_w.empty() ||
        pos_t.size() != pos_h.size() || pos_t.size() != pos_w.size()) {
        clear_mrope_positions();
        return;
    }
    session.mropePosT_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_t.size())}, dnnl::memory::data_type::s32);
    session.mropePosH_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_h.size())}, dnnl::memory::data_type::s32);
    session.mropePosW_ = Tensor::allocate(ctx, {static_cast<int64_t>(pos_w.size())}, dnnl::memory::data_type::s32);
    ctx.memcpy_h2d(session.mropePosT_.data(), pos_t.data(), pos_t.size() * sizeof(int));
    ctx.memcpy_h2d(session.mropePosH_.data(), pos_h.data(), pos_h.size() * sizeof(int));
    ctx.memcpy_h2d(session.mropePosW_.data(), pos_w.data(), pos_w.size() * sizeof(int));
    session.mropePromptLength_ = static_cast<int>(pos_t.size());
    session.mropeTextPositionDelta_ = text_pos_delta;
}

void Qwen4ExpBnb4Backend::clear_mrope_positions() {
    if (!defaultSession_) return;
    auto& session = *defaultSession_;
    session.mropePosT_ = Tensor();
    session.mropePosH_ = Tensor();
    session.mropePosW_ = Tensor();
    session.mropePromptLength_ = 0;
    session.mropeTextPositionDelta_ = 0;
}
