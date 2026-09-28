#include "Qwen4ExpBnb4Backend.hpp"
#include "../ops/QsaIndexerOps.hpp"
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
    std::size_t maxSnapshots)
    : context_(&ctx), owner_(std::move(owner)),
      capacity_(capacity), snapshots_(maxSnapshots) {
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
            cache.indexer_k = Tensor::allocate(ctx, {capacity, 128}, dnnl::memory::data_type::bf16);
            const int max_blocks = std::max(1, capacity / 4);
            cache.indexer_compressed_k = Tensor::allocate(ctx, {max_blocks, 128}, dnnl::memory::data_type::bf16);
            cache.num_compressed_blocks = 0;
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

    if (expert_offsets_host_) context_->free_host(expert_offsets_host_);
    if (expert_counts_host_) context_->free_host(expert_counts_host_);
    if (expert_curr_pos_host_) context_->free_host(expert_curr_pos_host_);
    if (expert_quad_pos_host_) context_->free_host(expert_quad_pos_host_);
    if (expert_pair_pos_host_) context_->free_host(expert_pair_pos_host_);
    if (expert_single_pos_host_) context_->free_host(expert_single_pos_host_);
    if (expert_map_token_host_) context_->free_host(expert_map_token_host_);
    if (expert_map_rank_host_) context_->free_host(expert_map_rank_host_);
    if (expert_map_slot_host_) context_->free_host(expert_map_slot_host_);
    if (expert_map_token_dev_) context_->free_device(expert_map_token_dev_);
    if (expert_map_rank_dev_) context_->free_device(expert_map_rank_dev_);
    if (expert_map_slot_dev_) context_->free_device(expert_map_slot_dev_);
    expert_offsets_host_ = nullptr;
    expert_counts_host_ = nullptr;
    expert_curr_pos_host_ = nullptr;
    expert_quad_pos_host_ = nullptr;
    expert_pair_pos_host_ = nullptr;
    expert_single_pos_host_ = nullptr;
    expert_map_token_host_ = nullptr;
    expert_map_rank_host_ = nullptr;
    expert_map_slot_host_ = nullptr;
    expert_map_token_dev_ = nullptr;
    expert_map_rank_dev_ = nullptr;
    expert_map_slot_dev_ = nullptr;
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
    max_seq_len_ = max_seq_len;

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

    static const bool s_q4e_async_dequant = aila::env::read_flag("AILA_Q4E_ASYNC_DEQUANT", true);
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
        if (s_q4e_async_dequant) {
            linear.set_allow_async_dequant(true);
        }
        static const bool s_direct_prefill_io = aila::env::read_flag("AILA_Q4E_DIRECT_PREFILL_IO", false);
        linear.set_direct_prefill_io(s_direct_prefill_io);
        static const bool s_q4e_fused_prefill = aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
        if (s_q4e_fused_prefill) {
            linear.set_allow_dense_xmx(true);
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

            // QSA Indexer (M2 & W2)
            const std::string index_qk_name = prefix + "self_attn.indexer.index_qk_proj.weight";
            if (weights.has_host(index_qk_name) || weights.has(index_qk_name)) {
                const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                layer.index_qk_weight = &weights.upload(index_qk_name, ctx);
                layer.index_qk_proj.init(ctx, *layer.index_qk_weight, hidden_size_, indexer_out_dim, false);
                const int indexer_k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
                const size_t k_offset = static_cast<size_t>(cfg_.indexer_n_heads * cfg_.indexer_head_dim) * hidden_size_;
                layer.index_k_weight = Tensor::view(
                    ctx, layer.index_qk_weight->data_as<bf16>() + k_offset,
                    {indexer_k_dim, hidden_size_});
                layer.index_k_proj.init(ctx, layer.index_k_weight, hidden_size_, indexer_k_dim, false);
                layer.index_q_norm_weight = plus_one_norm(prefix + "self_attn.indexer.q_layernorm.weight");
                layer.index_k_norm_weight = plus_one_norm(prefix + "self_attn.indexer.k_layernorm.weight");
            }
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
        const size_t linear_layers = layers_.size() - full_layers;
        const size_t kv_bytes = full_layers * 2 * cfg_.num_key_value_heads * cfg_.head_dim * sizeof(bf16) *
                                static_cast<size_t>(max_seq_len_);
        const size_t indexer_k_bytes = full_layers * cfg_.indexer_head_dim * sizeof(bf16) * static_cast<size_t>(max_seq_len_);
        const size_t indexer_compressed_k_bytes = full_layers * (static_cast<size_t>(max_seq_len_) / 4) * cfg_.indexer_head_dim * sizeof(bf16);
        const size_t recurrent_bytes = linear_layers * (48 * 128 * 128 + 3 * 10240) * sizeof(float);
        const size_t total_session_state = kv_bytes + indexer_k_bytes + indexer_compressed_k_bytes + recurrent_bytes;
        const size_t max_reserve =
            static_cast<size_t>(std::max(0, aila::env::read_int("AILA_Q4E_VRAM_RESERVE_MB", 2560))) << 20;
        const size_t reserve = max_reserve + total_session_state;
        const size_t min_bytes = static_cast<size_t>(1024) << 20;
        const size_t used = ctx.current_allocated_bytes() + reserve;
        expert_cache_bytes = vram > used + min_bytes ? vram - used : min_bytes;

        AILA_LOG_INFO(
            "[Qwen4Exp] Memory Admission Accounting: Total VRAM=%.0f MB, Static Weights=%.1f MB, "
            "Session State=%.1f MB (KV=%.1f MB, IndexerK=%.1f MB, IndexerCompressedK=%.1f MB, Recurrent=%.1f MB), "
            "Reserve Headroom=%.0f MB, Expert Cache=%.0f MB",
            vram / (1024.0 * 1024.0),
            ctx.current_allocated_bytes() / (1024.0 * 1024.0),
            total_session_state / (1024.0 * 1024.0),
            kv_bytes / (1024.0 * 1024.0),
            indexer_k_bytes / (1024.0 * 1024.0),
            indexer_compressed_k_bytes / (1024.0 * 1024.0),
            recurrent_bytes / (1024.0 * 1024.0),
            max_reserve / (1024.0 * 1024.0),
            expert_cache_bytes / (1024.0 * 1024.0));
    }
    AILA_LOG_INFO("[Qwen4Exp] Initializing Expert Cache (%.0f MB)...", expert_cache_bytes / (1024.0 * 1024.0));
    if (!expert_cache_.init(ctx, weights, cfg_.num_hidden_layers, expert_cache_bytes,
                            cfg_.num_experts, hidden_size_, cfg_.moe_intermediate_size, 64, error_message)) {
        return false;
    }

    ensure_runtime_buffers(ctx, 1);
    expert_executor_.init(ctx, expert_cache_, max_seq_len_, cfg_.num_experts_per_tok, hidden_size_, cfg_.moe_intermediate_size);
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

    // With K-only indexer projection (A2) the engine's 2-token warmup completes no QSA block, so
    // it launches no QsaIndexerOps kernel. The first request would then pay that module's JIT
    // (~1.2 s on A770) inside its first prefill; compress one dummy block here to move it to load.
    for (const auto& layer : layers_) {
        if (layer.is_linear || !layer.index_k_norm_weight) continue;
        const int k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
        const int ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
        Tensor dummy_k = Tensor::allocate(ctx, {ratio, k_dim}, dnnl::memory::data_type::bf16);
        Tensor dummy_compressed = Tensor::allocate(ctx, {1, k_dim}, dnnl::memory::data_type::bf16);
        ctx.queue().memset(dummy_k.data(), 0, static_cast<size_t>(ratio) * k_dim * sizeof(bf16));
        ops::qsa_compress_key_blocks(ctx, dummy_k, *layer.index_k_norm_weight, dummy_compressed,
                                     0, 1, cfg_.rope.rope_theta, false, nullptr, nullptr, nullptr,
                                     0, 0, cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1],
                                     cfg_.rope.mrope_section[2]);
        ctx.synchronize();
        break;
    }
    const bool s_expert_major = aila::env::read_flag("AILA_Q4E_EXPERT_MAJOR_PREFILL", true);
    const bool s_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);
    const bool s_pred_ids_only = aila::env::read_flag("AILA_Q4E_PRED_IDS_ONLY", true);
    const char* prefill_mode = s_expert_major ? "S4 (expert-major W4/G3)" : (s_moe_rows ? "S1 (batched-rows)" : "baseline (token-by-token)");
    AILA_LOG_INFO("[Qwen4Exp] Prefill MoE mode: %s, Router pred: %s",
                  prefill_mode, s_pred_ids_only ? "ids-only" : "full-softmax");
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
    // D1: Limit dense attention scratchpads to the maximum dense prefill length (2052).
    // Prefill exceeding this threshold runs sparse QSA attention via decode_scores and
    // decode_attn_partials, eliminating the 6 GiB - 96 GiB incr_scores memory explosion.
    constexpr int kDensePrefillLimit = 2052;
    const int dense_cap = std::min(cap, kDensePrefillLimit);
    const int dense_kv_cap = std::min(max_seq_len_, kDensePrefillLimit);
    buf_.scores = Tensor::allocate(ctx, {full_q_heads_, dense_cap, dense_cap}, dnnl::memory::data_type::f32);
    buf_.incr_scores = Tensor::allocate(ctx, {full_q_heads_, dense_cap, dense_kv_cap}, dnnl::memory::data_type::f32);

    // QSA Indexer buffers (M2, A3, A4)
    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
    const int indexer_q_dim = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
    buf_.indexer_qk = Tensor::allocate(ctx, {cap, indexer_out_dim}, dnnl::memory::data_type::bf16);
    buf_.indexer_q = Tensor::allocate(ctx, {cap, indexer_q_dim}, dnnl::memory::data_type::bf16);
    const int max_complete_blocks = (max_seq_len_ + 3) / 4;
    buf_.indexer_block_scores = Tensor::allocate(ctx, {max_complete_blocks}, dnnl::memory::data_type::f32);
    buf_.indexer_active_tokens = Tensor::allocate(ctx, {std::max(max_seq_len_, cfg_.indexer_budget + cfg_.indexer_compress_ratio)}, dnnl::memory::data_type::s32);
    const int block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / (cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4);
    buf_.indexer_active_blocks = Tensor::allocate(ctx, {block_topk + 4}, dnnl::memory::data_type::s32);
    static const int s_qsa_logits_budget = aila::env::read_int("AILA_QSA_LOGITS_BUDGET_BYTES", 4 * 1024 * 1024);
    const size_t row_bytes = static_cast<size_t>(max_complete_blocks) * sizeof(float);
    int default_tile = 64;
    if (row_bytes > 0) {
        default_tile = std::clamp(static_cast<int>(s_qsa_logits_budget / row_bytes), 16, 64);
    }
    qsa_query_tile_ = aila::env::read_int("AILA_QSA_QUERY_TILE", default_tile);
    buf_.qsa_tile_block_scores = Tensor::allocate(ctx, {qsa_query_tile_, max_complete_blocks}, dnnl::memory::data_type::f32);
    buf_.qsa_tile_active_blocks = Tensor::allocate(ctx, {qsa_query_tile_, block_topk + 4}, dnnl::memory::data_type::s32);
    buf_.qsa_tile_valid_counts = Tensor::allocate(ctx, {qsa_query_tile_, 2}, dnnl::memory::data_type::s32);


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
    embed_host_ids_ = static_cast<int*>(ctx.alloc_host(static_cast<size_t>(cap) * sizeof(int)));
    if (embed_host_.valid()) {
        embed_host_rows_ = static_cast<uint8_t*>(ctx.alloc_host(static_cast<size_t>(cap) * hidden_size_ * sizeof(bf16)));
    }
    const bool s_expert_major = aila::env::read_flag("AILA_Q4E_EXPERT_MAJOR_PREFILL", true);
    const bool s_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);

    if (s_expert_major || s_moe_rows) {
        batch_weights_dev_ = static_cast<float*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(float)));
    }
    if (s_moe_rows && !s_expert_major) {
        batch_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
        batch_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
    }
    buf_.moe_h = Tensor::allocate(ctx, {cap, cfg_.num_experts_per_tok, cfg_.moe_intermediate_size}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_buf = Tensor::allocate(ctx, {cap, 1}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_up_buf = Tensor::allocate(ctx, {cap, 1280}, dnnl::memory::data_type::bf16);
    buf_.shared_inter_buf = Tensor::allocate(ctx, {cap, 640}, dnnl::memory::data_type::bf16);
    buf_.shared_out_buf = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    if (s_expert_major) {
        buf_.moe_down_acc = Tensor::allocate(ctx, {cap, cfg_.num_experts_per_tok, hidden_size_}, dnnl::memory::data_type::f32);

        expert_offsets_host_ = static_cast<int*>(ctx.alloc_host((cfg_.num_experts + 1) * sizeof(int)));
        expert_counts_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_curr_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_quad_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_pair_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_single_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_map_token_host_ = static_cast<int*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_rank_host_ = static_cast<int*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_slot_host_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_host(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
        expert_map_token_dev_ = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_rank_dev_ = static_cast<int*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(int)));
        expert_map_slot_dev_ = static_cast<ops::MoeExpertSlotDev*>(
            ctx.alloc_device(static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertSlotDev)));
    } else {
        buf_.moe_down_acc = Tensor{};
    }

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

void Qwen4ExpBnb4Backend::RunHyperConnectionCombined(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject,
    const bf16* pending_y, const float* pending_inj) {
    if (pending_y && pending_inj) {
        ops::hc_inject_add_and_grouped_rms_norm(
            ctx, static_cast<bf16*>(buf_.hidden4.data()),
            pending_y, pending_inj,
            static_cast<const float*>(hc.hc_norm_weight->data()),
            cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
            count, 4, hidden_size_, true);
    } else {
        ops::grouped_rms_norm(
            ctx, static_cast<const bf16*>(buf_.hidden4.data()),
            static_cast<const float*>(hc.hc_norm_weight->data()),
            cfg_.rms_norm_eps, static_cast<bf16*>(buf_.hc_N.data()),
            count, 4, hidden_size_, true);
    }

    const int hc_dim = 4 * hidden_size_;
    const int lowrank = cfg_.hc_lowrank;

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

void Qwen4ExpBnb4Backend::RunHyperConnection(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject) {
    RunHyperConnectionCombined(ctx, hc, count, with_inject, nullptr, nullptr);
}

Tensor& Qwen4ExpBnb4Backend::ForwardImpl(
    Context& ctx, Session& session,
    const int* tokenIdsDevice, int count, bool lastTokenOnly,
    const ForwardOptions& options) {
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
    const int checkpoint_after = options.snapshot_policy == SnapshotPolicy::ReusablePrefix
        ? options.checkpoint_after : 0;
    if (checkpoint_after < 0 || (checkpoint_after > 0 &&
        (checkpoint_after < 2 || count - checkpoint_after < 2 || count > 64 ||
         cached_len > 2052 || session.snapshots_.size() < 2))) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: interior checkpoint requires short dense prefill, two multi-token sides and two slots");
    }
    StateSnapshot* interior = checkpoint_after > 0
        ? &ReserveSnapshot(session, start_pos + checkpoint_after) : nullptr;
    Bnb4BitLinear::PrefillDispatchStats prefill_stats_t0{};
    if (count > 1) {
        prefill_stats_t0 = Bnb4BitLinear::get_prefill_dispatch_stats();
    }

    // 1. Embedding lookup + optional vision embedding override
    EmbedTokens(ctx, tokenIdsDevice, count);

    // P0: Early PLE prepare pipeline (pure prepare + async DMA to double-buffered staging)
    const int* ple_ids_ptr = embed_host_ids_;
    if (!embed_host_.valid()) {
        ctx.memcpy_d2h(embed_host_ids_, tokenIdsDevice, static_cast<size_t>(count) * sizeof(int));
    }
    ple_ticket_ = ple_.prepare(ctx, session.ple_state_, ple_ids_ptr, count,
                               &session, session.generation(), start_pos);

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

    sycl::event fallback_last_down_event{};

    // 2. 48 Transformer Layers
    for (int L = 0; L < cfg_.num_hidden_layers; ++L) {
        auto& layer = layers_[L];
        auto& cache = session.layerCaches_[L];

        // 2.0. PLE on Layer 1 (Epilogue Residual Fused via P0 PleTicket)
        if (L == 1) {
            if (!ple_ticket_.valid || ple_ticket_.session_ptr != &session ||
                ple_ticket_.generation != session.generation() || ple_ticket_.start_pos != start_pos) {
                ple_ticket_ = ple_.prepare(ctx, session.ple_state_, ple_ids_ptr, count,
                                           &session, session.generation(), start_pos);
            }
            ple_.consume(ctx, ple_ticket_, session.ple_state_,
                         hidden4_ptr, nullptr, hidden4_ptr, checkpoint_after,
                         interior ? &interior->ple_snapshot : nullptr);
            ple_.commit(ple_ticket_, session.ple_state_);
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
                const auto run_range = [&](int offset, int length) {
                    const bf16* input = buf_.linear_all.data_as<bf16>() + static_cast<size_t>(offset) * linear_all_dim_;
                    bf16* output = buf_.attn_out.data_as<bf16>() + static_cast<size_t>(offset) * linear_kv_dim_;
                    ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, d_weights, d_state,
                                                             input, output, length);
                };
                if (interior) {
                    run_range(0, checkpoint_after);
                    auto& snapshot = interior->layers[L];
                    snapshot.linear_conv_head = d_state.conv_head;
                    // In-order D2H completes before the suffix may overwrite the recurrent state.
                    ctx.memcpy_d2h_async(snapshot.linear_state_host.data(), cache.linear_state.data(),
                                         cache.linear_state.size_bytes());
                    ctx.memcpy_d2h_async(snapshot.linear_conv_state_host.data(), cache.linear_conv_state.data(),
                                         cache.linear_conv_state.size_bytes());
                    run_range(checkpoint_after, count - checkpoint_after);
                } else {
                    run_range(0, count);
                }
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

                const int* active_indices = nullptr;
                int num_active_tokens = 0;
                const int* active_blocks = nullptr;
                int num_active_blocks = 0;
                if (layer.index_qk_weight && layer.index_q_norm_weight && layer.index_k_norm_weight && cache.indexer_k.valid()) {
                    const int compress_ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
                    const int block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / compress_ratio;
                    const int visible_len = start_pos + 1;
                    const int num_complete_blocks = visible_len / compress_ratio;

                    static const bool s_qsa_compressed_k = aila::env::read_flag("AILA_QSA_COMPRESSED_K", true);
                    static const bool s_qsa_k_only_proj = aila::env::read_flag("AILA_QSA_K_ONLY_PROJ", true);
                    static const bool s_qsa_block_native = aila::env::read_flag("AILA_QSA_BLOCK_NATIVE", true);
                    const bool dense_phase = (num_complete_blocks <= block_topk);

                    if (dense_phase && s_qsa_k_only_proj && layer.index_k_weight.valid()) {
                        // A2: K-only projection for visible_len <= 2051. Eliminates 80% GEMV output rows (512 Q rows skipped).
                        const int k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
                        Tensor k_target = Tensor::view(
                            ctx, cache.indexer_k.data_as<bf16>() + static_cast<size_t>(start_pos) * k_dim,
                            {1, k_dim});
                        DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, buf_.x, k_target,
                                    1, k_dim, hidden_size_);

                        // A1: If this token completed a block, incrementally compress it into indexer_compressed_k
                        if (s_qsa_compressed_k && visible_len % compress_ratio == 0 &&
                            num_complete_blocks > cache.num_compressed_blocks && cache.indexer_compressed_k.valid()) {
                            const int blocks_to_compress = num_complete_blocks - cache.num_compressed_blocks;
                            ops::qsa_compress_key_blocks(
                                ctx, cache.indexer_k, *layer.index_k_norm_weight, cache.indexer_compressed_k,
                                cache.num_compressed_blocks, blocks_to_compress, cfg_.rope.rope_theta,
                                cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2]);
                            cache.num_compressed_blocks = num_complete_blocks;
                        }
                        num_active_tokens = 0;
                    } else {
                        const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                        DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, buf_.x, buf_.indexer_qk,
                                    1, indexer_out_dim, hidden_size_);

                        if (s_qsa_block_native) {
                            num_active_blocks = ops::qsa_prepare_causal_query_and_select_blocks(
                                ctx, buf_.indexer_qk, *layer.index_q_norm_weight, *layer.index_k_norm_weight,
                                cache.indexer_k, buf_.indexer_q, buf_.indexer_block_scores,
                                buf_.indexer_active_blocks, start_pos, block_topk, compress_ratio,
                                cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2],
                                (s_qsa_compressed_k && cache.indexer_compressed_k.valid() ? &cache.indexer_compressed_k : nullptr),
                                &cache.num_compressed_blocks);
                            if (num_active_blocks > 0) {
                                active_blocks = buf_.indexer_active_blocks.data_as<int>();
                            }
                        } else {
                            num_active_tokens = ops::qsa_prepare_causal_query_and_select_tokens(
                                ctx, buf_.indexer_qk, *layer.index_q_norm_weight, *layer.index_k_norm_weight,
                                cache.indexer_k, buf_.indexer_q, buf_.indexer_block_scores,
                                buf_.indexer_active_tokens, start_pos, block_topk, compress_ratio,
                                cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2],
                                (s_qsa_compressed_k && cache.indexer_compressed_k.valid() ? &cache.indexer_compressed_k : nullptr),
                                &cache.num_compressed_blocks);
                            if (num_active_tokens > 0) {
                                active_indices = buf_.indexer_active_tokens.data_as<int>();
                            }
                        }
                    }
                }

                ops::attention_decode(
                    ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.decode_scores,
                    full_q_heads_, full_kv_heads_, full_head_dim_, cached_len,
                    &buf_.decode_attn_partials, &buf_.z, active_indices, num_active_tokens,
                    active_blocks, num_active_blocks);

                layer.o_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.attn_out, buf_.y, 1);
            } else {
                const bool has_qsa_indexer = layer.index_qk_weight && layer.index_q_norm_weight &&
                                             layer.index_k_norm_weight && cache.indexer_k.valid();
                const int compress_ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
                const int block_topk = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / compress_ratio;
                const bool sparse_prefill = has_qsa_indexer &&
                    (start_pos + count) / compress_ratio > block_topk;

                static const bool s_qsa_compressed_k = aila::env::read_flag("AILA_QSA_COMPRESSED_K", true);
                static const bool s_qsa_k_only_proj = aila::env::read_flag("AILA_QSA_K_ONLY_PROJ", true);

                if (has_qsa_indexer) {
                    if (!sparse_prefill && s_qsa_k_only_proj && layer.index_k_weight.valid()) {
                        // A2: K-only projection for dense prefill chunks (start_pos + count <= 2051).
                        // Skips 512 Q projection rows for all count tokens.
                        const int k_dim = cfg_.indexer_kv_heads * cfg_.indexer_head_dim;
                        Tensor k_target = Tensor::view(
                            ctx, cache.indexer_k.data_as<bf16>() + static_cast<size_t>(start_pos) * k_dim,
                            {count, k_dim});
                        DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, buf_.x, k_target,
                                    count, k_dim, hidden_size_);
                    } else {
                        const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                        DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, buf_.x, buf_.indexer_qk,
                                    count, indexer_out_dim, hidden_size_);
                    }
                }

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

                if (sparse_prefill) {
                    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                    const int indexer_q_dim = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
                    const int q_dim = full_q_heads_ * full_head_dim_;
                    const bf16* indexer_qk_base = buf_.indexer_qk.data_as<bf16>();
                    bf16* indexer_q_base = buf_.indexer_q.data_as<bf16>();
                    bf16* q_base = buf_.q.data_as<bf16>();
                    bf16* z_base = buf_.z.data_as<bf16>();
                    bf16* attn_out_base = buf_.attn_out.data_as<bf16>();

                    static const bool s_batched_qsa_prefill = aila::env::read_flag("AILA_QSA_BATCHED_PREFILL", true);
                    if (s_batched_qsa_prefill && cache.indexer_compressed_k.valid()) {
                        // Phase P2 (R4d): Reentrant fused pre-indexer (raw K caching + block compression + Q norm/RoPE)
                        ops::qsa_pre_indexer_batched(
                            ctx, buf_.indexer_qk, *layer.index_q_norm_weight, *layer.index_k_norm_weight,
                            cache.indexer_k, buf_.indexer_q,
                            (s_qsa_compressed_k && cache.indexer_compressed_k.valid() ? &cache.indexer_compressed_k : nullptr),
                            &cache.num_compressed_blocks,
                            start_pos, count, compress_ratio,
                            cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                            (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                            (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                            (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                            session.mropePromptLength_, session.mropeTextPositionDelta_,
                            cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2]);

                        for (int tile_offset = 0; tile_offset < count; tile_offset += qsa_query_tile_) {
                            const int tile_count = std::min(qsa_query_tile_, count - tile_offset);
                            const int tile_start_pos = start_pos + tile_offset;
                            const int tile_max_blocks = std::max(1, (tile_start_pos + tile_count) / compress_ratio);

                            Tensor q_tile = Tensor::view(
                                ctx, buf_.indexer_q.data_as<bf16>() + static_cast<size_t>(tile_offset) * indexer_q_dim,
                                {tile_count, indexer_q_dim});
                            Tensor q_full_tile = Tensor::view(
                                ctx, buf_.q.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                                {tile_count, q_dim});
                            Tensor z_tile = Tensor::view(
                                ctx, buf_.z.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                                {tile_count, q_dim});
                            Tensor attn_out_tile = Tensor::view(
                                ctx, buf_.attn_out.data_as<bf16>() + static_cast<size_t>(tile_offset) * q_dim,
                                {tile_count, q_dim});
                            Tensor valid_counts_tile = Tensor::view(
                                ctx, buf_.qsa_tile_valid_counts.data_as<int>(),
                                {tile_count, 2});

                            ops::qsa_compute_block_scores_compressed_batched(
                                ctx, q_tile, cache.indexer_compressed_k, buf_.qsa_tile_block_scores,
                                tile_count, tile_max_blocks, compress_ratio, tile_start_pos);

                            ops::qsa_select_topk_blocks_batched(
                                ctx, buf_.qsa_tile_block_scores, buf_.qsa_tile_active_blocks,
                                tile_count, tile_max_blocks, tile_start_pos, block_topk, compress_ratio,
                                &valid_counts_tile);

                            ops::attention_prefill_qsa_batched(
                                ctx, q_full_tile, cache.k, cache.v, buf_.qsa_tile_active_blocks, attn_out_tile,
                                tile_count, tile_start_pos, full_q_heads_, full_kv_heads_, full_head_dim_,
                                block_topk, compress_ratio, &z_tile, &valid_counts_tile);
                        }
                    } else {
                        // Fallback: cache raw K and compress blocks before row-by-row execution
                        ops::qsa_cache_k_batched(ctx, buf_.indexer_qk, cache.indexer_k, start_pos, count);
                        const int total_complete_blocks = (start_pos + count) / compress_ratio;
                        if (s_qsa_compressed_k && total_complete_blocks > cache.num_compressed_blocks &&
                            cache.indexer_compressed_k.valid()) {
                            const int blocks_to_compress = total_complete_blocks - cache.num_compressed_blocks;
                            ops::qsa_compress_key_blocks(
                                ctx, cache.indexer_k, *layer.index_k_norm_weight, cache.indexer_compressed_k,
                                cache.num_compressed_blocks, blocks_to_compress, cfg_.rope.rope_theta,
                                cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2]);
                            cache.num_compressed_blocks = total_complete_blocks;
                        }
                        // The upstream indexer builds a separate mask from each query's
                        // causal visible prefix. Keep this row-wise path as the correctness
                        // oracle for chunked prefill and appends across the sparse threshold.
                        for (int row = 0; row < count; ++row) {
                            const int pos = start_pos + row;
                            Tensor indexer_qk_row = Tensor::view(
                                ctx, const_cast<bf16*>(indexer_qk_base + static_cast<size_t>(row) * indexer_out_dim),
                                {1, indexer_out_dim});
                            Tensor indexer_q_row = Tensor::view(
                                ctx, indexer_q_base + static_cast<size_t>(row) * indexer_q_dim,
                                {cfg_.indexer_n_heads, cfg_.indexer_head_dim});
                            Tensor q_row = Tensor::view(
                                ctx, q_base + static_cast<size_t>(row) * q_dim, {1, q_dim});
                            Tensor z_row = Tensor::view(
                                ctx, z_base + static_cast<size_t>(row) * q_dim, {1, q_dim});
                            Tensor attn_out_row = Tensor::view(
                                ctx, attn_out_base + static_cast<size_t>(row) * q_dim, {1, q_dim});

                            const int num_active_tokens = ops::qsa_prepare_causal_query_and_select_tokens(
                                ctx, indexer_qk_row, *layer.index_q_norm_weight, *layer.index_k_norm_weight,
                                cache.indexer_k, indexer_q_row, buf_.indexer_block_scores,
                                buf_.indexer_active_tokens, pos, block_topk, compress_ratio,
                                cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2],
                                (s_qsa_compressed_k && cache.indexer_compressed_k.valid() ? &cache.indexer_compressed_k : nullptr),
                                &cache.num_compressed_blocks);
                            const int visible_len = pos + 1;
                            int attention_active_tokens = num_active_tokens;
                            if (attention_active_tokens == 0) {
                                // Keep prefill causal semantics independent of optional decode
                                // window/sink settings by passing an explicit dense prefix.
                                attention_active_tokens = ops::qsa_select_topk_and_expand_tokens(
                                    ctx, buf_.indexer_block_scores, visible_len / compress_ratio,
                                    block_topk, visible_len, compress_ratio, buf_.indexer_active_tokens);
                            }
                            ops::attention_decode(
                                ctx, q_row, cache.k, cache.v, attn_out_row, buf_.decode_scores,
                                full_q_heads_, full_kv_heads_, full_head_dim_, visible_len,
                                &buf_.decode_attn_partials, &z_row,
                                buf_.indexer_active_tokens.data_as<int>(), attention_active_tokens);
                        }
                    }
                    layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.y, count);
                } else {
                    if (has_qsa_indexer) {
                        // Short chunks stay on the existing batched dense prefill path;
                        // only a query whose visible prefix exceeds the block budget uses
                        // the row-wise QSA correctness path above.
                        if (!s_qsa_k_only_proj || !layer.index_k_weight.valid()) {
                            ops::qsa_cache_k_batched(ctx, buf_.indexer_qk, cache.indexer_k, start_pos, count);
                        }
                        const int total_complete_blocks = (start_pos + count) / compress_ratio;
                        if (s_qsa_compressed_k && total_complete_blocks > cache.num_compressed_blocks &&
                            cache.indexer_compressed_k.valid()) {
                            const int blocks_to_compress = total_complete_blocks - cache.num_compressed_blocks;
                            ops::qsa_compress_key_blocks(
                                ctx, cache.indexer_k, *layer.index_k_norm_weight, cache.indexer_compressed_k,
                                cache.num_compressed_blocks, blocks_to_compress, cfg_.rope.rope_theta,
                                cfg_.rope.mrope_interleaved,
                                (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                                (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                                (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                                session.mropePromptLength_, session.mropeTextPositionDelta_,
                                cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2]);
                            cache.num_compressed_blocks = total_complete_blocks;
                        }
                    }
                    if (start_pos == 0) {
                        ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores, count, full_q_heads_, full_kv_heads_, full_head_dim_, &buf_.z);
                    } else {
                        ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores, count, start_pos, full_q_heads_, full_kv_heads_, full_head_dim_, session.Capacity(), &buf_.z);
                    }
                    layer.o_proj.forward(ctx, linear_scratch_, buf_.attn_out, buf_.y, count);
                }
            }
        }

        stage(layer.is_linear ? kStageLinearAttn : kStageFullAttn);

        // 2.3. Attn Residual Add & 2.4. MLP Hyper-Connection (P1 Fused)
        static const bool s_fuse_attn_mlp_gr = aila::env::read_flag("AILA_Q4E_FUSE_ATTN_MLP_GR", true);
        if (s_fuse_attn_mlp_gr) {
            RunHyperConnectionCombined(ctx, layer.mlp_hc, count, true,
                                       static_cast<const bf16*>(buf_.y.data()),
                                       static_cast<const float*>(buf_.hc_inject_f32.data()));
        } else {
            ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                               static_cast<const bf16*>(buf_.y.data()),
                               static_cast<const float*>(buf_.hc_inject_f32.data()),
                               count, hidden_size_, 4);
            RunHyperConnection(ctx, layer.mlp_hc, count, true);
        }
        stage(kStageHc);

        // 2.5. MoE Computation
        DenseLinear(ctx, layer.router_gate, layer.router_w, buf_.x, buf_.router_scores,
                    count, cfg_.num_experts, hidden_size_);
        // Top-k writes straight into pinned host memory; the event wait is the only readback.
        sycl::event actual_route_event = ops::router_softmax_topk_device(
            ctx, static_cast<const bf16*>(buf_.router_scores.data()), nullptr,
            router_host_ids_, router_host_weights_,
            count, cfg_.num_experts, top_k, true);
        const bool predict_next = (count == 1)
            ? ((s_prefetch || s_predict_stats) && (L + 1 < cfg_.num_hidden_layers))
            : (s_prefetch && (s_prefill_prefetch_max > 0) && (L + 1 < cfg_.num_hidden_layers));
        sycl::event prediction_event{};
        if (predict_next) {
            // Route this layer's MoE input through the next layer's router to guess its experts.
            DenseLinear(ctx, layers_[L + 1].router_gate, layers_[L + 1].router_w, buf_.x, pred_scores_,
                        count, cfg_.num_experts, hidden_size_);
            static const bool s_pred_ids_only = aila::env::read_flag("AILA_Q4E_PRED_IDS_ONLY", true);
            if (s_pred_ids_only) {
                prediction_event = ops::router_topk_ids_device(
                    ctx, static_cast<const bf16*>(pred_scores_.data()),
                    pred_host_ids_, count, cfg_.num_experts, top_k);
            } else {
                prediction_event = ops::router_softmax_topk_device(
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

        auto ensure_prediction_ready = [&]() {
            if (prediction_event != sycl::event{}) {
                prediction_event.wait_and_throw();
                prediction_event = sycl::event{};
                if (count == 1) {
                    std::copy(pred_host_ids_, pred_host_ids_ + top_k, pred_ids.begin());
                }
            }
        };

        const auto prof_router_t0 = prof_clock::now();
        // B2: Decoupled route synchronization - wait ONLY for actual router, allowing concurrent prediction
        actual_route_event.wait_and_throw();
        const auto prof_router_t1 = prof_clock::now();
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
        prof_router_sync_ms += prof_ms(prof_router_t0, prof_router_t1);
        stage(kStageRouter);
        // Prefill: acquire the layer's unique experts in one batch (one streamed fill/DMA
        // instead of a blocking round trip per token), then map slots back per token.
        const auto prof_acq_batch_t0 = prof_clock::now();
        static const bool s_cpu_expert_dispatch = aila::env::read_flag("AILA_Q4E_CPU_EXPERT_DISPATCH", false);
        static const bool s_cpu_small_m = aila::env::read_flag("AILA_Q4E_CPU_SMALL_M", false);
        const bool cpu_small_m = s_cpu_expert_dispatch && s_cpu_small_m && count > 1 && count <= 4;
        const bool can_batch = count > 1 && !cpu_small_m && prefill_expert_slot_.size() == static_cast<size_t>(cfg_.num_experts);
        bool batch_acquire = false;
        sycl::event batch_dma_event{};
        static const bool s_expert_major_prefill = aila::env::read_flag("AILA_Q4E_EXPERT_MAJOR_PREFILL", true);
        static const bool s_prefill_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);
        static const int s_demand_microbatch = aila::env::read_int("AILA_Q4E_DEMAND_MICROBATCH", 0);
        const bool use_microbatch_pipeline = can_batch && s_expert_major_prefill && count > 1 && (s_demand_microbatch > 0);

        if (can_batch) {
            std::fill(prefill_expert_uses_.begin(), prefill_expert_uses_.end(), 0);
            prefill_unique_.clear();
            for (int i = 0; i < count * top_k; ++i) {
                const int e = router_host_ids_[i];
                if (prefill_expert_uses_[e]++ == 0) {
                    prefill_unique_.push_back(e);
                }
            }
            if (prefill_unique_.size() <= static_cast<size_t>(expert_cache_.num_slots())) {
                batch_acquire = true;
                prefill_unique_slots_.resize(prefill_unique_.size());
                if (!use_microbatch_pipeline) {
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
            }
        }
        if (s_prefetch && predict_next && count > 1 && batch_acquire && !use_microbatch_pipeline) {
            ensure_prediction_ready();
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
            expert_cache_.prefetch_async(L + 1, prefill_pred_unique_.data(),
                                         static_cast<int>(prefill_pred_unique_.size()), prefetch_count);
        }
        prof_acquire_ms += prof_ms(prof_acq_batch_t0, prof_clock::now());

        if (batch_acquire && s_expert_major_prefill && count > 1) {
            if (use_microbatch_pipeline) {
                // Phase B6: Demand microbatch double-buffered pipeline
                // Partition unique experts: ready (GPU resident hits) first, then misses
                std::stable_partition(prefill_unique_.begin(), prefill_unique_.end(),
                                      [&](int e) { return expert_cache_.contains(L, e); });

                const int total_acts = count * top_k;
                std::fill_n(expert_counts_host_, cfg_.num_experts, 0);
                for (int i = 0; i < total_acts; ++i) {
                    const int e = router_host_ids_[i];
                    if (e >= 0 && e < cfg_.num_experts) {
                        expert_counts_host_[e]++;
                    }
                }
                static const bool s_q4e_weight_reuse = aila::env::read_flag("AILA_Q4E_WEIGHT_REUSE", true);
                static const bool s_q4e_4token = aila::env::read_flag("AILA_Q4E_4TOKEN_REUSE", true);
                if (s_q4e_weight_reuse) {
                    int cur_chunk_start = 0;
                    size_t scan_u = 0;
                    while (scan_u < prefill_unique_.size()) {
                        const size_t c_u = std::min<size_t>(static_cast<size_t>(s_demand_microbatch),
                                                            prefill_unique_.size() - scan_u);
                        int c_quads = 0;
                        int c_pairs = 0;
                        int c_acts = 0;
                        for (size_t g = 0; g < c_u; ++g) {
                            const int e = prefill_unique_[scan_u + g];
                            const int cnt = expert_counts_host_[e];
                            if (s_q4e_4token) {
                                c_quads += cnt / 4;
                                c_pairs += (cnt % 4) / 2;
                            } else {
                                c_pairs += cnt / 2;
                            }
                            c_acts += cnt;
                        }
                        expert_offsets_host_[prefill_unique_[scan_u]] = cur_chunk_start;
                        int quad_off = cur_chunk_start;
                        int pair_off = cur_chunk_start + 4 * c_quads;
                        int single_off = cur_chunk_start + 4 * c_quads + 2 * c_pairs;
                        for (size_t g = 0; g < c_u; ++g) {
                            const int e = prefill_unique_[scan_u + g];
                            const int cnt = expert_counts_host_[e];
                            if (s_q4e_4token) {
                                expert_quad_pos_host_[e] = quad_off;
                                expert_pair_pos_host_[e] = pair_off;
                                expert_single_pos_host_[e] = single_off;
                                quad_off += (cnt / 4) * 4;
                                pair_off += ((cnt % 4) / 2) * 2;
                                single_off += (cnt % 2);
                            } else {
                                expert_pair_pos_host_[e] = pair_off;
                                expert_single_pos_host_[e] = single_off;
                                pair_off += (cnt / 2) * 2;
                                single_off += (cnt % 2);
                            }
                        }
                        cur_chunk_start += c_acts;
                        scan_u += c_u;
                    }
                    std::fill_n(expert_curr_pos_host_, cfg_.num_experts, 0);
                    for (int t = 0; t < count; ++t) {
                        for (int k = 0; k < top_k; ++k) {
                            const int i = t * top_k + k;
                            const int e = router_host_ids_[i];
                            if (e >= 0 && e < cfg_.num_experts) {
                                const int seen = expert_curr_pos_host_[e]++;
                                const int cnt = expert_counts_host_[e];
                                int pos;
                                if (s_q4e_4token) {
                                    const int quads_limit = (cnt / 4) * 4;
                                    const int pairs_limit = quads_limit + ((cnt % 4) / 2) * 2;
                                    if (seen < quads_limit) {
                                        pos = expert_quad_pos_host_[e] + seen;
                                    } else if (seen < pairs_limit) {
                                        pos = expert_pair_pos_host_[e] + (seen - quads_limit);
                                    } else {
                                        pos = expert_single_pos_host_[e] + (seen - pairs_limit);
                                    }
                                } else {
                                    const int pairs_limit = (cnt / 2) * 2;
                                    if (seen < pairs_limit) {
                                        pos = expert_pair_pos_host_[e] + seen;
                                    } else {
                                        pos = expert_single_pos_host_[e] + (seen - pairs_limit);
                                    }
                                }
                                expert_map_token_host_[pos] = t;
                                expert_map_rank_host_[pos] = k;
                            }
                        }
                    }
                } else {
                    int cur_act_offset = 0;
                    for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                        const int e = prefill_unique_[u];
                        expert_offsets_host_[e] = cur_act_offset;
                        expert_curr_pos_host_[e] = cur_act_offset;
                        cur_act_offset += expert_counts_host_[e];
                    }
                    for (int t = 0; t < count; ++t) {
                        for (int k = 0; k < top_k; ++k) {
                            const int i = t * top_k + k;
                            const int e = router_host_ids_[i];
                            if (e >= 0 && e < cfg_.num_experts) {
                                const int pos = expert_curr_pos_host_[e]++;
                                expert_map_token_host_[pos] = t;
                                expert_map_rank_host_[pos] = k;
                            }
                        }
                    }
                }

                ctx.queue().memcpy(expert_map_token_dev_, expert_map_token_host_,
                                    static_cast<size_t>(total_acts) * sizeof(int));
                ctx.queue().memcpy(expert_map_rank_dev_, expert_map_rank_host_,
                                    static_cast<size_t>(total_acts) * sizeof(int));
                ctx.queue().memcpy(batch_weights_dev_, router_host_weights_,
                                    static_cast<size_t>(total_acts) * sizeof(float));

                const bf16* x_ptr = static_cast<const bf16*>(buf_.x.data());
                bf16* h_ptr = static_cast<bf16*>(buf_.moe_h.data());
                bf16* y_ptr = static_cast<bf16*>(buf_.y.data());

                sycl::event last_dn_event{};
                size_t u = 0;
                while (u < prefill_unique_.size()) {
                    const size_t chunk_u = std::min<size_t>(static_cast<size_t>(s_demand_microbatch),
                                                            prefill_unique_.size() - u);
                    const int act_start = expert_offsets_host_[prefill_unique_[u]];
                    int act_count = 0;
                    int chunk_quads = 0;
                    int chunk_pairs = 0;
                    for (size_t g = 0; g < chunk_u; ++g) {
                        const int e = prefill_unique_[u + g];
                        const int cnt = expert_counts_host_[e];
                        act_count += cnt;
                        if (s_q4e_weight_reuse) {
                            if (s_q4e_4token) {
                                chunk_quads += cnt / 4;
                                chunk_pairs += (cnt % 4) / 2;
                            } else {
                                chunk_pairs += cnt / 2;
                            }
                        }
                    }

                    const auto prof_group_acq_t0 = prof_clock::now();
                    sycl::event group_dma_event = expert_cache_.acquire(
                        L, prefill_unique_.data() + u, static_cast<int>(chunk_u),
                        prefill_unique_slots_.data() + u);
                    prof_acquire_ms += prof_ms(prof_group_acq_t0, prof_clock::now());

                    for (size_t g = 0; g < chunk_u; ++g) {
                        const int e = prefill_unique_[u + g];
                        const int slot = prefill_unique_slots_[u + g];
                        prefill_expert_slot_[e] = slot;
                        const ops::MoeExpertSlotDev dev_s = expert_cache_.get_slot_dev(slot);
                        if (s_q4e_weight_reuse) {
                            if (s_q4e_4token) {
                                const int q_start = expert_quad_pos_host_[e];
                                const int q_cnt = (expert_counts_host_[e] / 4) * 4;
                                for (int p = q_start; p < q_start + q_cnt; ++p) {
                                    expert_map_slot_host_[p] = dev_s;
                                }
                                const int p_start = expert_pair_pos_host_[e];
                                const int p_cnt = ((expert_counts_host_[e] % 4) / 2) * 2;
                                for (int p = p_start; p < p_start + p_cnt; ++p) {
                                    expert_map_slot_host_[p] = dev_s;
                                }
                                if (expert_counts_host_[e] % 2 != 0) {
                                    expert_map_slot_host_[expert_single_pos_host_[e]] = dev_s;
                                }
                            } else {
                                const int p_start = expert_pair_pos_host_[e];
                                const int p_cnt = (expert_counts_host_[e] / 2) * 2;
                                for (int p = p_start; p < p_start + p_cnt; ++p) {
                                    expert_map_slot_host_[p] = dev_s;
                                }
                                if (expert_counts_host_[e] % 2 != 0) {
                                    expert_map_slot_host_[expert_single_pos_host_[e]] = dev_s;
                                }
                            }
                        } else {
                            const int e_start = expert_offsets_host_[e];
                            const int e_cnt = expert_counts_host_[e];
                            for (int p = e_start; p < e_start + e_cnt; ++p) {
                                expert_map_slot_host_[p] = dev_s;
                            }
                        }
                    }

                    if (act_count > 0) {
                        ctx.queue().memcpy(expert_map_slot_dev_ + act_start,
                                           expert_map_slot_host_ + act_start,
                                           static_cast<size_t>(act_count) * sizeof(ops::MoeExpertSlotDev));

                        sycl::event gu_event = ops::moe_expert_major_prefill_gate_up(
                            ctx, expert_map_slot_dev_ + act_start,
                            expert_map_token_dev_ + act_start,
                            expert_map_rank_dev_ + act_start,
                            nullptr, x_ptr, h_ptr, act_count, top_k,
                            hidden_size_, cfg_.moe_intermediate_size, 64,
                            &group_dma_event, expert_cache_.uses_nested_codes(),
                            s_q4e_weight_reuse ? chunk_pairs : 0,
                            (s_q4e_weight_reuse && s_q4e_4token) ? chunk_quads : 0);

                        sycl::event dn_event = ops::moe_expert_major_prefill_down(
                            ctx, expert_map_slot_dev_ + act_start,
                            expert_map_token_dev_ + act_start,
                            expert_map_rank_dev_ + act_start,
                            nullptr, h_ptr, static_cast<float*>(buf_.moe_down_acc.data()),
                            act_count, top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                            &gu_event, expert_cache_.uses_nested_codes(),
                            s_q4e_weight_reuse ? chunk_pairs : 0,
                            (s_q4e_weight_reuse && s_q4e_4token) ? chunk_quads : 0);

                        expert_cache_.record_consumer_reader_fences(
                            prefill_unique_slots_.data() + u, static_cast<int>(chunk_u), dn_event);
                        last_dn_event = dn_event;
                    }
                    u += chunk_u;
                }

                ops::moe_ordered_accumulate_down_rows(
                    ctx, static_cast<const float*>(buf_.moe_down_acc.data()), batch_weights_dev_,
                    y_ptr, count, top_k, hidden_size_,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) : nullptr,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) : nullptr,
                    &last_dn_event);

                static const bool s_prefill_demote = aila::env::read_flag("AILA_Q4E_PREFILL_DEMOTE", true);
                const size_t keep = static_cast<size_t>(expert_cache_.num_slots() / cfg_.num_hidden_layers);
                if (s_prefill_demote && prefill_unique_.size() > keep) {
                    stable_sort_with_scratch(
                        prefill_unique_.data(), prefill_unique_.size(), sort_scratch_.data(),
                        [&](int a, int b) { return prefill_expert_uses_[a] > prefill_expert_uses_[b]; });
                    expert_cache_.demote(L, prefill_unique_.data() + keep,
                                         static_cast<int>(prefill_unique_.size() - keep));
                }
                if (s_prefetch && predict_next && count > 1) {
                    ensure_prediction_ready();
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
                    expert_cache_.prefetch_async(L + 1, prefill_pred_unique_.data(),
                                                 static_cast<int>(prefill_pred_unique_.size()), prefetch_count);
                }
            } else {
                const int total_acts = count * top_k;
                std::fill_n(expert_counts_host_, cfg_.num_experts, 0);
                for (int i = 0; i < total_acts; ++i) {
                    const int e = router_host_ids_[i];
                    if (e >= 0 && e < cfg_.num_experts) {
                        expert_counts_host_[e]++;
                    }
                }
                static const bool s_q4e_weight_reuse = aila::env::read_flag("AILA_Q4E_WEIGHT_REUSE", true);
                static const bool s_q4e_4token = aila::env::read_flag("AILA_Q4E_4TOKEN_REUSE", true);
                int total_quads = 0;
                int total_pairs = 0;
                if (s_q4e_weight_reuse) {
                    if (s_q4e_4token) {
                        for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                            const int e = prefill_unique_[u];
                            const int cnt = expert_counts_host_[e];
                            total_quads += cnt / 4;
                            total_pairs += (cnt % 4) / 2;
                        }
                        int quad_off = 0;
                        int pair_off = 4 * total_quads;
                        int single_off = 4 * total_quads + 2 * total_pairs;
                        for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                            const int e = prefill_unique_[u];
                            const int cnt = expert_counts_host_[e];
                            expert_quad_pos_host_[e] = quad_off;
                            expert_pair_pos_host_[e] = pair_off;
                            expert_single_pos_host_[e] = single_off;
                            quad_off += (cnt / 4) * 4;
                            pair_off += ((cnt % 4) / 2) * 2;
                            single_off += (cnt % 2);
                        }
                        std::fill_n(expert_curr_pos_host_, cfg_.num_experts, 0);
                        for (int t = 0; t < count; ++t) {
                            for (int k = 0; k < top_k; ++k) {
                                const int i = t * top_k + k;
                                const int e = router_host_ids_[i];
                                if (e >= 0 && e < cfg_.num_experts) {
                                    const int seen = expert_curr_pos_host_[e]++;
                                    const int cnt = expert_counts_host_[e];
                                    const int quads_limit = (cnt / 4) * 4;
                                    const int pairs_limit = quads_limit + ((cnt % 4) / 2) * 2;
                                    int pos;
                                    if (seen < quads_limit) {
                                        pos = expert_quad_pos_host_[e] + seen;
                                    } else if (seen < pairs_limit) {
                                        pos = expert_pair_pos_host_[e] + (seen - quads_limit);
                                    } else {
                                        pos = expert_single_pos_host_[e] + (seen - pairs_limit);
                                    }
                                    expert_map_token_host_[pos] = t;
                                    expert_map_rank_host_[pos] = k;
                                    expert_map_slot_host_[pos] = expert_cache_.get_slot_dev(prefill_expert_slot_[e]);
                                }
                            }
                        }
                    } else {
                        for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                            const int e = prefill_unique_[u];
                            total_pairs += expert_counts_host_[e] / 2;
                        }
                        int pair_off = 0;
                        int single_off = 2 * total_pairs;
                        for (size_t u = 0; u < prefill_unique_.size(); ++u) {
                            const int e = prefill_unique_[u];
                            expert_pair_pos_host_[e] = pair_off;
                            expert_single_pos_host_[e] = single_off;
                            pair_off += (expert_counts_host_[e] / 2) * 2;
                            single_off += (expert_counts_host_[e] % 2);
                        }
                        std::fill_n(expert_curr_pos_host_, cfg_.num_experts, 0);
                        for (int t = 0; t < count; ++t) {
                            for (int k = 0; k < top_k; ++k) {
                                const int i = t * top_k + k;
                                const int e = router_host_ids_[i];
                                if (e >= 0 && e < cfg_.num_experts) {
                                    const int seen = expert_curr_pos_host_[e]++;
                                    const int pairs_limit = (expert_counts_host_[e] / 2) * 2;
                                    int pos;
                                    if (seen < pairs_limit) {
                                        pos = expert_pair_pos_host_[e] + seen;
                                    } else {
                                        pos = expert_single_pos_host_[e] + (seen - pairs_limit);
                                    }
                                    expert_map_token_host_[pos] = t;
                                    expert_map_rank_host_[pos] = k;
                                    expert_map_slot_host_[pos] = expert_cache_.get_slot_dev(prefill_expert_slot_[e]);
                                }
                            }
                        }
                    }
                } else {
                    expert_offsets_host_[0] = 0;
                    for (int e = 0; e < cfg_.num_experts; ++e) {
                        expert_offsets_host_[e + 1] = expert_offsets_host_[e] + expert_counts_host_[e];
                        expert_curr_pos_host_[e] = expert_offsets_host_[e];
                    }
                    for (int t = 0; t < count; ++t) {
                        for (int k = 0; k < top_k; ++k) {
                            const int i = t * top_k + k;
                            const int e = router_host_ids_[i];
                            if (e >= 0 && e < cfg_.num_experts) {
                                const int pos = expert_curr_pos_host_[e]++;
                                expert_map_token_host_[pos] = t;
                                expert_map_rank_host_[pos] = k;
                                expert_map_slot_host_[pos] = expert_cache_.get_slot_dev(prefill_expert_slot_[e]);
                            }
                        }
                    }
                }

                ctx.queue().memcpy(expert_map_slot_dev_, expert_map_slot_host_,
                                    static_cast<size_t>(total_acts) * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(expert_map_token_dev_, expert_map_token_host_,
                                    static_cast<size_t>(total_acts) * sizeof(int));
                ctx.queue().memcpy(expert_map_rank_dev_, expert_map_rank_host_,
                                    static_cast<size_t>(total_acts) * sizeof(int));
                ctx.queue().memcpy(batch_weights_dev_, router_host_weights_,
                                    static_cast<size_t>(total_acts) * sizeof(float));

                const bf16* x_ptr = static_cast<const bf16*>(buf_.x.data());
                bf16* h_ptr = static_cast<bf16*>(buf_.moe_h.data());
                bf16* y_ptr = static_cast<bf16*>(buf_.y.data());

                sycl::event gu_event = ops::moe_expert_major_prefill_gate_up(
                    ctx, expert_map_slot_dev_, expert_map_token_dev_, expert_map_rank_dev_,
                    nullptr, x_ptr, h_ptr, total_acts, top_k,
                    hidden_size_, cfg_.moe_intermediate_size, 64,
                    &batch_dma_event, expert_cache_.uses_nested_codes(),
                    s_q4e_weight_reuse ? total_pairs : 0,
                    (s_q4e_weight_reuse && s_q4e_4token) ? total_quads : 0);

                sycl::event dn_event = ops::moe_expert_major_prefill_down(
                    ctx, expert_map_slot_dev_, expert_map_token_dev_, expert_map_rank_dev_,
                    nullptr, h_ptr, static_cast<float*>(buf_.moe_down_acc.data()),
                    total_acts, top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                    &gu_event, expert_cache_.uses_nested_codes(),
                    s_q4e_weight_reuse ? total_pairs : 0,
                    (s_q4e_weight_reuse && s_q4e_4token) ? total_quads : 0);

                ops::moe_ordered_accumulate_down_rows(
                    ctx, static_cast<const float*>(buf_.moe_down_acc.data()), batch_weights_dev_,
                    y_ptr, count, top_k, hidden_size_,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) : nullptr,
                    s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) : nullptr,
                    &dn_event);

                expert_cache_.record_consumer_reader_fences(
                    prefill_unique_slots_.data(), static_cast<int>(prefill_unique_slots_.size()), dn_event);
            }
        } else if (batch_acquire && s_prefill_moe_rows && count > 1) {
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

            sycl::event dn_event = ops::moe_prefill_down_rows(
                ctx, batch_slots_dev_, batch_weights_dev_, nullptr, h_ptr, y_ptr,
                count, top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) : nullptr,
                s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) : nullptr,
                nullptr, expert_cache_.uses_nested_codes());

            expert_cache_.record_consumer_reader_fences(
                prefill_unique_slots_.data(), static_cast<int>(prefill_unique_slots_.size()), dn_event);
        } else {
            static const bool s_ready_first = aila::env::read_flag("AILA_Q4E_READY_FIRST", false);
            bool hetero_executed = false;
            if ((s_cpu_expert_dispatch || s_ready_first) && count <= 4 && !batch_acquire) {
                auto& plan = expert_executor_.plan_layer(L, router_host_ids_, router_host_weights_, count, top_k, s_cpu_expert_dispatch);
                const bool mixed_gpu = plan.ready_gpu_tasks > 0 && plan.pending_gpu_tasks > 0;
                if (plan.cpu_tasks > 0 || (s_ready_first && mixed_gpu)) {
                    const auto acquire_ns_before = expert_executor_.stats().gpu_acquire_ns;
                    const bool defer_acquire = s_ready_first || expert_executor_.cpu_overlap_enabled();
                    if (!defer_acquire) expert_executor_.ensure_gpu_pending(plan);
                    aila::ExpertPrefetchRequest next_prefetch;
                    if (s_prefetch && predict_next && count == 1) {
                        ensure_prediction_ready();
                        if (defer_acquire) next_prefetch = {L + 1, pred_ids.data(), top_k, s_prefetch_topn};
                        else expert_cache_.prefetch_async(L + 1, pred_ids.data(), top_k, s_prefetch_topn);
                    }
                    const bf16* x_ptr = static_cast<const bf16*>(buf_.x.data());
                    bf16* h_ptr = static_cast<bf16*>(buf_.moe_h.data());
                    bf16* y_ptr = static_cast<bf16*>(buf_.y.data());
                    const bf16* sh_out = s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) : nullptr;
                    const bf16* sh_gate = s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) : nullptr;

                    sycl::event hetero_ev = expert_executor_.execute_hetero_decode(
                        ctx, plan, x_ptr, h_ptr, router_host_weights_,
                        y_ptr, count, top_k, sh_out, sh_gate, expert_cache_.uses_nested_codes(), defer_acquire,
                        next_prefetch.expert_ids ? &next_prefetch : nullptr);
                    prof_acquire_ms += (expert_executor_.stats().gpu_acquire_ns - acquire_ns_before) * 1e-6;

                    // The in-order queue carries merge -> residual -> next route.
                    // Slot readers already have fences; ready-first needs no host drain here.
                    if (!defer_acquire) fallback_last_down_event = hetero_ev;
                    hetero_executed = true;
                }
            }

            if (!hetero_executed) {
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
                            ensure_prediction_ready();
                            expert_cache_.prefetch_async(L + 1, pred_ids.data(), top_k, s_prefetch_topn);
                        }
                    }

                ops::MoeExpertSlotDev dev_slots[kMaxTopK];
                expert_cache_.get_slots_dev(slots, top_k, dev_slots);

                const bf16* x_t = static_cast<const bf16*>(buf_.x.data()) + t * hidden_size_;
                bf16* y_t = static_cast<bf16*>(buf_.y.data()) + t * hidden_size_;

                expert_cache_.record_first_consumer_start();
                ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, dev_slots, nullptr, x_t,
                    static_cast<bf16*>(buf_.moe_h.data()),
                    top_k, hidden_size_, cfg_.moe_intermediate_size, 64,
                    dma_event);

                sycl::event down_ev{};
                if (!batch_acquire && count > 1) {
                    fallback_last_down_event = ops::moe_decode_down_weighted_accumulate(
                        ctx, dev_slots, nullptr, t_weights,
                        static_cast<const bf16*>(buf_.moe_h.data()),
                        nullptr, y_t,
                        top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                        s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) +
                                            static_cast<size_t>(t) * hidden_size_ : nullptr,
                        s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) + t : nullptr);
                    down_ev = fallback_last_down_event;
                } else {
                    down_ev = ops::moe_decode_down_weighted_accumulate(
                        ctx, dev_slots, nullptr, t_weights,
                        static_cast<const bf16*>(buf_.moe_h.data()),
                        nullptr, y_t,
                        top_k, cfg_.moe_intermediate_size, hidden_size_, 64,
                        s_fuse_shared ? static_cast<const bf16*>(buf_.shared_out_buf.data()) +
                                            static_cast<size_t>(t) * hidden_size_ : nullptr,
                        s_fuse_shared ? static_cast<const bf16*>(buf_.shared_gate_buf.data()) + t : nullptr);
                }
                expert_cache_.record_consumer_reader_fences(slots, top_k, down_ev);
            }
        }
        }
        if (fallback_last_down_event != sycl::event{}) {
            fallback_last_down_event.wait_and_throw();
            fallback_last_down_event = sycl::event{};
        }
        ensure_prediction_ready();
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

    if (fallback_last_down_event != sycl::event{}) {
        fallback_last_down_event.wait_and_throw();
        fallback_last_down_event = sycl::event{};
    }

    if (options.need_logits) {
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
    }

    expert_cache_.wait_prefetch();
    if (interior) {
        const int position = start_pos + checkpoint_after;
        const int ratio = std::max(1, cfg_.indexer_compress_ratio);
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) {
                // Raw K and KV retain the entire prefix; only valid block counts need rollback.
                interior->layers[i].num_compressed_blocks =
                    std::min(session.layerCaches_[i].num_compressed_blocks, position / ratio);
            }
        }
        ctx.synchronize();
        interior->length = position;
    }
    if (options.snapshot_policy == SnapshotPolicy::ReusablePrefix) {
        CaptureSnapshot(ctx, session, cached_len, count, options.capture_snapshot);
    }
    session.currentLength_ = cached_len;

    if (s_profile) {
        ctx.synchronize();
        const ExpertCacheStats& s1 = expert_cache_.stats();
        const uint64_t hits = s1.hit_count - prof_stats0.hit_count;
        const uint64_t misses = s1.miss_count - prof_stats0.miss_count;
        AILA_LOG_INFO("[Q4E Profile] pos=%d count=%d total=%.2fms router_sync=%.2fms acquire=%.2fms "
                      "(fill=%.2fms dma_wait=%.2fms pf_wait=%.2fms pf=%llu/%llu) hit=%llu/%llu xfer=%.1fMB logits=%d",
                      start_pos, count, prof_ms(prof_t0, prof_clock::now()),
                      prof_router_sync_ms, prof_acquire_ms,
                      (s1.fill_ns - prof_stats0.fill_ns) * 1e-6,
                      (s1.dma_wait_ns - prof_stats0.dma_wait_ns) * 1e-6,
                      (s1.prefetch_wait_ns - prof_stats0.prefetch_wait_ns) * 1e-6,
                      static_cast<unsigned long long>(s1.prefetch_used - prof_stats0.prefetch_used),
                      static_cast<unsigned long long>(s1.prefetch_issued - prof_stats0.prefetch_issued),
                      static_cast<unsigned long long>(hits),
                      static_cast<unsigned long long>(hits + misses),
                      (s1.total_transferred_bytes - prof_stats0.total_transferred_bytes) / (1024.0 * 1024.0),
                      options.need_logits ? 1 : 0);
        if (count > 1) {
            auto dstats = Bnb4BitLinear::get_prefill_dispatch_stats();
            AILA_LOG_INFO("[Q4E Prefill Dispatch] linear calls: fused_gemm=%llu cached=%llu on_demand=%llu sync_waits=%llu direct_io=%llu",
                          static_cast<unsigned long long>(dstats.fused_gemm),
                          static_cast<unsigned long long>(dstats.cached_weight),
                          static_cast<unsigned long long>(dstats.on_demand_dequant),
                          static_cast<unsigned long long>(dstats.sync_wait),
                          static_cast<unsigned long long>(dstats.direct_io));
        }
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
            if (expert_executor_.stats().layers_hetero_executed > 0) {
                const auto& hs = expert_executor_.stats();
                AILA_LOG_INFO("[Q4E Hetero] layers=%llu cpu_tasks=%llu (overlapped=%llu, stragglers=%llu) gpu_tasks=%llu cpu_compute=%.2f ms straggler_interval=%.2f ms",
                              static_cast<unsigned long long>(hs.layers_hetero_executed),
                              static_cast<unsigned long long>(hs.cpu_tasks_dispatched),
                              static_cast<unsigned long long>(hs.cpu_overlapped_count),
                              static_cast<unsigned long long>(hs.cpu_straggler_count),
                              static_cast<unsigned long long>(hs.gpu_tasks_dispatched),
                              static_cast<double>(hs.cpu_compute_ns) / 1e6,
                              static_cast<double>(hs.cpu_straggler_interval_ns) / 1e6);
            }
        }
        if (expert_executor_.stats().ready_first_layers > 0 || expert_executor_.stats().cpu_tasks_dispatched > 0) {
            AILA_LOG_INFO("[Q4E ReadyFirst] layers=%llu cpu_tasks=%llu cpu_compute=%.2fms cpu_activation_wait=%.2fms",
                          static_cast<unsigned long long>(expert_executor_.stats().ready_first_layers),
                          static_cast<unsigned long long>(expert_executor_.stats().cpu_tasks_dispatched),
                          expert_executor_.stats().cpu_compute_ns * 1e-6,
                          expert_executor_.stats().cpu_activation_wait_ns * 1e-6);
        }
        if (expert_executor_.stats().cpu_async_layers > 0) {
            const auto& hs = expert_executor_.stats();
            AILA_LOG_INFO("[Q4E CPU Overlap] layers=%llu join_wait=%.2fms acquire_overlap=%.2fms max_tasks=%llu host_prepare=%.2fms",
                          static_cast<unsigned long long>(hs.cpu_async_layers),
                          hs.cpu_join_wait_ns * 1e-6, hs.cpu_acquire_overlap_ns * 1e-6,
                          static_cast<unsigned long long>(hs.cpu_max_batch_tasks), hs.cpu_host_prepare_ns * 1e-6);
            AILA_LOG_INFO("[Q4E CPU Dataflow] prepare_batches=%llu prepared_experts=%llu result_copies=%llu result_bytes=%llu result_submit=%.2fms",
                          static_cast<unsigned long long>(hs.cpu_host_prepare_batches),
                          static_cast<unsigned long long>(hs.cpu_host_prepared_experts),
                          static_cast<unsigned long long>(hs.cpu_result_copies),
                          static_cast<unsigned long long>(hs.cpu_result_bytes), hs.cpu_result_submit_ns * 1e-6);
        }
    }
    static const bool s_log_dispatch = aila::env::read_flag("AILA_Q4E_LOG_DISPATCH", false);
    if (s_log_dispatch && count > 1) {
        const auto prefill_stats_t1 = Bnb4BitLinear::get_prefill_dispatch_stats();
        const uint64_t d_fused = prefill_stats_t1.fused_gemm - prefill_stats_t0.fused_gemm;
        const uint64_t d_dequant = prefill_stats_t1.on_demand_dequant - prefill_stats_t0.on_demand_dequant;
        const uint64_t d_sync = prefill_stats_t1.sync_wait - prefill_stats_t0.sync_wait;
        AILA_LOG_INFO("[Q4E Prefill] dispatch delta: fused_gemm=+%llu, on_demand_dequant=+%llu, sync_wait=+%llu",
                      static_cast<unsigned long long>(d_fused),
                      static_cast<unsigned long long>(d_dequant),
                      static_cast<unsigned long long>(d_sync));
    }
    return options.need_logits ? lastTokenLogits_ : unusedLogits_;
}

// ============================================================================
// Snapshot & Rollback
// ============================================================================

Qwen4ExpBnb4Backend::StateSnapshot& Qwen4ExpBnb4Backend::ReserveSnapshot(Session& session, int nextLength) {
    auto& snapshots = session.snapshots_;
    auto slot = std::find_if(snapshots.begin(), snapshots.end(),
        [nextLength](const auto& snapshot) { return snapshot.length == nextLength; });
    if (slot == snapshots.end()) {
        slot = std::min_element(snapshots.begin(), snapshots.end(),
            [](const auto& left, const auto& right) { return left.length < right.length; });
    }
    auto& snapshot = *slot;
    snapshot.length = -1;
    return snapshot;
}

void Qwen4ExpBnb4Backend::CaptureSnapshot(
    Context& ctx, Session& session, int nextLength, int tokenCount, bool force) {
    if (session.snapshots_.empty()) return;
    int step = aila::env::read_int("AILA_Q4E_PREFILL_STEP", 64);
    if (step <= 0) step = 64;
    if (!force && tokenCount <= 1 && nextLength % step != 0) return;

    ctx.synchronize();
    auto& snapshot = ReserveSnapshot(session, nextLength);
    try {
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
            }
        }
        if (last_d2h_event != sycl::event{}) {
            last_d2h_event.wait_and_throw();
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto& cache = session.layerCaches_[i];
            if (layers_[i].is_linear) {
                std::copy(cache.host_linear_state.begin(), cache.host_linear_state.end(),
                          snapshot.layers[i].linear_state_host.begin());
                std::copy(cache.host_linear_conv_state.begin(), cache.host_linear_conv_state.end(),
                          snapshot.layers[i].linear_conv_state_host.begin());
                cache.device_state_dirty = false;
                snapshot.layers[i].linear_conv_head = cache.linear_conv_head;
            } else {
                snapshot.layers[i].num_compressed_blocks = cache.num_compressed_blocks;
            }
        }
        ple_.capture_snapshot(ctx, session.ple_state_, snapshot.ple_snapshot);
        snapshot.length = nextLength;
    } catch (...) {
        ctx.synchronize();
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
    const StateSnapshot* selected = nullptr;
    for (const auto& snapshot : session.snapshots_) {
        if (snapshot.length <= newLength && snapshot.length > target) {
            target = snapshot.length;
            selected = &snapshot;
        }
    }
    if (target < 0) {
        Reset(session);
        return 0;
    }
    const auto& snapshot = *selected;
    sycl::event last_h2d_event{};
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& cache = session.layerCaches_[i];
        const auto& snap_layer = snapshot.layers[i];
        if (layers_[i].is_linear) {
            std::copy(snap_layer.linear_state_host.begin(), snap_layer.linear_state_host.end(),
                      cache.host_linear_state.begin());
            std::copy(snap_layer.linear_conv_state_host.begin(), snap_layer.linear_conv_state_host.end(),
                      cache.host_linear_conv_state.begin());
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
        } else {
            cache.num_compressed_blocks = snap_layer.num_compressed_blocks;
        }
    }
    if (last_h2d_event != sycl::event{}) {
        last_h2d_event.wait_and_throw();
    }
    ple_.restore_snapshot(*session.context_, session.ple_state_, snapshot.ple_snapshot);
    session.bump_generation();
    ple_ticket_.valid = false;

    for (auto& entry : session.snapshots_) {
        if (entry.length > target) entry.length = -1;
    }
    session.currentLength_ = target;
    return target;
}

void Qwen4ExpBnb4Backend::Reset(Session& session) {
    ValidateSession(session);
    session.context_->synchronize();
    for (auto& snapshot : session.snapshots_) snapshot.length = -1;
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
        } else {
            cache.num_compressed_blocks = 0;
        }
    }
    ple_.init_state(*session.context_, session.ple_state_);
    session.bump_generation();
    ple_ticket_.valid = false;
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
    auto session = std::unique_ptr<Session>(new Session(ctx, max_seq_len_, sessionOwner_, maxSnapshots));
    ple_.init_state(ctx, session->ple_state_);
    for (auto& snapshot : session->snapshots_) {
        snapshot.layers.resize(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (!layers_[i].is_linear) continue;
            const auto& cache = session->layerCaches_[i];
            snapshot.layers[i].linear_state_host.resize(cache.host_linear_state.size());
            snapshot.layers[i].linear_conv_state_host.resize(cache.host_linear_conv_state.size());
        }
    }
    return session;
}

Tensor& Qwen4ExpBnb4Backend::Forward(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count,
    const ForwardOptions& options) {
    return ForwardImpl(ctx, session, tokenIdsDevice, count, true, options);
}

Tensor& Qwen4ExpBnb4Backend::ForwardLegacyLogits(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count) {
    return ForwardImpl(ctx, session, tokenIdsDevice, count, false, {});
}

Tensor& Qwen4ExpBnb4Backend::forward(
    Context& ctx, const int* token_ids_device, int seq_len) {
    return Forward(ctx, DefaultSession(ctx), token_ids_device, seq_len);
}

bool Qwen4ExpBnb4Backend::supports_interior_checkpoint(int count, int checkpoint_after) const {
    bool explicit_inline = false;
    const bool has_env = aila::env::read_flag_if_present("AILA_Q4E_INLINE_CHECKPOINT", &explicit_inline);
    if (has_env && !explicit_inline) {
        return false;
    }
    const bool allowed_anchor = (has_env && explicit_inline) || (checkpoint_after == 26 || checkpoint_after == 28);
    auto* self = const_cast<Qwen4ExpBnb4Backend*>(this);
    if (!defaultSession_ && context_) {
        self->DefaultSession(*context_);
    }
    return allowed_anchor && defaultSession_ && defaultSession_->snapshots_.size() >= 2 &&
        count <= 64 && checkpoint_after >= 2 && count - checkpoint_after >= 2 &&
        defaultSession_->CurrentLength() + count <= 2052 &&
        defaultSession_->embeddingOverridePositions_.empty() && !defaultSession_->mropePosT_.valid();
}

Tensor& Qwen4ExpBnb4Backend::forward(
    Context& ctx, const int* token_ids_device, int seq_len, const ForwardOptions& options) {
    return Forward(ctx, DefaultSession(ctx), token_ids_device, seq_len, options);
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
