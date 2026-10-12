#include "Qwen4ExpBnb4Backend.hpp"
#include "../memory/SessionMemory.hpp"
#include "../ops/QsaIndexerOps.hpp"
#include "../ops/Qwen4ExpAttentionPair.hpp"
#include "models/Qwen4ExpHotsetPolicy.hpp"
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
#include <optional>
#include <stdexcept>

namespace fs = std::filesystem;
using bf16 = sycl::ext::oneapi::bfloat16;
using aila::StagingBatch;
using aila::Qwen4ExpAdaptiveCache;
using aila::Qwen4ExpHostArena;

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
    // INT8 KV is opt-in and default-off; both switches must be present.
    const bool int8_kv = aila::env::read_flag("AILA_KV_QUANT", false) &&
                         aila::env::kv_quant_format_is_int8();
    if (aila::env::read_flag("AILA_KV_QUANT", false)) {
        AILA_LOG_INFO("[Q4E KV] AILA_KV_QUANT=1 format=%s",
                      int8_kv ? "int8" : "fp8(ignored: Qwen4Exp keeps bf16 KV)");
    }
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
            const auto kv_dtype = int8_kv ? dnnl::memory::data_type::s8
                                          : dnnl::memory::data_type::bf16;
            cache.k = Tensor::allocate(ctx, {2, capacity, 256}, kv_dtype);
            cache.v = Tensor::allocate(ctx, {2, capacity, 256}, kv_dtype);
            if (int8_kv) {
                cache.k_scale = Tensor::allocate(ctx, {2, capacity}, dnnl::memory::data_type::f32);
                cache.v_scale = Tensor::allocate(ctx, {2, capacity}, dnnl::memory::data_type::f32);
            }
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
    expert_executor_.AbortResidentCold();
    free_pinned_host_buffers();
    if (resident_selected_host_ && context_) context_->free_host(resident_selected_host_);
    if (copy_q_) {
        try { copy_q_->wait(); } catch (...) {}
        copy_q_.reset();
    }
    adaptive_cache_.Release();
    host_arena_.Release();
    if (resident_embedding_ && context_) context_->free_host(resident_embedding_);
}

void Qwen4ExpBnb4Backend::PrepareDecodeVerification(Context& ctx) {
    if (context_ != &ctx) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: verification requires the model Context");
    }
    // Called under the execution-domain lock before verification, never in its hot loop.
    // Keep the bundle alive until backend teardown has drained this Context.
    if (!verification_attention_pair_bundle_)
        verification_attention_pair_bundle_.emplace(ops::prepare_q4e_attention_pair(ctx));
    // Multi-row (2..4 token) CPU verifier scratch, allocated once here so the
    // verification window itself performs zero device allocations. Only the
    // arena verifier consumes these; ordinary decode never touches them.
    const int top_k = cfg_.num_experts_per_tok;
    if (!verify_down_.valid())
        verify_down_ = Tensor::allocate(ctx, {4LL * top_k * hidden_size_}, dnnl::memory::data_type::f32);
    if (!verify_weights_.valid())
        verify_weights_ = Tensor::allocate(ctx, {4LL * top_k}, dnnl::memory::data_type::f32);
}

void Qwen4ExpBnb4Backend::free_pinned_host_buffers() {
    if (!context_) return;
    // Buffer growth already checks Context errors before entering here. Teardown
    // must also release raw buffers after a forward fault without rethrowing the
    // latched async error from a destructor. Always drain the actual queue first.
    try { context_->queue().wait_and_throw(); } catch (...) {}
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
    if (expert_tiles_host_) context_->free_host(expert_tiles_host_);
    if (expert_curr_pos_host_) context_->free_host(expert_curr_pos_host_);
    if (expert_eight_pos_host_) context_->free_host(expert_eight_pos_host_);
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
    expert_tiles_host_ = nullptr;
    expert_curr_pos_host_ = nullptr;
    expert_eight_pos_host_ = nullptr;
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
    moe_compact_tile_ = 0; // adaptive prefill uses the existing ordered rank buffer
    if (moe_compact_tile_ && (moe_compact_tile_ != 256 && moe_compact_tile_ != 512 && moe_compact_tile_ != 1024))
        throw std::invalid_argument("AILA_Q4E_MOE_COMPACT_ACTS must be 0, 256, 512 or 1024");
    if (moe_compact_tile_ && (!aila::env::read_flag("AILA_Q4E_EXPERT_MAJOR_PREFILL", true) ||
        aila::env::read_int("AILA_Q4E_DEMAND_MICROBATCH",0) <= 0 ||
        aila::env::read_int("AILA_Q4E_GROUPED_XMX_MIN_ROWS",0) > 0 ||
        aila::env::read_flag("AILA_Q4E_CPU_EXPERT_DISPATCH",false)))
        throw std::invalid_argument("Compact MoE requires GPU ALU expert-major demand microbatches");
    prefill_attention_tile_ = aila::env::read_int("AILA_Q4E_PREFILL_ATTN_TILE", 0);
    if (prefill_attention_tile_ != 0 && prefill_attention_tile_ != 128 &&
        prefill_attention_tile_ != 256 && prefill_attention_tile_ != 512)
        throw std::invalid_argument("AILA_Q4E_PREFILL_ATTN_TILE must be 0, 128, 256 or 512");
    grouped_xmx_supported_ = ctx.queue().get_device().get_info<sycl::info::device::name>().find("Arc(TM) A770") != std::string::npos;
    expert_histogram_enabled_ = aila::env::read_flag("AILA_Q4E_EXPERT_HISTOGRAM", false);
    gdn_chunk_mode_ = std::clamp(aila::env::read_int("AILA_Q4E_GDN_CHUNK_MODE", 0), 0, 2);
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
        static const bool s_q4e_prefill_xmx =
            aila::env::read_flag("AILA_Q4E_PREFILL_XMX", false) ||
            aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
        static const bool s_direct_prefill_io =
            aila::env::read_flag("AILA_Q4E_DIRECT_PREFILL_IO", true);
        linear.set_direct_prefill_io(s_direct_prefill_io);
        if (s_q4e_prefill_xmx) {
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

    static const bool s_lm_head_q8 = aila::env::read_flag("AILA_Q4E_LM_HEAD_Q8", false);
    lm_head_host_ = weights.has_host("lm_head.weight") ? weights.host_view("lm_head.weight") : HostTensorView{};
    const bool use_q8_head = s_lm_head_q8 && hidden_size_ == 2560 && cfg_.vocab_size == 248320 &&
        lm_head_host_.valid() && lm_head_host_.dtype == DT_BF16 && lm_head_host_.shape.size() == 2 &&
        lm_head_host_.shape[0] == cfg_.vocab_size && lm_head_host_.shape[1] == hidden_size_ &&
        ops::bf16_q8_head_supported(ctx, hidden_size_, cfg_.vocab_size);
    Tensor& raw_lm_head = weights.upload("lm_head.weight", ctx);
    if (use_q8_head) {
        if (raw_lm_head.context() != &ctx || raw_lm_head.dtype() != dnnl::memory::data_type::bf16 ||
            raw_lm_head.ndim() != 2 || raw_lm_head.shape(0) != cfg_.vocab_size || raw_lm_head.shape(1) != hidden_size_)
            throw std::runtime_error("Qwen4Exp Q8 head requires raw BF16 [248320,2560] in model Context");
        lm_head_q8_weight_ = Tensor::allocate(ctx, {cfg_.vocab_size, hidden_size_}, dnnl::memory::data_type::u8);
        lm_head_q8_scale_ = Tensor::allocate(ctx, {cfg_.vocab_size, hidden_size_ / 64}, dnnl::memory::data_type::f32);
        lm_head_q8_input_ = Tensor::allocate(ctx, {hidden_size_}, dnnl::memory::data_type::u8);
        lm_head_q8_input_scale_ = Tensor::allocate(ctx, {hidden_size_ / 64}, dnnl::memory::data_type::f32);
        ops::quantize_bf16_weights_q8_64(ctx, raw_lm_head, lm_head_q8_weight_, lm_head_q8_scale_);
        ctx.synchronize(); // Raw BF16 cannot be freed until conversion drains.
        weights.erase("lm_head.weight"); // Keeps host mmap; no transposed BF16 or oneDNN head.
        lmHeadQ8Enabled_ = true;
        AILA_LOG_INFO("[Q4E LM Head Q8 Load] actual_enabled=1 hidden=%d vocab=%d block=64 weight_bytes=%llu scale_bytes=%llu saved_head_bytes=%llu host_original_bf16=1 experimental=1",
            hidden_size_, cfg_.vocab_size,
            static_cast<unsigned long long>(lm_head_q8_weight_.size_bytes()),
            static_cast<unsigned long long>(lm_head_q8_scale_.size_bytes()),
            static_cast<unsigned long long>(size_t(hidden_size_) * cfg_.vocab_size * sizeof(bf16) -
                lm_head_q8_weight_.size_bytes() - lm_head_q8_scale_.size_bytes()));
    } else {
        if (s_lm_head_q8) AILA_LOG_INFO("[Q4E LM Head Q8 Load] requested=1 actual_enabled=0 unsupported_shape_or_context=1");
        lm_head_weight_transposed_ = Tensor::allocate(ctx, {hidden_size_, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
        ops::transpose(ctx, raw_lm_head, lm_head_weight_transposed_);
        ctx.synchronize();
        weights.erase("lm_head.weight");
        lm_head_.init(ctx, lm_head_weight_transposed_, hidden_size_, cfg_.vocab_size, /*preprocessed=*/true, /*prepare_gemv=*/false);
    }

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
    // GPU experts and native RAM experts are complementary owners. Allocate
    // the contiguous embedding before the smaller large-page expert chunks.
    if (embed_host_.valid()) {
        resident_embedding_=ctx.alloc_large_host(embed_host_.bytes);
        std::memcpy(resident_embedding_,embed_host_.ptr,embed_host_.bytes);
        embed_host_.ptr=static_cast<const uint8_t*>(resident_embedding_);
    }
    aila::AdaptiveCacheConfig cache_cfg;
    cache_cfg.num_layers=cfg_.num_hidden_layers;
    cache_cfg.num_experts_per_layer=cfg_.num_experts;
    cache_cfg.hidden_size=hidden_size_;
    cache_cfg.moe_intermediate_size=cfg_.moe_intermediate_size;
    cache_cfg.staging_slots_per_buffer=16;
    const size_t slot_bytes=size_t(3)*hidden_size_*cfg_.moe_intermediate_size*33/64;
    // The 32K session and prefill projections reserve more VRAM than the CLI's
    // 4K smoke. Fit the cache after those reservations, including both banks.
    // The hard cap only guards against pathological budgets; raise it when a
    // smaller KV layout (AILA_KV_QUANT_FORMAT=int8) frees device bytes.
    const int max_gpu_slots = aila::env::read_int("AILA_Q4E_MAX_GPU_SLOTS", 2800);
    cache_cfg.num_gpu_slots=std::min(max_gpu_slots,int(expert_cache_bytes/slot_bytes)-32);
    if (cache_cfg.num_gpu_slots<=0) {
        if (error_message) *error_message="Qwen4Exp resident cache exceeds the admitted VRAM budget";
        return false;
    }
    const auto hotset_profile_path=aila::env::read_string("AILA_Q4E_HOTSET_PROFILE","");
    const bool profile_hotset=!hotset_profile_path.empty();
    std::vector<std::pair<int,int>> initial_ranked_profile;
    if (profile_hotset) {
        // Build the same source-byte identity used by the legacy cache. The
        // adaptive arena's expert payload is always compact nested NF4.
        std::vector<aila::ExpertLayerSource> profile_sources(cfg_.num_hidden_layers);
        for (int layer=0;layer<cfg_.num_hidden_layers;++layer) {
            const std::string prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.";
            auto& source=profile_sources[layer];
            std::string detail;
            const auto gate_up_base=prefix+"gate_up_proj",down_base=prefix+"down_proj";
            const auto gate_up_name=weights.has_host(gate_up_base)?gate_up_base:gate_up_base+".weight";
            const auto down_name=weights.has_host(down_base)?down_base:down_base+".weight";
            if (!LoadBnb4BitHostRef(weights,gate_up_name,source.gate_up,&detail,hidden_size_) ||
                !LoadBnb4BitHostRef(weights,down_name,source.down,&detail,cfg_.moe_intermediate_size)) {
                if (error_message) *error_message="Cannot validate AILA_Q4E_HOTSET_PROFILE identity: "+detail;
                return false;
            }
            source.valid=true;
        }
        const std::string identity=Qwen4ExpHotsetPolicy::MakeIdentity(profile_sources,
            cfg_.num_hidden_layers,cfg_.num_experts,hidden_size_,cfg_.moe_intermediate_size,64,true);
        Qwen4ExpHotsetPolicy policy;
        policy.Init(cfg_.num_hidden_layers,cfg_.num_experts);
        try {
            policy.Load(hotset_profile_path,identity);
        } catch (const std::exception& e) {
            if (error_message) *error_message=e.what();
            return false;
        }
        const size_t resident_rank_count=std::min<size_t>(
            static_cast<size_t>(cache_cfg.num_gpu_slots),policy.ranking.size());
        initial_ranked_profile.reserve(resident_rank_count);
        for (size_t i=0;i<resident_rank_count;++i) {
            const int key=policy.ranking[i];
            initial_ranked_profile.emplace_back(key/cfg_.num_experts,key%cfg_.num_experts);
        }
        AILA_LOG_INFO("[Q4E Adaptive Residency] profile=%s identity=%s ranked_keys=%zu resident_slots=%d",
            hotset_profile_path.c_str(),identity.c_str(),initial_ranked_profile.size(),cache_cfg.num_gpu_slots);
    }
    if (!adaptive_cache_.Allocate(ctx,cache_cfg,error_message)) return false;
    auto gpu_owners=adaptive_cache_.InitializeResidency(
        profile_hotset ? &initial_ranked_profile : nullptr);
    aila::HostArenaConfig arena_cfg;
    arena_cfg.num_layers=cfg_.num_hidden_layers; arena_cfg.num_experts_per_layer=cfg_.num_experts;
    arena_cfg.hidden_size=hidden_size_; arena_cfg.moe_intermediate_size=cfg_.moe_intermediate_size;
    if (!host_arena_.Allocate(ctx,arena_cfg,error_message,gpu_owners)) return false;
    AILA_LOG_INFO("[Qwen4Exp] Loading native RAM complement: %zu experts, %.2f GiB; GPU-only experts=%d",
        host_arena_.ResidentExperts(),host_arena_.TotalAllocatedBytes()/double(1ULL<<30),adaptive_cache_.NumGpuSlots());
    if (!host_arena_.LoadFromModelWeights(weights,error_message)) return false;
    adaptive_cache_.FinishInitialLoad(ctx.queue());
    copy_q_=std::make_unique<sycl::queue>(ctx.queue().get_context(),ctx.queue().get_device(),
        sycl::property_list{sycl::property::queue::in_order()});
    prefill_expert_uses_.assign(cfg_.num_experts,0);
    prefill_unique_.reserve(cfg_.num_experts);
    adaptive_prefill_slots_.resize(cfg_.num_experts);
    AILA_LOG_INFO("[Qwen4Exp] Adaptive residency ready: expert_mmap_fallback=0 small_host_staging_mib=%.1f",
        adaptive_cache_.TotalStagingBytes()/double(1ULL<<20));

    ensure_runtime_buffers(ctx, 1);
    static const int s_resident_decode = aila::env::read_int("AILA_Q4E_RESIDENT_DECODE", 0);
    cold_mode_ = aila::env::read_int("AILA_Q4E_COLD_MODE",0);
    if(profile_hotset && !cold_mode_)cold_mode_=1;
    resident_prefill_ = aila::env::read_flag("AILA_Q4E_RESIDENT_PREFILL",false);
    if(cold_mode_<0 || cold_mode_>2)throw std::invalid_argument("Cold mode must be 0/1/2");
    resident_decode_mode_ = cold_mode_ ? 2 : s_resident_decode;
    if (resident_decode_mode_ < 0 || resident_decode_mode_ > 2 ||
        (resident_decode_mode_ && aila::env::read_flag("AILA_Q4E_CPU_EXPERT_DISPATCH", false))) {
        if (error_message) *error_message = "Resident decode currently supports GPU-only mode 0/1/2";
        return false;
    }
    if (resident_decode_mode_ || use_adaptive_pipeline_) {
        resident_down_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, hidden_size_}, dnnl::memory::data_type::f32);
        resident_weights_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok}, dnnl::memory::data_type::f32);
        if (resident_decode_mode_ == 2 && !use_adaptive_pipeline_) {
            expert_cache_.EnableResidentDeviceTables();
            resident_route_ids_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok}, dnnl::memory::data_type::s32);
            resident_selected_ = Tensor::allocate(ctx, {cfg_.num_experts_per_tok}, dnnl::memory::data_type::s32);
            resident_selected_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts_per_tok * sizeof(int)));
        }
        AILA_LOG_INFO("[Q4E ResidentDecode] mode=%d adaptive_pipeline=%d", resident_decode_mode_, int(use_adaptive_pipeline_));
    }
    if (!use_adaptive_pipeline_) {
        expert_executor_.init(ctx, expert_cache_, max_seq_len_, cfg_.num_experts_per_tok, hidden_size_, cfg_.moe_intermediate_size);
        expert_cache_.ConfigureHotset();
        static const bool large_page_resident = aila::env::read_flag("AILA_Q4E_LARGE_PAGE_RESIDENT", false);
        if (large_page_resident) expert_cache_.InitializeLargePageResidency(&embed_host_);
        expert_executor_.PrepareAsyncResidentSources(cfg_.num_hidden_layers, cfg_.num_experts);
        // Batched prefill acquire needs every unique expert of a layer resident at once.
        prefill_expert_slot_.assign(cfg_.num_experts, -1);
        prefill_expert_uses_.assign(cfg_.num_experts, 0);
        prefill_unique_.reserve(cfg_.num_experts);
        prefill_unique_slots_.reserve(cfg_.num_experts);
        if (expert_cache_.num_slots() < cfg_.num_experts + cfg_.num_experts_per_tok) {
            AILA_LOG_WARN("[Qwen4Exp] Expert cache has %d slots (< %d): prefill acquires experts per token",
                          expert_cache_.num_slots(), cfg_.num_experts + cfg_.num_experts_per_tok);
        }
        prefill_pred_uses_.assign(cfg_.num_experts, 0);
        prefill_pred_unique_.reserve(cfg_.num_experts);
        sort_scratch_.resize(cfg_.num_experts);
    } else {
        // Reuse the existing CPU worker and tiny decode banks. The retired
        // cache supplies no weights/slots; StartArenaCold reads native owners.
        expert_executor_.init(ctx,expert_cache_,1,cfg_.num_experts_per_tok,hidden_size_,cfg_.moe_intermediate_size);
    }

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
    const bool s_expert_major = true;
    const bool s_moe_rows = aila::env::read_flag("AILA_Q4E_PREFILL_MOE_ROWS", false);
    const bool s_pred_ids_only = aila::env::read_flag("AILA_Q4E_PRED_IDS_ONLY", true);
    const bool s_dense_xmx = aila::env::read_flag("AILA_Q4E_PREFILL_XMX", false) ||
                             aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
    const bool s_direct_io = aila::env::read_flag("AILA_Q4E_DIRECT_PREFILL_IO", true);
    const char* prefill_mode = s_expert_major ? "S4 (expert-major W4/G3)" : (s_moe_rows ? "S1 (batched-rows)" : "baseline (token-by-token)");
    AILA_LOG_INFO("[Qwen4Exp] Prefill MoE mode: %s, Router pred: %s",
                  prefill_mode, s_pred_ids_only ? "ids-only" : "full-softmax");
    AILA_LOG_INFO("[Qwen4Exp] Prefill Dense Linear: %s (direct_io=%s)",
                  s_dense_xmx ? "XMX Joint Matrix" : "oneDNN on-demand",
                  s_direct_io ? "enabled" : "disabled");
    AILA_LOG_INFO("[Qwen4Exp] Backend successfully loaded (48 layers, max_seq_len=%d)", max_seq_len_);
    AILA_LOG_INFO("[Q4E ExecutionConfig] preset=%s chunk=%d native_ram_mib=%.1f slots=%d dense_xmx=%d direct_io=%d phase_scratch=%d staging_experts=%d checkpoint=%s attn_tile=%d compact_acts=%d",
        aila::env::read_string("AILA_Q4E_ACTIVE_PRESET", "none").c_str(),
        aila::env::read_int("AILA_Q35_PREFILL_CHUNK", 512),
        (host_arena_.TotalAllocatedBytes() + (resident_embedding_ ? aila::memory::LargePageAllocationSize(embed_host_.bytes) : 0)) / double(1ULL<<20), adaptive_cache_.NumGpuSlots(),
        int(s_dense_xmx), int(s_direct_io), int(aila::env::read_flag("AILA_Q4E_PHASE_SCRATCH", false)),
        adaptive_cache_.StagingSlotsPerBuffer(),
        aila::env::read_string("AILA_Q4E_INLINE_CHECKPOINT", "auto-30/26").c_str(), prefill_attention_tile_, moe_compact_tile_);
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
    // Growth is outside decode's steady-state path. Drain readers before any
    // old tensor/view backing store is replaced (including oneDNN consumers).
    ctx.synchronize();
    static const bool s_phase_scratch = aila::env::read_flag("AILA_Q4E_PHASE_SCRATCH", false);
    const bool s_expert_major = true;
    constexpr int kDensePrefillLimit = 2052;
    const int attn_cap = prefill_attention_tile_ ? std::min(cap, prefill_attention_tile_) : cap;
    const int dense_cap = std::min(attn_cap, kDensePrefillLimit);
    const int dense_kv_cap = std::min(max_seq_len_, kDensePrefillLimit);
    const int down_rows = moe_compact_tile_ ? std::max(moe_compact_tile_, std::min(cap,4)*cfg_.num_experts_per_tok)
                                             : cap*cfg_.num_experts_per_tok;
    if (moe_compact_tile_) {
        buf_.moe_window_sum = Tensor::allocate(ctx,{int64_t(cap)*hidden_size_*int64_t(sizeof(int64_t))},dnnl::memory::data_type::u8);
        if(!buf_.moe_compact_invalid.valid())buf_.moe_compact_invalid=Tensor::allocate(ctx,{1},dnnl::memory::data_type::s32);
        buf_.moe_inverse_map = Tensor::allocate(ctx,{cap,cfg_.num_experts_per_tok},dnnl::memory::data_type::s32);
    } else { buf_.moe_window_sum=Tensor{};buf_.moe_inverse_map=Tensor{}; }
    if (s_phase_scratch) {
        // All five lifetimes are ordered on ctx.queue(): QKV -> split/prepare
        // -> dense attention, or GDN -> o_proj, then MoE -> accumulation.
        // scores/incr_scores are exclusive start_pos==0/!=0 branches. PLE,
        // prediction, staging, persistent state and attn_out never alias here.
        const size_t bytes[] = {size_t(attn_cap)*full_fused_qkv_dim_*sizeof(bf16),
            size_t(attn_cap)*linear_all_dim_*sizeof(bf16),
            size_t(full_q_heads_)*dense_cap*dense_cap*sizeof(float),
            size_t(full_q_heads_)*dense_cap*dense_kv_cap*sizeof(float),
            s_expert_major ? size_t(down_rows)*hidden_size_*sizeof(float) : 0};
        size_t peak=0,total=0;for(size_t n:bytes){peak=std::max(peak,n);total+=n;}
        buf_.phase_scratch=Tensor::allocate(ctx,{int64_t(peak)},dnnl::memory::data_type::u8);
        AILA_LOG_INFO("[Q4E PhaseScratch] cap=%d bytes=%zu separate_bytes=%zu saved_bytes=%zu",
            cap,peak,total,total-peak);
    }
    auto phase = [&](const std::vector<int64_t>& shape,dnnl::memory::data_type type) {
        return s_phase_scratch ? Tensor::view(ctx,buf_.phase_scratch.data(),shape,type)
                               : Tensor::allocate(ctx,shape,type);
    };

    buf_.hidden4 = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_N = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_down = Tensor::allocate(ctx, {cap, 320}, dnnl::memory::data_type::bf16);
    buf_.hc_mix_up = Tensor::allocate(ctx, {cap, 10240}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_buf = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::bf16);
    buf_.hc_inject_f32 = Tensor::allocate(ctx, {cap, 4}, dnnl::memory::data_type::f32);
    buf_.x = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);
    buf_.y = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    // Full Attention buffers
    buf_.full_qkv = phase({attn_cap, full_fused_qkv_dim_}, dnnl::memory::data_type::bf16);
    buf_.q = Tensor::allocate(ctx, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.z = Tensor::allocate(ctx, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
    buf_.k = Tensor::allocate(ctx, {attn_cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.v = Tensor::allocate(ctx, {attn_cap, full_kv_dim_}, dnnl::memory::data_type::bf16);
    buf_.attn_out = Tensor::allocate(ctx, {attn_cap, full_q_dim_}, dnnl::memory::data_type::bf16);
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
    buf_.scores = phase({full_q_heads_, dense_cap, dense_cap}, dnnl::memory::data_type::f32);
    buf_.incr_scores = phase({full_q_heads_, dense_cap, dense_kv_cap}, dnnl::memory::data_type::f32);

    // QSA Indexer buffers (M2, A3, A4)
    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
    const int indexer_q_dim = cfg_.indexer_n_heads * cfg_.indexer_head_dim;
    buf_.indexer_qk = Tensor::allocate(ctx, {attn_cap, indexer_out_dim}, dnnl::memory::data_type::bf16);
    buf_.indexer_q = Tensor::allocate(ctx, {attn_cap, indexer_q_dim}, dnnl::memory::data_type::bf16);
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
    buf_.linear_all = phase({attn_cap, linear_all_dim_}, dnnl::memory::data_type::bf16);
    static const bool s_attention_nf4_q8 = aila::env::read_flag("AILA_Q4E_ATTN_NF4_Q8", false);
    attentionNf4Q8Enabled_ = s_attention_nf4_q8;
    if (attentionNf4Q8Enabled_) {
        const int projection_input = std::max({hidden_size_, linear_kv_dim_, full_q_dim_});
        const int quantized_capacity = ((projection_input + 63) / 64) * 64;
        if (!buf_.attn_xq.valid() || buf_.attn_xq.numel() < quantized_capacity) {
            // Shared by successive decode projections on the in-order queue;
            // retain these fixed-size banks through prefill growth and trimming.
            buf_.attn_xq = Tensor::allocate(ctx, {quantized_capacity}, dnnl::memory::data_type::u8);
            buf_.attn_xs = Tensor::allocate(ctx, {quantized_capacity / 64}, dnnl::memory::data_type::f32);
        }
    }
    if (gdn_chunk_mode_ > 0 && grouped_xmx_supported_)
        gdn_chunk_scratch_.Reserve(ctx, attn_cap, linear_kv_heads_);

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
    static const bool s_gpu_nf4_q8 = aila::env::read_flag("AILA_Q4E_GPU_NF4_Q8", false);
    moeFrontFusedGpu_ = ctx.queue().get_device().is_gpu();
    gpuNf4Q8Supported_ = s_gpu_nf4_q8 && hidden_size_ == 2560 && cfg_.moe_intermediate_size == 640 &&
        cfg_.num_experts_per_tok >= 1 && cfg_.num_experts_per_tok <= 16 &&
        ctx.queue().get_device().get_info<sycl::info::device::max_work_group_size>() >= 1024;
    if (gpuNf4Q8Supported_ && !buf_.moe_x_q8.valid()) {
        // Independent of the prefill capacity: these tiny buffers persist through
        // prompt growth/trim and introduce no steady-state decode allocations.
        buf_.moe_x_q8 = Tensor::allocate(ctx, {hidden_size_}, dnnl::memory::data_type::u8);
        buf_.moe_x_q8_scale = Tensor::allocate(ctx, {hidden_size_ / 64}, dnnl::memory::data_type::f32);
        buf_.moe_h_q8 = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, cfg_.moe_intermediate_size}, dnnl::memory::data_type::u8);
        buf_.moe_h_q8_scale = Tensor::allocate(ctx, {cfg_.num_experts_per_tok, cfg_.moe_intermediate_size / 64}, dnnl::memory::data_type::f32);
    }
    buf_.shared_gate_buf = Tensor::allocate(ctx, {cap, 1}, dnnl::memory::data_type::bf16);
    buf_.shared_gate_up_buf = Tensor::allocate(ctx, {cap, 1280}, dnnl::memory::data_type::bf16);
    buf_.shared_inter_buf = Tensor::allocate(ctx, {cap, 640}, dnnl::memory::data_type::bf16);
    buf_.shared_out_buf = Tensor::allocate(ctx, {cap, hidden_size_}, dnnl::memory::data_type::bf16);

    if (s_expert_major) {
        buf_.moe_down_acc = phase({down_rows, hidden_size_}, dnnl::memory::data_type::f32);

        expert_offsets_host_ = static_cast<int*>(ctx.alloc_host((cfg_.num_experts + 1) * sizeof(int)));
        expert_counts_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        // At most one descriptor per activation, including every padded expert tail.
        expert_tiles_host_ = static_cast<ops::MoeExpertTile*>(ctx.alloc_host(
            static_cast<size_t>(cap) * cfg_.num_experts_per_tok * sizeof(ops::MoeExpertTile)));
        expert_tiles_dev_ = Tensor::allocate(ctx, {cap, cfg_.num_experts_per_tok, 2}, dnnl::memory::data_type::s32);
        expert_curr_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
        expert_eight_pos_host_ = static_cast<int*>(ctx.alloc_host(cfg_.num_experts * sizeof(int)));
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

void Qwen4ExpBnb4Backend::TrimDecodeRuntimeBuffers(Context& ctx) {
    static const int trim_cap = aila::env::read_int_raw("AILA_Q4E_DECODE_TRIM_CAP", 0);
    if (trim_cap == 0) return;
    if (trim_cap < 64)
        throw std::invalid_argument("AILA_Q4E_DECODE_TRIM_CAP requires 0 or a capacity of at least 64");
    if (runtime_seq_capacity_ <= trim_cap) return;

    // This boundary follows the last explicit prefill forward, including its
    // logits/checkpoint consumers. Keep KV, recurrent state, weights, logits,
    // oneDNN scratch and resident tables; replace only capacity-sized runtime
    // scratch and its host mirrors. The next large prefill grows normally.
    expert_executor_.AbortResidentCold();
    ctx.synchronize();
    const int old_cap = runtime_seq_capacity_;
    const uint64_t bytes_before = ctx.current_allocated_bytes();
    runtime_seq_capacity_ = 0;
    ensure_runtime_buffers(ctx, trim_cap);
    const uint64_t bytes_after = ctx.current_allocated_bytes();
    AILA_LOG_INFO("[Q4E DecodeTrim] cap_before=%d cap_after=%d bytes_before=%llu bytes_after=%llu freed_bytes=%llu boundary=end_prefill",
        old_cap, runtime_seq_capacity_, (unsigned long long)bytes_before,
        (unsigned long long)bytes_after,
        (unsigned long long)(bytes_before > bytes_after ? bytes_before - bytes_after : 0));
}

Qwen4ExpBnb4Backend::ResidentColdWork::~ResidentColdWork() noexcept {
    if (active) executor.AbortResidentCold();
}

sycl::event Qwen4ExpBnb4Backend::RunResidentDecode(Context& ctx, int layer, bool fuse_shared, int count,
                                                   bool verify_rows) {
    if (use_adaptive_pipeline_ && adaptive_cache_.IsAllocated()) {
        constexpr int kMaxTopK = 16;
        const int top_k = cfg_.num_experts_per_tok, intermediate = cfg_.moe_intermediate_size;
        const bool nested = true;
        // Adaptive descriptors are produced only by MakeResidentSlot, which
        // stamps nested blocksize 256; the kernels drop the runtime division.
        const bool nested_fixed256 = true;
        const auto& nested_meta = host_arena_.GetLayerNestedMetadata(layer);

        auto* h_base = buf_.moe_h.data_as<bf16>();
        auto* down = resident_down_.data_as<float>();
        auto* x_base = buf_.x.data_as<bf16>();
        auto* y_base = buf_.y.data_as<bf16>();
        auto* sh_out_base = fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr;
        auto* sh_gate_base = fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr;

        sycl::event last{};
        ResidentColdWork cpu_work{expert_executor_};

        // Opt-in multi-row CPU verification. The native host arena is the RAM
        // complement of the GPU hotset, so every routed miss can be served by
        // the bounded CPU worker in one rows-token batch while the resident
        // hotset runs on the GPU. Enabled only through AILA_Q4E_ARENA_VERIFY_ROWS
        // with the PrepareDecodeVerification scratch present (MTP path) and a
        // complete arena/GPU cover for every routed rank; otherwise the
        // per-token loop below is byte-for-byte unchanged.
        bool arena_verify = verify_rows && count >= 2 && count <= 4 && !decodeHcReferenceForTest_ &&
            verify_down_.valid() && verify_weights_.valid() &&
            expert_executor_.CanExecuteArenaVerifyRows(host_arena_, count, top_k);
        if (arena_verify)
            for (int t = 0; t < count && arena_verify; ++t)
                for (int k = 0; k < top_k; ++k) {
                    const int e = router_host_ids_[t * top_k + k];
                    if (adaptive_cache_.GetSlot(layer, e) < 0 && !host_arena_.GetExpertPointer(layer, e)) {
                        arena_verify = false;
                        break;
                    }
                }
        if (arena_verify) {
            expert_executor_.StageResidentInput(x_base, count);
            float* verify_down = verify_down_.data_as<float>();
            expert_executor_.StartArenaVerifyRows(host_arena_, layer, count, router_host_ids_, sycl::event{});
            ctx.queue().memcpy(verify_weights_.data_as<float>(), router_host_weights_,
                               size_t(count) * top_k * sizeof(float));
            for (int t = 0; t < count; ++t) {
                const int* t_ids = router_host_ids_ + t * top_k;
                const bf16* x_t = x_base + size_t(t) * hidden_size_;
                std::array<int, kMaxTopK> hit_slots{}, hit_ranks{};
                int hit_count = 0;
                for (int k = 0; k < top_k; ++k) {
                    const int slot = adaptive_cache_.GetSlot(layer, t_ids[k]);
                    if (slot >= 0) { hit_slots[hit_count] = slot; hit_ranks[hit_count] = k; ++hit_count; }
                }
                adaptive_cache_.ObserveRouting(layer, t_ids, top_k);
                if (!hit_count) continue;
                std::array<ops::MoeExpertSlotDev, kMaxTopK> hot{};
                for (int i = 0; i < hit_count; ++i)
                    hot[i] = adaptive_cache_.GetResidentSlotDev(hit_slots[i], &nested_meta);
                float* t_down = verify_down + size_t(t) * top_k * hidden_size_;
                last = ops::moe_decode_gate_up_swiglu_grouped(ctx, hot.data(), nullptr, x_t, h_base,
                    hit_count, hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(), nullptr,
                    nested, nested_fixed256);
                last = ops::moe_decode_down_to_rank_f32(ctx, hot.data(), hit_ranks.data(), h_base, t_down,
                    hit_count, top_k, intermediate, hidden_size_, 64, nullptr, nullptr, &last, nested,
                    nested_fixed256);
                adaptive_cache_.RecordResidentConsumer(last);
            }
            const auto cpu_ready = expert_executor_.JoinArenaVerifyRows(verify_down);
            if (cpu_ready != sycl::event{}) last = cpu_ready;
            last = ops::moe_ordered_accumulate_down_rows(ctx, verify_down, verify_weights_.data_as<float>(),
                y_base, count, top_k, hidden_size_, sh_out_base, sh_gate_base, &last);
            if (resident_probe_) resident_probe_(resident_probe_data_, *this, layer);
            return last;
        }


        for (int t = 0; t < count; ++t) {
            const int* t_ids = router_host_ids_ + t * top_k;
            const float* t_weights = router_host_weights_ + t * top_k;
            const bf16* x_t = x_base + static_cast<size_t>(t) * hidden_size_;
            bf16* y_t = y_base + static_cast<size_t>(t) * hidden_size_;
            const bf16* sh_out_t = sh_out_base ? sh_out_base + static_cast<size_t>(t) * hidden_size_ : nullptr;
            const bf16* sh_gate_t = sh_gate_base ? sh_gate_base + t : nullptr;

            // 1. Separate hits and misses via adaptive cache
            int hit_count = 0;
            int miss_count = 0;
            std::array<int, kMaxTopK> hit_slots{};
            std::array<int, kMaxTopK> hit_ranks{};
            std::array<int, kMaxTopK> miss_experts{};
            std::array<int, kMaxTopK> miss_ranks{};

            for (int k = 0; k < top_k; ++k) {
                const int exp_id = t_ids[k];
                const int slot = adaptive_cache_.GetSlot(layer, exp_id);
                if (slot >= 0) {
                    hit_slots[hit_count] = slot;
                    hit_ranks[hit_count] = k;
                    hit_count++;
                } else {
                    miss_experts[miss_count] = exp_id;
                    miss_ranks[miss_count] = k;
                    miss_count++;
                }
            }

            // 2. Observe routing for frequency tracking & stats
            adaptive_cache_.ObserveRouting(layer, t_ids, top_k);

            // Native CPU sources remain owned until Join below. CPU work can
            // overlap the already submitted shared expert and resident kernels.
            const int cpu_count=count==1 && expert_executor_.cpu_overlap_enabled()
                ? expert_executor_.StartArenaCold(host_arena_,layer,miss_experts.data(),miss_ranks.data(),miss_count,x_t) : 0;
            cpu_work.active=cpu_count>0;
            const int gpu_cold_count=miss_count-cpu_count;
            StagingBatch staging{};
            if (gpu_cold_count>0) {
                sycl::queue& dma_q = copy_q_ ? *copy_q_ : ctx.queue();
                staging = adaptive_cache_.StageColdExpertsAsync(
                    host_arena_, dma_q, layer, miss_experts.data()+cpu_count, gpu_cold_count, ping_pong_index_,
                    &expert_executor_);
                ping_pong_index_ ^= 1;
            }

            // 4. Build GPU descriptors and launch XMX fused kernels
            // 4a. Execute hot hits (immediate GPU execution)
            if (hit_count > 0) {
                std::array<ops::MoeExpertSlotDev, kMaxTopK> hot{};
                for (int i = 0; i < hit_count; ++i) {
                    hot[i] = adaptive_cache_.GetResidentSlotDev(hit_slots[i], &nested_meta);
                }
                last = ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, hot.data(), nullptr,
                    x_t, h_base, hit_count,
                    hidden_size_, intermediate, 64, adaptive_cache_.ResidentReady(), nullptr, nested,
                    nested_fixed256);

                last = ops::moe_decode_down_to_rank_f32(
                    ctx, hot.data(), hit_ranks.data(), h_base,
                    down, hit_count, top_k, intermediate, hidden_size_, 64,
                    nullptr, nullptr, &last, nested, nested_fixed256);
                adaptive_cache_.RecordResidentConsumer(last);
            }

            // 4b. Execute cold staging (waiting on PCIe DMA completion)
            if (staging.count > 0) {
                std::array<ops::MoeExpertSlotDev, kMaxTopK> cold{};
                for (int i = 0; i < staging.count; ++i) {
                    cold[i] = adaptive_cache_.GetStagingSlotDev(staging.ping_pong_index, staging.staging_slots[i], &nested_meta);
                }
                auto* cold_h = h_base + size_t(hit_count) * intermediate;

                last = ops::moe_decode_gate_up_swiglu_grouped(
                    ctx, cold.data(), nullptr,
                    x_t, cold_h, staging.count,
                    hidden_size_, intermediate, 64, &staging.ready_event, nullptr, nested,
                    nested_fixed256);

                last = ops::moe_decode_down_to_rank_f32(
                    ctx, cold.data(), miss_ranks.data()+cpu_count, cold_h,
                    down, staging.count, top_k, intermediate, hidden_size_, 64,
                    nullptr, nullptr, &last, nested, nested_fixed256);

                adaptive_cache_.RecordStagingConsumer(staging.ping_pong_index,last);
            }

            const auto cpu_ready=expert_executor_.JoinResidentCold(down);
            if (cpu_count) last=cpu_ready;
            cpu_work.active=false;
            // 5. Accumulate ranked down rows into y
            ctx.queue().memcpy(resident_weights_.data(), t_weights, top_k * sizeof(float));

            const auto* cpu_rows=expert_executor_.ResidentCpuHostRows();
            last = cpu_rows ? ops::moe_ordered_accumulate_cpu_host_rows(ctx,down,resident_weights_.data_as<float>(),
                y_t,1,top_k,hidden_size_,*cpu_rows,sh_out_t,sh_gate_t,&last)
                : ops::moe_ordered_accumulate_down_rows(ctx,down,resident_weights_.data_as<float>(),
                    y_t,1,top_k,hidden_size_,sh_out_t,sh_gate_t,&last,expert_executor_.ResidentCpuResultPacket());
            expert_executor_.FinishResidentCpuRead(last);
        }

        if (resident_probe_) resident_probe_(resident_probe_data_, *this, layer);
        return last;
    }
    return sycl::event{};
}

sycl::event Qwen4ExpBnb4Backend::RunResidentDecode(Context& ctx, int layer, bool fuse_shared,
    GpuReadyLease* device_lease, ResidentColdWork* early_cold) {
    constexpr int kMaxTopK = 16;
    const int top_k = cfg_.num_experts_per_tok, intermediate = cfg_.moe_intermediate_size;
    const bool nested = use_adaptive_pipeline_ || expert_cache_.uses_nested_codes();
    auto lease = device_lease ? std::move(*device_lease) : expert_cache_.BindReady(layer, router_host_ids_, top_k);
    ResidentColdWork cold_guard{expert_executor_};
    const bool cpu_collected = early_cold && early_cold->completed;
    const bool cpu_started = cpu_collected || (early_cold && early_cold->active);
    if (early_cold && early_cold->active) {
        // The local guard is destroyed before the moved route lease. Transfer
        // ownership without leaving the CPU job unguarded or starting it twice.
        cold_guard.active = true;
        early_cold->active = false;
    }
    const int64_t ready_begin = expert_cache_.tracing() ? Qwen4ExpCacheTrace::NowNs() : 0;
    std::array<ops::MoeExpertSlotDev, kMaxTopK> hot{}, cold{};
    auto* h = buf_.moe_h.data_as<bf16>();
    auto* down = resident_down_.data_as<float>();
    sycl::event last{};
    // Every original rank is written exactly once by one of these subsets.
    // No atomics, new quantization, or route-weight multiplication in the dots.
    if (!device_lease && lease.ready_count) {
        expert_cache_.get_slots_dev(lease.ready_slots.data(), lease.ready_count, hot.data());
        last = ops::moe_decode_gate_up_swiglu_grouped(ctx, hot.data(), nullptr,
            buf_.x.data_as<bf16>(), h, lease.ready_count, hidden_size_, intermediate, 64,
            nullptr, nullptr, nested);
        last = ops::moe_decode_down_to_rank_f32(ctx, hot.data(), lease.ready_ranks.data(), h,
            down, lease.ready_count, top_k, intermediate, hidden_size_, 64, nullptr, nullptr, &last, nested);
        if (expert_cache_.tracing()) expert_cache_.trace_host_interval("resident_submit", layer, ready_begin, Qwen4ExpCacheTrace::NowNs());
    }
    std::array<int, kMaxTopK> cold_slots{};
    if(!cpu_started && cold_mode_==2 && expert_cache_.frozen() && lease.cold_count) {
        cold_guard.active=true;
        expert_executor_.StartResidentCold(lease,buf_.x.data_as<bf16>());
    }
    if (lease.cold_count) {
        const auto dma = expert_cache_.AcquireCold(lease, cold_slots.data());
        expert_cache_.get_slots_dev(cold_slots.data(), lease.cold_count, cold.data());
        auto* cold_h = h + size_t(lease.ready_count) * intermediate;
        expert_cache_.record_first_consumer_start();
        last = ops::moe_decode_gate_up_swiglu_grouped(ctx, cold.data(), nullptr,
            buf_.x.data_as<bf16>(), cold_h, lease.cold_count, hidden_size_, intermediate, 64,
            &dma, nullptr, nested);
        last = ops::moe_decode_down_to_rank_f32(ctx, cold.data(), lease.cold_ranks.data(), cold_h,
            down, lease.cold_count, top_k, intermediate, hidden_size_, 64, nullptr, nullptr, &last, nested);
        expert_cache_.record_consumer_reader_fences(cold_slots.data(), lease.cold_count, last);
        expert_cache_.DiscardTransient(cold_slots.data(),lease.cold_count);
    }
    const void* cpu_result_packet = cpu_collected ? expert_executor_.ResidentCpuResultPacket() : nullptr;
    if(cold_guard.active) {
        expert_executor_.JoinResidentCold(down);
        cpu_result_packet = expert_executor_.ResidentCpuResultPacket();
        cold_guard.active=false;
    }
    if (!device_lease) ctx.queue().memcpy(resident_weights_.data(), router_host_weights_, top_k * sizeof(float));
    const auto* cpu_host_rows = expert_executor_.ResidentCpuHostRows();
    auto merge = cpu_host_rows ? ops::moe_ordered_accumulate_cpu_host_rows(ctx, down, resident_weights_.data_as<float>(),
        buf_.y.data_as<bf16>(), 1, top_k, hidden_size_, *cpu_host_rows,
        fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr,
        fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr, &last)
        : ops::moe_ordered_accumulate_down_rows(ctx, down, resident_weights_.data_as<float>(),
        buf_.y.data_as<bf16>(), 1, top_k, hidden_size_,
        fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr,
        fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr, &last, cpu_result_packet);
    expert_executor_.FinishResidentCpuRead(merge);
    lease.Finish(merge); // in-order merge covers both reader subsets and metadata uploads
    if(resident_probe_)resident_probe_(resident_probe_data_,*this,layer);
    expert_cache_.TraceDeviceEvent("resident_hot_down",layer,device_lease?resident_hot_reader_:sycl::event{});
    expert_cache_.TraceDeviceEvent("cold_gpu_down",layer,lease.cold_count?last:sycl::event{});
    expert_cache_.TraceDeviceEvent("rank_merge",layer,merge);
    return merge;
}

sycl::event Qwen4ExpBnb4Backend::RunAdaptivePrefill(Context& ctx,int layer,bool fuse_shared,int count) {
    const int top_k=cfg_.num_experts_per_tok;
    adaptive_cache_.ObserveRouting(layer,router_host_ids_,count*top_k);
    std::fill(prefill_expert_uses_.begin(),prefill_expert_uses_.end(),0);
    prefill_unique_.clear();
    for (int i=0;i<count*top_k;++i) {
        const int e=router_host_ids_[i];
        if (e<0 || e>=cfg_.num_experts) throw std::runtime_error("Invalid Qwen4Exp route");
        ++prefill_expert_uses_[e];
    }
    const auto& metadata=host_arena_.GetLayerNestedMetadata(layer);
    for (int e=0;e<cfg_.num_experts;++e) {
        const int slot=adaptive_cache_.GetSlot(layer,e);
        if (prefill_expert_uses_[e] && slot>=0) {
            prefill_unique_.push_back(e);
            adaptive_prefill_slots_[e]=adaptive_cache_.GetResidentSlotDev(slot,&metadata);
        }
    }
    const int hits=int(prefill_unique_.size());
    for (int e=0;e<cfg_.num_experts;++e)
        if (prefill_expert_uses_[e] && adaptive_cache_.GetSlot(layer,e)<0) prefill_unique_.push_back(e);
    static const int xmx_rows=aila::env::read_int("AILA_Q4E_GROUPED_XMX_MIN_ROWS",0);
    const int threshold=grouped_xmx_min_rows_override_>=0 ? grouped_xmx_min_rows_override_ : xmx_rows;
    const int min_rows=grouped_xmx_supported_ && threshold>0 ? threshold : std::numeric_limits<int>::max();
    int act_start=0,tile_start=0;
    sycl::event last{};
    if (hits) {
        const auto* ready=adaptive_cache_.ResidentReady();
        last=RunGroupedExperts(ctx,layer,count,0,hits,act_start,tile_start,min_rows,ready ? *ready : sycl::event{});
        adaptive_cache_.RecordResidentConsumer(last);
    }
    // A cold expert is uploaded once per layer/window, regardless of its row
    // count. Keep the original W4/G3/XMX math and rank-ordered accumulation.
    for (int start=hits;start<int(prefill_unique_.size());) {
        const int n=std::min(adaptive_cache_.StagingSlotsPerBuffer(),int(prefill_unique_.size())-start);
        auto batch=adaptive_cache_.StageColdExpertsAsync(host_arena_,*copy_q_,layer,
            prefill_unique_.data()+start,n,ping_pong_index_,&expert_executor_);
        ping_pong_index_^=1;
        for (int j=0;j<n;++j)
            adaptive_prefill_slots_[prefill_unique_[start+j]]=adaptive_cache_.GetStagingSlotDev(batch.ping_pong_index,j,&metadata);
        last=RunGroupedExperts(ctx,layer,count,start,n,act_start,tile_start,min_rows,batch.ready_event);
        adaptive_cache_.RecordStagingConsumer(batch.ping_pong_index,last);
        start+=n;
    }
    if (act_start!=count*top_k) throw std::logic_error("Incomplete Qwen4Exp prefill routing map");
    ctx.queue().memcpy(batch_weights_dev_,router_host_weights_,size_t(count)*top_k*sizeof(float));
    return ops::moe_ordered_accumulate_down_rows(ctx,buf_.moe_down_acc.data_as<float>(),batch_weights_dev_,
        buf_.y.data_as<bf16>(),count,top_k,hidden_size_,
        fuse_shared ? buf_.shared_out_buf.data_as<bf16>() : nullptr,
        fuse_shared ? buf_.shared_gate_buf.data_as<bf16>() : nullptr,&last,nullptr);
}

sycl::event Qwen4ExpBnb4Backend::RunGroupedExperts(
    Context& ctx, int layer, int count, int firstUnique, int numUnique,
    int& actStart, int& tileStart, int minRows, const sycl::event& ready) {
    const int top_k = cfg_.num_experts_per_tok;
    static const bool reuse = aila::env::read_flag("AILA_Q4E_WEIGHT_REUSE", true);
    static const bool quads = aila::env::read_flag("AILA_Q4E_4TOKEN_REUSE", true);
    static const bool eights = aila::env::read_flag("AILA_Q4E_8TOKEN_REUSE", false);
    int ne = 0, nq = 0, np = 0, ns = 0, nx = 0;
    for (int u = firstUnique; u < firstUnique + numUnique; ++u) {
        const int n = prefill_expert_uses_[prefill_unique_[u]];
        if (n >= minRows) nx += n;
        else {
            ne += reuse && eights ? n / 8 : 0;
            const int rem8 = reuse && eights ? n % 8 : n;
            nq += reuse && quads ? rem8 / 4 : 0;
            const int rem4 = reuse && quads ? rem8 % 4 : rem8;
            np += reuse ? rem4 / 2 : 0;
            ns += reuse ? rem4 % 2 : rem4;
        }
    }
    const int alu = ne * 8 + nq * 4 + np * 2 + ns;
    // Eights lead the ALU layout so the descending-width dispatch chain keeps
    // every section behind its own gate/up section. With the default zero
    // eights this reduces to the HEAD quad/pair/single layout verbatim.
    int eo = actStart, qo = eo + ne * 8, po = qo + nq * 4, so = po + np * 2, xo = actStart + alu;
    const int tileBegin = tileStart;
    std::fill_n(expert_offsets_host_, cfg_.num_experts, -1);
    for (int u = firstUnique; u < firstUnique + numUnique; ++u) {
        const int e = prefill_unique_[u], n = prefill_expert_uses_[e];
        expert_curr_pos_host_[e] = 0;
        expert_offsets_host_[e] = n >= minRows ? xo : -2;
        if (n >= minRows) {
            for (int row = 0; row < n; row += 8)
                expert_tiles_host_[tileStart++] = {xo + row, std::min(8, n - row)};
            xo += n;
        } else {
            const int g = reuse && eights ? n / 8 : 0;
            const int rem = n - g * 8;
            const int q = reuse && quads ? rem / 4 : 0;
            const int rem4 = rem - q * 4, p = reuse ? rem4 / 2 : 0;
            expert_eight_pos_host_[e] = eo; eo += g * 8;
            expert_quad_pos_host_[e] = qo; qo += q * 4;
            expert_pair_pos_host_[e] = po; po += p * 2;
            expert_single_pos_host_[e] = so; so += rem4 - p * 2;
        }
    }
    for (int i = 0; i < count * top_k; ++i) {
        const int e = router_host_ids_[i], base = expert_offsets_host_[e];
        if (base == -1) continue; // another ready/miss microbatch
        const int seen = expert_curr_pos_host_[e]++, n = prefill_expert_uses_[e];
        int pos;
        if (base >= 0) pos = base + seen;
        else {
            const int g = reuse && eights ? (n / 8) * 8 : 0;
            const int q = reuse && quads ? ((n - g) / 4) * 4 : 0;
            const int p = reuse ? ((n - g - q) / 2) * 2 : 0;
            pos = seen < g ? expert_eight_pos_host_[e] + seen :
                seen < g + q ? expert_quad_pos_host_[e] + seen - g :
                seen < g + q + p ? expert_pair_pos_host_[e] + seen - g - q :
                expert_single_pos_host_[e] + seen - g - q - p;
        }
        expert_map_token_host_[pos] = i / top_k;
        expert_map_rank_host_[pos] = i % top_k;
        expert_map_slot_host_[pos] = use_adaptive_pipeline_ ? adaptive_prefill_slots_[e]
            : expert_cache_.get_slot_dev(prefill_expert_slot_[e]);
    }
    const int acts = alu + nx, tiles = tileStart - tileBegin;
    // Microbatches use disjoint slices: host descriptors remain immutable until
    // their DMA is done. The next layer's router wait drains these consumers.
    ctx.queue().memcpy(expert_map_token_dev_ + actStart, expert_map_token_host_ + actStart, size_t(acts) * sizeof(int));
    ctx.queue().memcpy(expert_map_rank_dev_ + actStart, expert_map_rank_host_ + actStart, size_t(acts) * sizeof(int));
    ctx.queue().memcpy(expert_map_slot_dev_ + actStart, expert_map_slot_host_ + actStart, size_t(acts) * sizeof(ops::MoeExpertSlotDev));
    auto* tilesDev = reinterpret_cast<ops::MoeExpertTile*>(expert_tiles_dev_.data());
    if (tiles) ctx.queue().memcpy(tilesDev + tileBegin, expert_tiles_host_ + tileBegin, size_t(tiles) * sizeof(ops::MoeExpertTile));
    sycl::event last = ready;
    const bool nested = use_adaptive_pipeline_ || expert_cache_.uses_nested_codes();
    const bool nested_fixed256 = use_adaptive_pipeline_;
    if (alu) {
        last = ops::moe_expert_major_prefill_gate_up(ctx, expert_map_slot_dev_ + actStart,
            expert_map_token_dev_ + actStart, expert_map_rank_dev_ + actStart, nullptr,
            buf_.x.data_as<bf16>(), buf_.moe_h.data_as<bf16>(), alu, top_k,
            hidden_size_, cfg_.moe_intermediate_size, 64, &last, nested, np, nq, ne, nested_fixed256);
        last = ops::moe_expert_major_prefill_down(ctx, expert_map_slot_dev_ + actStart,
            expert_map_token_dev_ + actStart, expert_map_rank_dev_ + actStart, nullptr,
            buf_.moe_h.data_as<bf16>(), buf_.moe_down_acc.data_as<float>(), alu, top_k,
            cfg_.moe_intermediate_size, hidden_size_, 64, &last, nested, np, nq, ne, false, nested_fixed256);
    }
    if (tiles) {
        last = ops::moe_grouped_xmx_gate_up(ctx, expert_map_slot_dev_, tilesDev + tileBegin, tiles,
            expert_map_token_dev_, expert_map_rank_dev_, nullptr, buf_.x.data_as<bf16>(),
            buf_.moe_h.data_as<bf16>(), top_k, hidden_size_, cfg_.moe_intermediate_size, nested, &last);
        last = ops::moe_grouped_xmx_down(ctx, expert_map_slot_dev_, tilesDev + tileBegin, tiles,
            expert_map_token_dev_, expert_map_rank_dev_, nullptr, buf_.moe_h.data_as<bf16>(),
            buf_.moe_down_acc.data_as<float>(), top_k, hidden_size_, cfg_.moe_intermediate_size, nested, &last);
        grouped_xmx_tiles_ += tiles;
        grouped_xmx_acts_ += nx;
        if (grouped_expert_probe_) grouped_expert_probe_(*this, layer, count, actStart + alu, tileBegin, tiles, last);
    }
    if (expert_quality_probe_)
        last = expert_quality_probe_(expert_quality_probe_data_, *this, layer,
                                     count, actStart, acts, last);
    // This event covers ALU, XMX and any test-only readers on the compute queue.
    if (!use_adaptive_pipeline_) expert_cache_.record_consumer_reader_fences(prefill_unique_slots_.data() + firstUnique, numUnique, last);
    actStart += acts;
    return last;
}

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
    const int component = component_profile_active_for_test_ && count == 2 && in_features == hidden_size_
        ? (out_features == cfg_.num_experts ? 3 : out_features == 1 ? 4 : -1) : -1;
    if (component >= 0) ctx.synchronize();
    const auto begin = component >= 0 ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    linear.forward(ctx, input, output, count);
    if (component >= 0) {
        ctx.synchronize();
        component_profile_ms_for_test_[component] += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
}

Qwen4ExpBnb4Backend::DecodePendingMlpResidual::~DecodePendingMlpResidual() noexcept {
    // On a host exception, enqueue the residual while its buffers still exist.
    // A failed device queue cannot promise materialization, just as on the old path.
    try { Materialize(); } catch (...) {}
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::Defer() {
    if (pending) throw std::logic_error("MLP residual already pending");
    pending = true;
    ++deferred;
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::Materialize() {
    if (!pending) return;
    ops::hc_inject_add(ctx, hidden, y, inject, 1, hiddenSize, 4);
    pending = false;
    ++materialized;
}

void Qwen4ExpBnb4Backend::DecodePendingMlpResidual::ConsumeNorm(
    const float* normWeight, float eps, bf16* normalized) {
    if (!pending) throw std::logic_error("No pending MLP residual to normalize");
    if (!ops::try_hc_decode_inject_add_and_grouped_rms_norm(ctx, hidden, y, inject,
        normWeight, eps, normalized, hiddenSize, true)) {
        // Clear residual ownership after its submit, before the norm submit can
        // throw. The non-vec path keeps both original kernels unchanged.
        Materialize();
        ops::grouped_rms_norm(ctx, hidden, normWeight, eps, normalized, 1, 4, hiddenSize, true);
        return;
    }
    // Clear immediately after norm submission, before subsequent HC submissions
    // can throw. Unwinding must never inject a successfully consumed residual twice.
    pending = false;
    ++consumedNorm;
}

void Qwen4ExpBnb4Backend::RunHyperConnectionCombined(
    Context& ctx, Qwen4ExpHyperConnectionLayer& hc, int count, bool with_inject,
    const bf16* pending_y, const float* pending_inj, DecodePendingMlpResidual* pending_mlp) {
    if (pending_mlp && pending_mlp->pending) {
        pending_mlp->ConsumeNorm(hc.hc_norm_weight->data_as<float>(), cfg_.rms_norm_eps,
            buf_.hc_N.data_as<bf16>());
    } else if (pending_y && pending_inj) {
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
    static const bool s_small_rows_hc = aila::env::read_flag("AILA_Q4E_SMALL_ROWS_HC", false);
    static const int s_hc_down_wg = [] {
        const int value = aila::env::read_int_raw("AILA_Q4E_HC_DECODE_DOWN_WG", 256);
        return value == 64 || value == 128 ? value : 256;
    }();
    static const bool s_hc_up_streams = aila::env::read_flag("AILA_Q4E_HC_DECODE_UP_STREAMS", false);
    const int hc_down_wg = decodeHcReferenceForTest_ ? 256 : s_hc_down_wg;
    const bool hc_up_streams = !decodeHcReferenceForTest_ && s_hc_up_streams;
    if ((s_small_rows_hc || verify_rows_alignment_) && s_fused_hc && count >= 2 && count <= 4 &&
        hc.mix_down_w && hc.mix_up_w && (!with_inject || hc.inject_w)) {
        ops::hc_decode_down_inject_rows(ctx, buf_.hc_N.data_as<bf16>(),
            hc.mix_down_w->data_as<bf16>(), with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
            buf_.hc_mix_down.data_as<bf16>(), buf_.hc_inject_f32.data_as<float>(),
            lowrank, hidden_size_, 4, count, hc_down_wg);
        ops::hc_decode_up_mix_rows(ctx, buf_.hc_mix_down.data_as<bf16>(), hc.mix_up_w->data_as<bf16>(),
            buf_.hc_N.data_as<bf16>(), buf_.x.data_as<bf16>(), lowrank, hidden_size_, 4, count, hc_up_streams);
        return;
    }
    if (count == 1 && s_fused_hc && hc.mix_down_w && hc.mix_up_w && (!with_inject || hc.inject_w)) {
        ops::hc_decode_down_inject(ctx, static_cast<const bf16*>(buf_.hc_N.data()),
                                   hc.mix_down_w->data_as<bf16>(),
                                   with_inject ? hc.inject_w->data_as<bf16>() : nullptr,
                                   static_cast<bf16*>(buf_.hc_mix_down.data()),
                                   static_cast<float*>(buf_.hc_inject_f32.data()),
                                   lowrank, hidden_size_, 4, hc_down_wg);
        ops::hc_decode_up_mix(ctx, static_cast<const bf16*>(buf_.hc_mix_down.data()),
                              hc.mix_up_w->data_as<bf16>(),
                              static_cast<const bf16*>(buf_.hc_N.data()),
                              static_cast<bf16*>(buf_.x.data()),
                              lowrank, hidden_size_, 4, hc_up_streams);
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
    static const bool profile_components = aila::env::read_flag("AILA_Q4E_PROFILE_COMPONENTS", false);
    component_profile_active_for_test_ = profile_components && options.decode_verification && count == 2;
    component_profile_ms_for_test_.fill(0);
    if (options.decode_verification && options.prefill_remaining >= 0)
        throw std::invalid_argument("Decode verification cannot be a prefill slice");

    int start_pos = session.currentLength_;
    if (!use_adaptive_pipeline_) {
        expert_cache_.SetDemandForwardRows(count);
        if(expert_cache_.CpuPackedPromptRefreshPending(start_pos,count,options.prefill_remaining,options.decode_verification))
            expert_executor_.AbortResidentCold(); // drains CPU readers before rewriting their source arena
        if(cold_epoch_enabled_ || expert_cache_.adaptive_hotset())expert_cache_.BeginHotsetForward(start_pos,count,options.prefill_remaining,options.decode_verification);
    }
    int cached_len = start_pos + count;
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: context length " +
                                 std::to_string(cached_len) + " exceeds max_seq_len " +
                                 std::to_string(max_seq_len_));
    }

    const int checkpoint_after = options.snapshot_policy == SnapshotPolicy::ReusablePrefix
        ? options.checkpoint_after : 0;
    if (checkpoint_after < 0 || (checkpoint_after > 0 &&
        !CanCaptureInteriorCheckpoint(session, count, checkpoint_after))) {
        throw std::invalid_argument("Qwen4ExpBnb4Backend: interior checkpoint requires short dense prefill, two multi-token sides and two slots");
    }
    ensure_runtime_buffers(ctx, count);
    if (moe_compact_tile_ && count > 4) ctx.queue().memset(buf_.moe_compact_invalid.data(),0,sizeof(int));
    StateSnapshot* interior = checkpoint_after > 0
        ? &ReserveSnapshot(session, start_pos + checkpoint_after) : nullptr;
    Bnb4BitLinear::PrefillDispatchStats prefill_stats_t0{};
    if (count > 1) {
        prefill_stats_t0 = Bnb4BitLinear::get_prefill_dispatch_stats();
    }

    if (count > 1 || start_pos == 0) {
        session.consecutive_no_miss_layers_ = 0;
    }

    // 1. Embedding lookup + optional vision embedding override
    EmbedTokens(ctx, tokenIdsDevice, count);

    // P0: Early PLE prepare pipeline (pure prepare + async DMA to double-buffered staging)
    const int* ple_ids_ptr = embed_host_ids_;
    if (!embed_host_.valid()) {
        ctx.memcpy_d2h(embed_host_ids_, tokenIdsDevice, static_cast<size_t>(count) * sizeof(int));
    }
    static const bool ple_lookahead = aila::env::read_flag("AILA_Q4E_PLE_ASYNC", true);
    const bool reuse_ple = ple_lookahead && options.prefill_remaining >= 0 &&
        ple_next_ticket_.session_ptr == &session && ple_next_ticket_.generation == session.generation() &&
        ple_next_ticket_.start_pos == start_pos && ple_next_ticket_.prev_tokens_in == session.ple_state_.prev_tokens &&
        ple_.MatchesInput(ple_next_ticket_, ple_ids_ptr, count);
    if (reuse_ple) ple_ticket_ = ple_next_ticket_;
    else ple_ticket_ = ple_.prepare(ctx, session.ple_state_, ple_ids_ptr, count,
                                    &session, session.generation(), start_pos);
    ple_next_ticket_.valid = false;
    if (ple_lookahead && options.prefill_remaining > 0 && options.next_prefill_count > 0) {
        if (!options.next_prefill_tokens || options.next_prefill_count > options.prefill_remaining ||
            options.next_prefill_count > max_seq_len_ - cached_len)
            throw std::invalid_argument("Invalid Qwen4Exp PLE lookahead chunk");
        // N-gram lookup depends only on the known tokens and two-token history.
        // The existing worker and other staging slot prepare it while this
        // chunk runs; recurrent state is committed only by the actual forward.
        Qwen4ExpPleState future;
        future.prev_tokens = ple_ticket_.prev_tokens_next;
        ple_next_ticket_ = ple_.prepare(ctx, future, options.next_prefill_tokens,
            options.next_prefill_count, &session, session.generation(), cached_len);
    }

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

    constexpr int kMaxTopK = 16; // moe_decode_* kernels capture at most 16 slots
    const int top_k = cfg_.num_experts_per_tok;
    if (top_k > kMaxTopK) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: num_experts_per_tok exceeds 16");
    }
    // Experimental tiny-prefill/verification route. Keep it opt-in until full
    // model numerics and state semantics are accepted for each supported shape.
    static const bool s_small_rows_decode = aila::env::read_flag("AILA_Q4E_SMALL_ROWS_DECODE", false);
    static const bool s_verify_small_rows = aila::env::read_flag("AILA_Q4E_VERIFY_SMALL_ROWS", false);
    const bool verify_rows_alignment = s_verify_small_rows && options.decode_verification &&
        count >= 2 && count <= 4;
    const bool small_rows_decode = (s_small_rows_decode || verify_rows_alignment) && count >= 2 && count <= 4;
    // Scope the verify-path alignment to this forward; the in-order submission
    // means the flag only affects kernels queued here.
    struct VerifyRowsAlignScope {
        Qwen4ExpBnb4Backend& self;
        bool prev;
        VerifyRowsAlignScope(Qwen4ExpBnb4Backend& s, bool v)
            : self(s), prev(s.verify_rows_alignment_) {
            self.verify_rows_alignment_ = v;
            self.ple_.set_rows_decode(v);
        }
        ~VerifyRowsAlignScope() {
            self.verify_rows_alignment_ = prev;
            self.ple_.set_rows_decode(prev);
        }
    } verifyRowsAlignScope{*this, verify_rows_alignment};
    auto project_prefill = [&](Bnb4BitLinear& projection, Tensor& input, Tensor& output) {
        if (small_rows_decode) projection.ForwardDecodeRows(ctx, linear_scratch_, input, output, count);
        else projection.forward(ctx, linear_scratch_, input, output, count);
    };
    // Verify-path alignment: a raw-weight BF16 GEMV per row so router/shared
    // gate projections in a 2..4 row window match sequential B1 decode. Inert
    // (single GEMM) unless AILA_Q4E_VERIFY_SMALL_ROWS is on.
    auto dense_rows = [&](Linear& linear, Tensor* raw_weight, Tensor& input, Tensor& output,
                          int out_features, int in_features) {
        if (!verify_rows_alignment_) {
            DenseLinear(ctx, linear, raw_weight, input, output, count, out_features, in_features);
            return;
        }
        for (int r = 0; r < count; ++r) {
            auto in_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * in_features, {1, in_features});
            auto out_row = Tensor::view(ctx, output.data_as<bf16>() + size_t(r) * out_features, {1, out_features});
            DenseLinear(ctx, linear, raw_weight, in_row, out_row, 1, out_features, in_features);
        }
    };
    static const std::string s_dump_dir = aila::env::read_string("AILA_DEBUG_Q4E_DUMP_DIR", "");
    static const bool s_profile = aila::env::read_flag("AILA_PROFILE_Q4E_DECODE", false);
    static const bool s_work_summary = aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    using prof_clock = std::chrono::steady_clock;
    auto prof_ms = [](prof_clock::time_point a, prof_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto prof_t0 = prof_clock::now();
    static const bool s_trace_markers = aila::env::read_flag("AILA_Q4E_TRACE_MARKERS", false);
    double prof_router_sync_ms = 0.0;
    double prof_acquire_ms = 0.0;
    ExpertCacheStats prof_stats0{};
    const auto adaptive_stats0 = adaptive_cache_.Stats();
    aila::HeteroExecutionStats cold_stats0 = expert_executor_.stats();
    if (!use_adaptive_pipeline_) {
        expert_cache_.trace_forward(start_pos, count, true, options.need_logits);
        if (expert_cache_.tracing() && s_trace_markers)
            ctx.queue().single_task<class Q4ETraceForwardBegin>([] {});
        prof_stats0 = expert_cache_.stats();
        cold_stats0 = expert_executor_.stats();
        if(cold_mode_ && cold_epoch_enabled_ && count==1 && start_pos>0 && !expert_cache_.frozen() && !expert_cache_.adaptive_hotset())
            expert_cache_.FreezeReady(std::min(cfg_.num_experts,expert_cache_.num_slots()-1));
        if (resident_decode_mode_ == 2 && count == 1) expert_cache_.BeginResidentDeviceEpoch();
    }

    // AILA_PROFILE_Q4E_STAGES serializes after every stage; use for relative cost only.
    static const bool s_stages = aila::env::read_flag("AILA_PROFILE_Q4E_STAGES", false);
    enum { kStagePle, kStageHc, kStageLinearAttn, kStageFullAttn, kStageRouter,
           kStageExperts, kStageShared, kStageHead, kStageCount };
    double stage_ms[kStageCount] = {};
    auto stage_t = prof_clock::now();
    static const bool s_predict_stats = aila::env::read_flag("AILA_Q4E_PREDICT_STATS", false);
    // Decode-time speculative prefetch of layer L+1 experts (predicted from layer L's MoE input)
    // overlapping host fill + PCIe DMA with GPU compute of layer L+1's attention.
    static const bool configured_prefetch = aila::env::read_flag("AILA_Q4E_PREFETCH", true);
    // Prediction is a supply policy, independent of the real router. Allow
    // decoding to stop speculative file fills while retaining prefill overlap.
    static const bool decode_prefetch = aila::env::read_flag("AILA_Q4E_PREFETCH_DECODE", true);
    static const bool prefill_prefetch = aila::env::read_flag("AILA_Q4E_PREFETCH_PREFILL", true);
    const bool is_prefill = !options.decode_verification && (options.prefill_remaining >= 0 || count > 1);
    static const bool s_decode_pending_mlp = aila::env::read_flag("AILA_Q4E_DECODE_PENDING_MLP_RESIDUAL", false);
    const bool decode_pending_mlp = !decodeHcReferenceForTest_ && s_decode_pending_mlp && count == 1 && !is_prefill && !interior;
    DecodePendingMlpResidual pending_mlp(ctx, buf_.hidden4.data_as<bf16>(),
        buf_.y.data_as<bf16>(), buf_.hc_inject_f32.data_as<float>(), hidden_size_);
    const bool s_prefetch = configured_prefetch && (is_prefill ? prefill_prefetch : decode_prefetch);
    static const bool s_router_resident_fused = aila::env::read_flag("AILA_Q4E_ROUTER_RESIDENT_FUSED", false);
    static const bool s_moe_front_fused = aila::env::read_flag("AILA_Q4E_MOE_FRONT_FUSED", false);
    static const bool s_front_native_gemv = aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    static const bool s_cpu_early_start = aila::env::read_flag("AILA_Q4E_CPU_EARLY_START", false);
    static const bool s_cpu_async_route = aila::env::read_flag("AILA_Q4E_CPU_ASYNC_ROUTE", false);
    static const int s_hotset_swaps = aila::env::read_int_raw("AILA_Q4E_HOTSET_SWAPS", 96);
    static const bool s_fuse_shared = aila::env::read_flag("AILA_Q4E_MOE_SHARED_FUSED", false);
    static const bool s_predict_gating = aila::env::read_flag("AILA_Q4E_PREDICT_GATING", false);
    // Router top-k output is sorted by probability; only the most confident guesses are prefetched.
    static const int s_prefetch_topn = std::max(1, aila::env::read_int("AILA_Q4E_PREFETCH_TOPN", 10));
    static const int s_prefill_prefetch_max = aila::env::read_int("AILA_Q4E_PREFILL_PREFETCH_MAX", 48);
    // Note: pred_scores_, pred_host_ids_, and pred_host_weights_ are pre-allocated in ensure_runtime_buffers
    std::array<int, kMaxTopK> pred_ids{};
    bool have_pred = false;
    auto stage = [&](int s) {
        if (!s_stages && !s_work_summary) return;
        if (s_stages) ctx.synchronize();
        const auto now = prof_clock::now();
        stage_ms[s] += prof_ms(stage_t, now);
        stage_t = now;
    };

    // 2. 48 Transformer Layers
    for (int L = 0; L < cfg_.num_hidden_layers; ++L) {
        auto& layer = layers_[L];
        auto& cache = session.layerCaches_[L];

        // 2.0. PLE on Layer 1 (Epilogue Residual Fused via P0 PleTicket)
        if (L == 1) {
            // PLE reads the residual stack before this layer's HC norm.
            pending_mlp.Materialize();
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
        if (verification_hidden_in_for_test_ &&
            count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_hidden_in_for_test_ + size_t(L) * verification_stage_rows_for_test_ * 10240,
                buf_.hidden4.data(), size_t(count) * 10240 * sizeof(bf16));
        }
        if (pending_mlp.pending)
            RunHyperConnectionCombined(ctx, layer.attn_hc, 1, true, nullptr, nullptr, &pending_mlp);
        else
            RunHyperConnection(ctx, layer.attn_hc, count, true);
        if (verification_attn_in_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_in_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.x.data(), size_t(count) * hidden_size_ * sizeof(bf16));
        }
        stage(kStageHc);

        // Attention tiles advance this layer's state before routing the whole window.
        RunAttentionWindow(ctx, session, L, count, start_pos, interior, checkpoint_after, !is_prefill);
        if (verification_attn_raw_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_raw_for_test_ + size_t(L) * verification_stage_rows_for_test_ * 6144,
                buf_.attn_out.data(), size_t(count) * 6144 * sizeof(bf16));
        }
        if (verification_attn_out_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_attn_out_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.y.data(), size_t(count) * hidden_size_ * sizeof(bf16));
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
        static const int s_verify_cpu_rows = aila::env::read_int("AILA_Q4E_VERIFY_CPU_ROWS", 0);
        const bool verify_cpu_rows = options.decode_verification && count <= s_verify_cpu_rows &&
            cold_mode_ == 2 && !decodeHcReferenceForTest_ && expert_executor_.CanExecuteVerifyRows(count, top_k);
        std::optional<GpuResidentLayerLease> resident_epoch;
        std::optional<GpuReadyLease> resident_binding;
        ResidentColdWork early_cold{expert_executor_}; // drain CPU before either route lease is released
        if (!use_adaptive_pipeline_ && resident_decode_mode_ == 2 && count == 1)
            resident_epoch.emplace(expert_cache_.BeginResidentLayer(L));
        const bool cpu_async_route = !use_adaptive_pipeline_ && s_cpu_async_route && !decodeHcReferenceForTest_ && resident_epoch &&
            cold_mode_ == 2 && !is_prefill && !s_prefetch &&
            (!expert_cache_.adaptive_hotset() || s_hotset_swaps == 0) && expert_executor_.CanStartAsyncResidentRoute(L);
        const bool cpu_early_start = !use_adaptive_pipeline_ && !cpu_async_route && s_cpu_early_start && resident_epoch &&
            cold_mode_ == 2 && expert_cache_.frozen() && !is_prefill;
        static const bool s_verify_async_route = aila::env::read_flag("AILA_Q4E_VERIFY_ASYNC_ROUTE", false);
        // The four-row screen regressed; keep this measured overlap path at T2.
        const bool cpu_verify_async = s_verify_async_route && count == 2 && verify_cpu_rows && !s_prefetch &&
            (!expert_cache_.adaptive_hotset() || s_hotset_swaps == 0) && expert_executor_.CanStartAsyncResidentRoute(L);
        // Place the activation copy ahead of the router so waiting for the
        // route also makes CPU input ready, before resident/shared GPU work.
        if (cpu_early_start || cpu_async_route || cpu_verify_async ||
            (use_adaptive_pipeline_ && count==1 && expert_executor_.cpu_overlap_enabled()))
            expert_executor_.StageResidentInput(buf_.x.data_as<bf16>(), count);
        const bool gpu_nf4_q8 = resident_epoch && gpuNf4Q8Supported_ && !gpuNf4Q8ReferenceForTest_ &&
            !decodeHcReferenceForTest_ && count == 1 && !is_prefill && resident_decode_mode_ == 2;
        const bool moe_front_fused = s_moe_front_fused && s_front_native_gemv && moeFrontFusedGpu_ &&
            !decodeHcReferenceForTest_ && !gpuNf4Q8ReferenceForTest_ && count == 1 && !is_prefill &&
            hidden_size_ == 2560 && cfg_.num_experts == 512 && layer.router_w && layer.shared_expert_gate_w &&
            layer.router_w->dtype() == dnnl::memory::data_type::bf16 &&
            layer.shared_expert_gate_w->dtype() == dnnl::memory::data_type::bf16 &&
            layer.router_w->numel() == int64_t(512) * hidden_size_ && layer.shared_expert_gate_w->numel() == hidden_size_;
        if (moe_front_fused) {
            ops::bf16_moe_front_fused(ctx, layer.router_w->data_as<bf16>(), layer.shared_expert_gate_w->data_as<bf16>(),
                buf_.x.data_as<bf16>(), buf_.router_scores.data_as<bf16>(), buf_.shared_gate_buf.data_as<bf16>(),
                gpu_nf4_q8 ? buf_.moe_x_q8.data_as<int8_t>() : nullptr,
                gpu_nf4_q8 ? buf_.moe_x_q8_scale.data_as<float>() : nullptr, hidden_size_, cfg_.num_experts);
            if (!moeFrontFusedLogged_) {
                AILA_LOG_INFO("[Q4E MoE Front Fused] actual_enabled=1 hidden=%d experts=%d shared_gate=1 q8_input=%d launches_saved_per_layer=%d",
                    hidden_size_, cfg_.num_experts, int(gpu_nf4_q8), gpu_nf4_q8 ? 2 : 1);
                moeFrontFusedLogged_ = true;
            }
        } else {
            dense_rows(layer.router_gate, layer.router_w, buf_.x, buf_.router_scores,
                       cfg_.num_experts, hidden_size_);
        }
        if (verification_router_inputs_for_test_ && options.decode_verification && count == 2)
            ctx.queue().memcpy(verification_router_inputs_for_test_ + size_t(L) * 2 * hidden_size_,
                buf_.x.data(), size_t(2) * hidden_size_ * sizeof(bf16));
        if (verification_router_x_for_test_ && count <= verification_stage_rows_for_test_) {
            ctx.queue().memcpy(
                verification_router_x_for_test_ + size_t(L) * verification_stage_rows_for_test_ * hidden_size_,
                buf_.x.data(), size_t(count) * hidden_size_ * sizeof(bf16));
        }
        // Top-k writes straight into pinned host memory; the event wait is the only readback.
        sycl::event actual_route_event;
        if (resident_epoch && s_router_resident_fused && cfg_.num_experts == 512) {
            actual_route_event = ops::router_softmax_topk_resident(
                ctx, buf_.router_scores.data_as<bf16>(), nullptr,
                resident_route_ids_.data_as<int>(), resident_weights_.data_as<float>(),
                expert_cache_.resident_layer_table(L), resident_selected_.data_as<int>(),
                router_host_ids_, router_host_weights_, resident_selected_host_,
                cfg_.num_experts, top_k, true);
        } else {
            actual_route_event = ops::router_softmax_topk_device(
                ctx, static_cast<const bf16*>(buf_.router_scores.data()), nullptr,
                resident_epoch ? resident_route_ids_.data_as<int>() : router_host_ids_,
                resident_epoch ? resident_weights_.data_as<float>() : router_host_weights_,
                count, cfg_.num_experts, top_k, true);
            if (resident_epoch) {
                actual_route_event = ops::moe_resident_classify(ctx, resident_route_ids_.data_as<int>(), resident_weights_.data_as<float>(),
                    expert_cache_.resident_layer_table(L), resident_selected_.data_as<int>(), router_host_ids_, router_host_weights_,
                    resident_selected_host_, top_k, cfg_.num_experts);
            }
        }
        auto bind_actual_route = [&] {
            if (!use_adaptive_pipeline_ && cold_epoch_enabled_) expert_cache_.ObserveHotsetRoute(L, router_host_ids_, count, top_k);
            if (resident_epoch)
                resident_binding.emplace(resident_epoch->BindRoute(router_host_ids_, resident_selected_host_, top_k));
            if (!use_adaptive_pipeline_ && expert_cache_.tracing()) expert_cache_.trace_router(L, router_host_ids_, count, top_k);
        };
        auto wait_actual_route = [&] {
            const auto begin = prof_clock::now();
            actual_route_event.wait_and_throw();
            bind_actual_route();
            const auto end = prof_clock::now();
            prof_router_sync_ms += prof_ms(begin, end);
            if (!use_adaptive_pipeline_ && expert_cache_.tracing()) {
                expert_cache_.trace_host_interval("router_wait", L,
                    std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count());
            }
        };
        if (cpu_verify_async) {
            early_cold.active = true;
            expert_executor_.StartAsyncVerifyRoute(L, count, router_host_ids_, actual_route_event);
        } else if (cpu_async_route) {
            early_cold.active = true;
            expert_executor_.StartAsyncResidentRoute(L, router_host_ids_, resident_selected_host_, actual_route_event);
        } else if (cpu_early_start) {
            wait_actual_route();
            if (resident_binding->cold_count) {
                early_cold.active = true;
                expert_executor_.StartResidentCold(*resident_binding, buf_.x.data_as<bf16>());
            }
        }
        if (resident_epoch) {
            const int64_t submit_begin = expert_cache_.tracing() ? Qwen4ExpCacheTrace::NowNs() : 0;
            if(!cpu_early_start && !cpu_async_route && cold_mode_==2 && expert_cache_.frozen())expert_executor_.StageResidentInput(buf_.x.data_as<bf16>());
            if (gpu_nf4_q8) {
                // The in-order queue preserves router publication -> quantized
                // resident projections -> shared work -> ordered rank merge.
                if (!moe_front_fused)
                    ops::quantize_moe_input_q8_64(ctx, buf_.x.data_as<bf16>(),
                        buf_.moe_x_q8.data_as<int8_t>(), buf_.moe_x_q8_scale.data_as<float>(), hidden_size_);
                ops::moe_resident_gate_up_q8_64(ctx, expert_cache_.resident_descriptors(), resident_selected_.data_as<int>(),
                    buf_.moe_x_q8.data_as<int8_t>(), buf_.moe_x_q8_scale.data_as<float>(),
                    buf_.moe_h.data_as<bf16>(), buf_.moe_h_q8.data_as<int8_t>(), buf_.moe_h_q8_scale.data_as<float>(),
                    top_k, hidden_size_, cfg_.moe_intermediate_size, expert_cache_.uses_nested_codes());
                resident_hot_reader_ = ops::moe_resident_down_q8_64(ctx, expert_cache_.resident_descriptors(), resident_selected_.data_as<int>(),
                    buf_.moe_h_q8.data_as<int8_t>(), buf_.moe_h_q8_scale.data_as<float>(), resident_down_.data_as<float>(),
                    top_k, hidden_size_, cfg_.moe_intermediate_size, expert_cache_.uses_nested_codes());
                if (!gpuNf4Q8Logged_) {
                    AILA_LOG_INFO("[Q4E GPU NF4Q8] actual_enabled=1 hidden=%d intermediate=%d top_k=%d nested=%d activation_block=64",
                        hidden_size_, cfg_.moe_intermediate_size, top_k, int(expert_cache_.uses_nested_codes()));
                    gpuNf4Q8Logged_ = true;
                }
            } else {
                ops::moe_resident_gate_up(ctx, expert_cache_.resident_descriptors(), resident_selected_.data_as<int>(),
                    buf_.x.data_as<bf16>(), buf_.moe_h.data_as<bf16>(), top_k, hidden_size_, cfg_.moe_intermediate_size, expert_cache_.uses_nested_codes());
                resident_hot_reader_ = ops::moe_resident_down(ctx, expert_cache_.resident_descriptors(), resident_selected_.data_as<int>(),
                    buf_.moe_h.data_as<bf16>(), resident_down_.data_as<float>(), top_k, hidden_size_, cfg_.moe_intermediate_size, expert_cache_.uses_nested_codes());
            }
            if (expert_cache_.tracing()) expert_cache_.trace_host_interval("resident_gpu_submit", L, submit_begin, Qwen4ExpCacheTrace::NowNs());
        }
        const auto cur_cache_stats = !use_adaptive_pipeline_ ? expert_cache_.stats() : ExpertCacheStats{};
        const bool high_hit_rate = (cur_cache_stats.hit_count + cur_cache_stats.miss_count > 100) &&
                                   (cur_cache_stats.hit_rate() >= 0.90);
        const bool gate_prediction = s_predict_gating && (count == 1) &&
            (session.consecutive_no_miss_layers_ >= 2 || (high_hit_rate && session.consecutive_no_miss_layers_ > 0));

        const bool predict_next = !use_adaptive_pipeline_ && ((count == 1)
            ? (!gate_prediction && (s_prefetch || s_predict_stats) && (L + 1 < cfg_.num_hidden_layers))
            : (s_prefetch && (s_prefill_prefetch_max > 0) && (L + 1 < cfg_.num_hidden_layers)));
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
        if (!moe_front_fused)
            dense_rows(layer.shared_expert_gate, layer.shared_expert_gate_w, buf_.x, buf_.shared_gate_buf,
                       1, hidden_size_);
        if (count == 1) {
            bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                ctx, layer.shared_gate_up_proj, buf_.x, buf_.shared_inter_buf, cfg_.shared_expert_intermediate_size);
            if (!fused_ok) {
                layer.shared_gate_up_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.x, buf_.shared_gate_up_buf, 1);
                ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                               static_cast<bf16*>(buf_.shared_inter_buf.data()), 1, cfg_.shared_expert_intermediate_size);
            }
            layer.shared_down_proj.ForwardDecodeRows(ctx, linear_scratch_, buf_.shared_inter_buf, buf_.shared_out_buf, 1);
        } else if (verify_rows_alignment_) {
            // One row at a time keeps the fused shared gate/up/SiLU/down kernels
            // identical to B1 decode for every row of a 2..4 row verify window.
            for (int r = 0; r < count; ++r) {
                auto x_row = Tensor::view(ctx, buf_.x.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                auto gu_row = Tensor::view(ctx, buf_.shared_gate_up_buf.data_as<bf16>() + size_t(r) * cfg_.shared_expert_intermediate_size,
                                           {1, cfg_.shared_expert_intermediate_size});
                auto si_row = Tensor::view(ctx, buf_.shared_inter_buf.data_as<bf16>() + size_t(r) * cfg_.shared_expert_intermediate_size,
                                           {1, cfg_.shared_expert_intermediate_size});
                auto so_row = Tensor::view(ctx, buf_.shared_out_buf.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                bool fused_ok = Bnb4BitLinear::try_forward_decode_gate_up_swiglu(
                    ctx, layer.shared_gate_up_proj, x_row, si_row, cfg_.shared_expert_intermediate_size);
                if (!fused_ok) {
                    layer.shared_gate_up_proj.ForwardDecodeRows(ctx, linear_scratch_, x_row, gu_row, 1);
                    ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(gu_row.data()),
                                                   static_cast<bf16*>(si_row.data()), 1, cfg_.shared_expert_intermediate_size);
                }
                layer.shared_down_proj.ForwardDecodeRows(ctx, linear_scratch_, si_row, so_row, 1);
            }
        } else {
            project_prefill(layer.shared_gate_up_proj, buf_.x, buf_.shared_gate_up_buf);
            ops::moe_silu_gate_up_mul_flat(ctx, static_cast<const bf16*>(buf_.shared_gate_up_buf.data()),
                                           static_cast<bf16*>(buf_.shared_inter_buf.data()), count, cfg_.shared_expert_intermediate_size);
            project_prefill(layer.shared_down_proj, buf_.shared_inter_buf, buf_.shared_out_buf);
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

        // B2: Decoupled route synchronization - wait ONLY for actual router, allowing concurrent prediction
        if (cpu_async_route) {
            // The worker waits for route publication and computes mapped cold
            // experts while these resident/shared kernels are already queued.
            expert_executor_.JoinResidentCold(resident_down_.data_as<float>());
            bind_actual_route();
            expert_executor_.RetireAsyncResidentRoute(*resident_binding);
            early_cold.active=false;early_cold.completed=true;
        } else if (!cpu_early_start) wait_actual_route();
        if (!use_adaptive_pipeline_ && have_pred && s_predict_stats && count == 1) {
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
        stage(kStageRouter);
        // Unified Prefill and Decode via Qwen4ExpAdaptiveCache + Qwen4ExpHostArena
        if (use_adaptive_pipeline_ && adaptive_cache_.IsAllocated()) {
            if (count>4) RunAdaptivePrefill(ctx,L,s_fuse_shared,count);
            else RunResidentDecode(ctx,L,s_fuse_shared,count, options.decode_verification);
        } else {
            throw std::runtime_error("Qwen4Exp: legacy expert_cache pipeline has been removed; please enable adaptive pipeline.");
        }
        ensure_prediction_ready();
        stage(kStageExperts);

        const uint64_t misses_after_moe = use_adaptive_pipeline_
            ? static_cast<uint64_t>(adaptive_cache_.Stats().cache_misses)
            : expert_cache_.stats().miss_count;
        if (misses_after_moe > cur_cache_stats.miss_count) {
            session.consecutive_no_miss_layers_ = 0;
        } else {
            session.consecutive_no_miss_layers_++;
        }

        if (!s_fuse_shared) {
            add_shared_expert_to_y(ctx, buf_.y, buf_.shared_out_buf, buf_.shared_gate_buf,
                                   count, hidden_size_);
        }

        // 2.6. MLP Residual Add
        if (decode_pending_mlp) {
            pending_mlp.Defer();
            // Per-layer observers and serial stage profiling require a complete
            // hidden stack here. Ordinary decode consumes it in the next norm.
            if (!s_dump_dir.empty() || s_stages) pending_mlp.Materialize();
        } else {
            ops::hc_inject_add(ctx, static_cast<bf16*>(buf_.hidden4.data()),
                               static_cast<const bf16*>(buf_.y.data()),
                               static_cast<const float*>(buf_.hc_inject_f32.data()),
                               count, hidden_size_, 4);
        }
        stage(kStageShared);

        if (!s_dump_dir.empty()) {
            std::string dump_path = s_dump_dir + "/layer_" + std::to_string(L) + ".bin";
            dump_hidden_row(ctx, dump_path, buf_.hidden4, count - 1, 10240);
        }
    }

    if (use_adaptive_pipeline_ && adaptive_cache_.IsAllocated() && count == 1) {
        // Raw reads so explicit 0 is meaningful: FREQ=0 disables the synchronous
        // swap pass (measured: production decode +8.0% rate with bit-identical
        // requests; see docs/plans/qwen38-swap-longwindow-2026-10-10.md) and
        // MAX_SWAPS=0 keeps the documented "no total cap" meaning.
        static const int s_adaptive_swap_freq = aila::env::read_int_raw("AILA_Q4E_ADAPTIVE_SWAP_FREQ", 0);
        static const int s_max_swaps = aila::env::read_int_raw("AILA_Q4E_MAX_SWAPS", 2);
        if (s_adaptive_swap_freq > 0 && (cached_len % s_adaptive_swap_freq == 0)) {
            auto swap_plans = adaptive_cache_.PlanSwaps(1, s_max_swaps);
            if (!swap_plans.empty()) {
                sycl::queue& dma_q = copy_q_ ? *copy_q_ : ctx.queue();
                adaptive_cache_.ExecuteSwapsAsync(host_arena_, dma_q, swap_plans);
            }
        }
    }

    if (options.need_logits) {
        // 3. Final Hyper-Connection Mixer
        if (pending_mlp.pending)
            RunHyperConnectionCombined(ctx, mixer_, 1, false, nullptr, nullptr, &pending_mlp);
        else
            RunHyperConnection(ctx, mixer_, count, false);

        if (!s_dump_dir.empty()) {
            std::string mixer_path = s_dump_dir + "/mixer.bin";
            dump_hidden_row(ctx, mixer_path, buf_.x, count - 1, 2560);
        }

        // 4. LM Head (Compute for last token)
        auto forward_head = [&](Tensor& input, int rows) {
            if (lmHeadQ8Enabled_) {
                ops::bf16_q8_head_forward(ctx, lm_head_q8_weight_, lm_head_q8_scale_, input, buf_.logits,
                    lm_head_q8_input_, lm_head_q8_input_scale_, rows, hidden_size_, cfg_.vocab_size);
                ++lmHeadQ8ForwardCount_;
                lmHeadQ8RowCount_ += static_cast<uint64_t>(rows);
                if (lmHeadQ8ForwardCount_ == 1)
                    AILA_LOG_INFO("[Q4E LM Head Q8] actual_enabled=1 rows=%d hidden=%d vocab=%d activation_block=64 input_quantization=1 experimental=1",
                        rows, hidden_size_, cfg_.vocab_size);
            } else if (verify_rows_alignment_ && rows > 1) {
                // Verify-path alignment: one head GEMV per row so the window
                // matches sequential B1 logits exactly.
                for (int r = 0; r < rows; ++r) {
                    auto x_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                    auto y_row = Tensor::view(ctx, buf_.logits.data_as<bf16>() + size_t(r) * cfg_.vocab_size, {1, cfg_.vocab_size});
                    lm_head_.forward(ctx, x_row, y_row, 1);
                }
            } else lm_head_.forward(ctx, input, buf_.logits, rows);
        };
        if (lastTokenOnly) {
            const bf16* last_x = static_cast<const bf16*>(buf_.x.data()) + static_cast<size_t>(count - 1) * hidden_size_;
            Tensor x_last_view = Tensor::view(ctx, const_cast<bf16*>(last_x), {1, hidden_size_});
            forward_head(x_last_view, 1);
            lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {1, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
        } else {
            if (buf_.logits.shape(0) < count) {
                buf_.logits = Tensor::allocate(ctx, {count, cfg_.vocab_size}, dnnl::memory::data_type::bf16);
            }
            forward_head(buf_.x, count);
            lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(), {count, static_cast<int64_t>(cfg_.vocab_size)}, buf_.logits.dtype());
        }

        stage(kStageHead);

        if (!s_dump_dir.empty()) {
            std::string logits_path = s_dump_dir + "/logits.bin";
            dump_hidden_row(ctx, logits_path, buf_.logits, 0, cfg_.vocab_size);
        }
    }

    // No-logits forwards must also leave a materialized stack before snapshot,
    // state publication or return. This is a no-op after the final mixer.
    pending_mlp.Materialize();
    if (!use_adaptive_pipeline_) {
        expert_cache_.wait_prefetch();
    }
    if (moe_compact_tile_ && count > 4) {
        int invalid=0;ctx.memcpy_d2h(&invalid,buf_.moe_compact_invalid.data(),sizeof(invalid));
        if(invalid)throw std::runtime_error("Compact MoE Q36 contribution is nonfinite or exceeds 2^20; window not committed");
    }
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
    if (!use_adaptive_pipeline_) {
        if (expert_cache_.tracing() && s_trace_markers)
            ctx.queue().single_task<class Q4ETraceForwardEnd>([] {});
        expert_cache_.trace_forward(start_pos, count, false, options.need_logits);
        if(cold_epoch_enabled_)expert_cache_.EndHotsetForward(count,options.prefill_remaining);
    }
    // An explicit remaining=0 is the final prefill slice. Trim after all
    // checkpoint/logit work is submitted and before the first decode call;
    // steady-state decode never allocates or frees this workspace.
    if (options.prefill_remaining == 0) TrimDecodeRuntimeBuffers(ctx);
    if (!use_adaptive_pipeline_ && expert_cache_.adaptive_hotset()) {
        const auto& hs=expert_cache_.stats();
        AILA_LOG_INFO("[Q4E HotsetEpoch] pos=%d count=%d slots=%d swaps=%llu refill_bytes=%llu refill_ns=%llu publishes=%llu ready_adoptions=%llu ready_adoption_ns=%llu",
            start_pos,count,expert_cache_.frozen_slots(),(unsigned long long)hs.epoch_swaps,
            (unsigned long long)hs.epoch_refill_bytes,(unsigned long long)hs.epoch_refill_ns,(unsigned long long)hs.epoch_publishes,
            (unsigned long long)hs.ready_adoptions,(unsigned long long)hs.ready_adoption_ns);
    }

    if (expert_histogram_enabled_ && count > 1) {
        for (int b = 0; b < 6; ++b) {
            const auto& bin = expert_histogram_[b];
            AILA_LOG_INFO("[Q4E ExpertHistogram] cumulative bucket=%d requests=%llu acts=%llu hits=%llu misses=%llu",
                b, static_cast<unsigned long long>(bin[0]), static_cast<unsigned long long>(bin[1]),
                static_cast<unsigned long long>(bin[2]), static_cast<unsigned long long>(bin[3]));
        }
        AILA_LOG_INFO("[Q4E GroupedXMX] cumulative tiles=%llu acts=%llu",
            static_cast<unsigned long long>(grouped_xmx_tiles_), static_cast<unsigned long long>(grouped_xmx_acts_));
    }
    if (use_adaptive_pipeline_ && s_work_summary) {
        const auto& a = adaptive_cache_.Stats();
        const auto& c = expert_executor_.stats();
        // These are coordinator/CPU intervals, never an additive GPU profile.
        // No new queue wait or barrier is introduced by this summary.
        AILA_LOG_INFO("[Q4E AdaptiveWork] pos=%d count=%d ple_reused=%d submitted_ms=%.3f router_wait_ms=%.3f "
            "ple_host_ms=%.3f hc_host_ms=%.3f linear_host_ms=%.3f full_host_ms=%.3f "
            "experts_host_ms=%.3f head_host_ms=%.3f fill_ms=%.3f bank_wait_ms=%.3f swap_ms=%.3f "
            "cpu_compute_ms=%.3f cpu_join_ms=%.3f cpu_tasks=%llu hits=%llu misses=%llu staged=%llu h2d_mb=%.3f",
            start_pos,count,int(reuse_ple),prof_ms(prof_t0,prof_clock::now()),prof_router_sync_ms,
            stage_ms[kStagePle],stage_ms[kStageHc],stage_ms[kStageLinearAttn],stage_ms[kStageFullAttn],
            stage_ms[kStageExperts],stage_ms[kStageHead],
            (a.staging_fill_ns-adaptive_stats0.staging_fill_ns)*1e-6,
            (a.staging_wait_ns-adaptive_stats0.staging_wait_ns)*1e-6,
            (a.swap_host_ns-adaptive_stats0.swap_host_ns)*1e-6,
            (c.cpu_compute_ns-cold_stats0.cpu_compute_ns)*1e-6,
            (c.cpu_join_wait_ns-cold_stats0.cpu_join_wait_ns)*1e-6,
            (unsigned long long)(c.cpu_tasks_completed-cold_stats0.cpu_tasks_completed),
            (unsigned long long)(a.cache_hits-adaptive_stats0.cache_hits),
            (unsigned long long)(a.cache_misses-adaptive_stats0.cache_misses),
            (unsigned long long)(a.cold_experts_staged-adaptive_stats0.cold_experts_staged),
            (a.cold_bytes_transferred-adaptive_stats0.cold_bytes_transferred)/1048576.0);
    }
    if (!use_adaptive_pipeline_ && s_work_summary) {
        const auto& supply = expert_cache_.stats();
        AILA_LOG_INFO("[Q4E Supply] pos=%d count=%d acquire_ms=%.6f router_wait_ms=%.6f fill_ms=%.6f dma_wait_ms=%.6f demand_wait_ms=%.6f",
            start_pos, count, prof_acquire_ms, prof_router_sync_ms,
            (supply.fill_ns-prof_stats0.fill_ns)*1e-6, (supply.dma_wait_ns-prof_stats0.dma_wait_ns)*1e-6,
            (supply.demand_exposed_wait_ns-prof_stats0.demand_exposed_wait_ns)*1e-6);
        if (decode_pending_mlp) {
            AILA_LOG_INFO("[Q4E PendingMlp] pos=%d count=%d deferred=%llu fused=%llu materialized=%llu",
                start_pos, count, (unsigned long long)pending_mlp.deferred,
                (unsigned long long)pending_mlp.consumedNorm, (unsigned long long)pending_mlp.materialized);
        }
        const auto& after = expert_cache_.stats();
        if(cold_mode_ || resident_prefill_) {
            const auto& cpu=expert_executor_.stats();
            if (cpu.cpu_scale_cache_bytes) {
                AILA_LOG_INFO("[Q4E CpuScaleCache] pos=%d count=%d bytes=%llu hits=%llu misses=%llu fills=%llu evictions=%llu fallbacks=%llu prepare_ns=%llu",
                    start_pos,count,(unsigned long long)cpu.cpu_scale_cache_bytes,
                    (unsigned long long)(cpu.cpu_scale_cache_hits-cold_stats0.cpu_scale_cache_hits),
                    (unsigned long long)(cpu.cpu_scale_cache_misses-cold_stats0.cpu_scale_cache_misses),
                    (unsigned long long)(cpu.cpu_scale_cache_fills-cold_stats0.cpu_scale_cache_fills),
                    (unsigned long long)(cpu.cpu_scale_cache_evictions-cold_stats0.cpu_scale_cache_evictions),
                    (unsigned long long)(cpu.cpu_scale_cache_fallbacks-cold_stats0.cpu_scale_cache_fallbacks),
                    (unsigned long long)(cpu.cpu_scale_cache_prepare_ns-cold_stats0.cpu_scale_cache_prepare_ns));
            }
            if(cpu.cpu_route_async_layers>cold_stats0.cpu_route_async_layers) {
                AILA_LOG_INFO("[Q4E AsyncRoute] pos=%d count=%d layers=%llu worker_route_wait_ns=%llu",
                    start_pos,count,(unsigned long long)(cpu.cpu_route_async_layers-cold_stats0.cpu_route_async_layers),
                    (unsigned long long)(cpu.cpu_route_wait_ns-cold_stats0.cpu_route_wait_ns));
            }
            if(cpu.cpu_verify_batches>cold_stats0.cpu_verify_batches) {
                AILA_LOG_INFO("[Q4E CpuVerify] pos=%d count=%d batches=%llu bitwise=1",
                    start_pos,count,(unsigned long long)(cpu.cpu_verify_batches-cold_stats0.cpu_verify_batches));
            }
            if(cpu.cpu_a64_verify_batches>cold_stats0.cpu_a64_verify_batches) {
                AILA_LOG_INFO("[Q4E CpuA64Verify] pos=%d count=%d batches=%llu bitwise=1 reference=original_a64",
                    start_pos,count,(unsigned long long)(cpu.cpu_a64_verify_batches-cold_stats0.cpu_a64_verify_batches));
            }
            if(cpu.cpu_diagnostic_batches>cold_stats0.cpu_diagnostic_batches) {
                AILA_LOG_INFO("[Q4E CpuDiagnostic] pos=%d count=%d batches=%llu failures=%llu process_faults=%llu process_user_ns=%llu process_kernel_ns=%llu gate_up_ns=%llu quantize_ns=%llu down_ns=%llu gate_chunks=%llu down_chunks=%llu",
                    start_pos,count,
                    (unsigned long long)(cpu.cpu_diagnostic_batches-cold_stats0.cpu_diagnostic_batches),
                    (unsigned long long)(cpu.cpu_diagnostic_failures-cold_stats0.cpu_diagnostic_failures),
                    (unsigned long long)(cpu.cpu_process_faults-cold_stats0.cpu_process_faults),
                    (unsigned long long)(cpu.cpu_process_user_ns-cold_stats0.cpu_process_user_ns),
                    (unsigned long long)(cpu.cpu_process_kernel_ns-cold_stats0.cpu_process_kernel_ns),
                    (unsigned long long)(cpu.cpu_gate_up_ns-cold_stats0.cpu_gate_up_ns),
                    (unsigned long long)(cpu.cpu_quantize_ns-cold_stats0.cpu_quantize_ns),
                    (unsigned long long)(cpu.cpu_down_ns-cold_stats0.cpu_down_ns),
                    (unsigned long long)(cpu.cpu_gate_chunks-cold_stats0.cpu_gate_chunks),
                    (unsigned long long)(cpu.cpu_down_chunks-cold_stats0.cpu_down_chunks));
            }
            AILA_LOG_INFO("[Q4E Cold] pos=%d count=%d mode=%d prefill_direct=%d hot_slots=%d cpu_tasks=%llu eligible=%llu budget_rejected=%llu compute_ns=%llu join_ns=%llu queue_ns=%llu activation_ns=%llu result_bytes=%llu direct=%llu cold=%llu acquire_calls=%llu mapped_tasks=%llu cpu_cache_tasks=%llu source_prefetch_ns=%llu source_prefetch_calls=%llu source_prefetch_ranges=%llu source_prefetch_bytes=%llu source_prefetch_failures=%llu source_touch_ns=%llu source_touch_pages=%llu",
                start_pos,count,cold_mode_,int(resident_prefill_),expert_cache_.frozen_slots(),
                (unsigned long long)(cpu.cpu_tasks_completed-cold_stats0.cpu_tasks_completed),
                (unsigned long long)(cpu.cold_eligible-cold_stats0.cold_eligible),
                (unsigned long long)(cpu.cold_budget_rejected-cold_stats0.cold_budget_rejected),
                (unsigned long long)(cpu.cpu_compute_ns-cold_stats0.cpu_compute_ns),
                (unsigned long long)(cpu.cpu_join_wait_ns-cold_stats0.cpu_join_wait_ns),
                (unsigned long long)(cpu.cold_queue_ns-cold_stats0.cold_queue_ns),
                (unsigned long long)(cpu.cpu_activation_wait_ns-cold_stats0.cpu_activation_wait_ns),
                (unsigned long long)(cpu.cpu_result_bytes-cold_stats0.cpu_result_bytes),
                (unsigned long long)(after.direct_entries-prof_stats0.direct_entries),
                (unsigned long long)(after.cold_request_entries-prof_stats0.cold_request_entries),
                (unsigned long long)(after.acquire_calls-prof_stats0.acquire_calls),
                (unsigned long long)(cpu.cpu_mapped_tasks_completed-cold_stats0.cpu_mapped_tasks_completed),
                (unsigned long long)(cpu.cpu_cached_tasks_completed-cold_stats0.cpu_cached_tasks_completed),
                (unsigned long long)(cpu.cpu_mmap_prefetch_ns-cold_stats0.cpu_mmap_prefetch_ns),
                (unsigned long long)(cpu.cpu_mmap_prefetch_calls-cold_stats0.cpu_mmap_prefetch_calls),
                (unsigned long long)(cpu.cpu_mmap_prefetch_ranges-cold_stats0.cpu_mmap_prefetch_ranges),
                (unsigned long long)(cpu.cpu_mmap_prefetch_bytes-cold_stats0.cpu_mmap_prefetch_bytes),
                (unsigned long long)(cpu.cpu_mmap_prefetch_failures-cold_stats0.cpu_mmap_prefetch_failures),
                (unsigned long long)(cpu.cpu_mmap_touch_ns-cold_stats0.cpu_mmap_touch_ns),
                (unsigned long long)(cpu.cpu_mmap_touch_pages-cold_stats0.cpu_mmap_touch_pages));
        }
        if (resident_decode_mode_ && count == 1) {
            AILA_LOG_INFO("[Q4E Resident] pos=%d mode=%d route=%llu resident=%llu direct=%llu cold=%llu acquire_calls=%llu demand_bytes=%llu metadata_bytes=%llu",
                start_pos, resident_decode_mode_,
                static_cast<unsigned long long>(after.direct_route_entries - prof_stats0.direct_route_entries),
                static_cast<unsigned long long>(after.resident_route_entries - prof_stats0.resident_route_entries),
                static_cast<unsigned long long>(after.direct_entries - prof_stats0.direct_entries),
                static_cast<unsigned long long>(after.cold_request_entries - prof_stats0.cold_request_entries),
                static_cast<unsigned long long>(after.acquire_calls - prof_stats0.acquire_calls),
                static_cast<unsigned long long>(after.total_transferred_bytes - prof_stats0.total_transferred_bytes),
                static_cast<unsigned long long>(after.resident_metadata_bytes - prof_stats0.resident_metadata_bytes));
        }
        AILA_LOG_INFO("[Q4E Work] pos=%d count=%d demand_bytes=%llu host_misses=%llu host_hits=%llu device_bytes=%zu complement_evictions=%llu complement_scans=%llu complement_searches=%llu short_part_fills=%llu",
            start_pos, count,
            static_cast<unsigned long long>(after.total_transferred_bytes - prof_stats0.total_transferred_bytes),
            static_cast<unsigned long long>(after.host_pool_misses - prof_stats0.host_pool_misses),
            static_cast<unsigned long long>(after.host_pool_hits - prof_stats0.host_pool_hits),
            ctx.current_allocated_bytes(),
            static_cast<unsigned long long>(after.host_complement_evictions - prof_stats0.host_complement_evictions),
            static_cast<unsigned long long>(after.host_complement_scans - prof_stats0.host_complement_scans),
            static_cast<unsigned long long>(after.host_complement_searches - prof_stats0.host_complement_searches),
            static_cast<unsigned long long>(after.demand_short_part_fills - prof_stats0.demand_short_part_fills));
    }
    if (component_profile_active_for_test_) {
        const auto& p = component_profile_ms_for_test_;
        AILA_LOG_INFO("[Q4E Components] pos=%d count=%d linear_in=%.4f gdn=%.4f linear_out=%.4f router=%.4f shared_gate=%.4f ms serialized=1",
            start_pos,count,p[0],p[1],p[2],p[3],p[4]);
        component_profile_active_for_test_=false;
    }
    if (s_profile) {
        ctx.synchronize();
        if (use_adaptive_pipeline_) {
            const auto& as = adaptive_cache_.Stats();
            AILA_LOG_INFO("[Q4E Adaptive Profile] pos=%d count=%d total=%.2fms router_sync=%.2fms hits=%llu misses=%llu staged=%llu DMA=%.1fMB logits=%d",
                          start_pos, count, prof_ms(prof_t0, prof_clock::now()),
                          prof_router_sync_ms,
                          (unsigned long long)as.cache_hits,
                          (unsigned long long)as.cache_misses,
                          (unsigned long long)as.cold_experts_staged,
                          (as.warm_bytes_transferred + as.cold_bytes_transferred + as.swap_bytes_transferred) / (1024.0 * 1024.0),
                          options.need_logits ? 1 : 0);
        } else {
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
        }
        if (count > 1) {
            auto dstats = Bnb4BitLinear::get_prefill_dispatch_stats();
            AILA_LOG_INFO("[Q4E Prefill Dispatch] linear calls: fused_gemm=%llu cached=%llu on_demand=%llu sync_waits=%llu direct_io=%llu",
                          static_cast<unsigned long long>(dstats.fused_gemm),
                          static_cast<unsigned long long>(dstats.cached_weight),
                          static_cast<unsigned long long>(dstats.on_demand_dequant),
                          static_cast<unsigned long long>(dstats.sync_wait),
                          static_cast<unsigned long long>(dstats.direct_io));
        }
        if (!use_adaptive_pipeline_ && s_predict_stats && pred_total_ > 0) {
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
            if (!use_adaptive_pipeline_ && expert_executor_.stats().layers_hetero_executed > 0) {
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
        if (!use_adaptive_pipeline_ && (expert_executor_.stats().ready_first_layers > 0 || expert_executor_.stats().cpu_tasks_dispatched > 0)) {
            AILA_LOG_INFO("[Q4E ReadyFirst] layers=%llu cpu_tasks=%llu cpu_compute=%.2fms cpu_activation_wait=%.2fms",
                          static_cast<unsigned long long>(expert_executor_.stats().ready_first_layers),
                          static_cast<unsigned long long>(expert_executor_.stats().cpu_tasks_dispatched),
                          expert_executor_.stats().cpu_compute_ns * 1e-6,
                          expert_executor_.stats().cpu_activation_wait_ns * 1e-6);
        }
        if (!use_adaptive_pipeline_ && expert_executor_.stats().cpu_async_layers > 0) {
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
    ple_next_ticket_.valid = false;

    for (auto& entry : session.snapshots_) {
        if (entry.length > target) entry.length = -1;
    }
    session.currentLength_ = target;
    session.consecutive_no_miss_layers_ = 0;
    return target;
}

void Qwen4ExpBnb4Backend::Reset(Session& session) {
    ValidateSession(session);
    session.context_->synchronize();
    session.consecutive_no_miss_layers_ = 0;
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
    ple_next_ticket_.valid = false;
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
    // Exact device payload of the tensors this Session owns, matching
    // qwen4exp_session_memory's device reservation. Model weights, expert
    // cache and shared execution scratch are budgeted elsewhere.
    size_t cacheBytes = 0;
    for (const auto& cache : session->layerCaches_) {
        for (const Tensor* tensor : {&cache.k, &cache.v, &cache.indexer_k,
                                     &cache.indexer_compressed_k, &cache.linear_state,
                                     &cache.linear_conv_state, &cache.k_scale,
                                     &cache.v_scale}) {
            cacheBytes = aila::memory::checked_memory_sum(cacheBytes, tensor->size_bytes());
        }
    }
    session->cacheBytes_ = aila::memory::checked_memory_sum(
        cacheBytes, session->ple_state_.conv_state.size_bytes());
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

bool Qwen4ExpBnb4Backend::CanCaptureInteriorCheckpoint(
    const Session& session, int count, int checkpointAfter) const {
    if (count < 4 || count > 64 || checkpointAfter < 2 || checkpointAfter > count - 2 ||
        session.snapshots_.size() < 2 || count > session.Capacity() - session.CurrentLength() ||
        !session.embeddingOverridePositions_.empty() || session.mropePosT_.valid()) return false;
    // Match ForwardImpl's sparse-prefill predicate exactly. At the default ratio/budget,
    // 2052 is already sparse (513 complete blocks); the final dense position is 2051.
    const int ratio = cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4;
    const int blockTopK = (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) / ratio;
    return (session.CurrentLength() + count) / ratio <= blockTopK;
}

bool Qwen4ExpBnb4Backend::supports_interior_checkpoint(int count, int checkpoint_after) const {
    bool explicit_inline = false;
    const bool has_env = aila::env::read_flag_if_present("AILA_Q4E_INLINE_CHECKPOINT", &explicit_inline);
    if (has_env && !explicit_inline) return false;
    auto* self = const_cast<Qwen4ExpBnb4Backend*>(this);
    if (!defaultSession_ && context_) {
        self->DefaultSession(*context_);
    }
    if (!defaultSession_ || !CanCaptureInteriorCheckpoint(*defaultSession_, count, checkpoint_after)) return false;
    // X2 measured default: initial 30-token text prefill with a 26-token reusable prefix.
    // Explicit opt-in enables other structurally valid anchors, without a bitwise whitelist.
    return has_env || (defaultSession_->CurrentLength() == 0 && count == 30 && checkpoint_after == 26);
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
