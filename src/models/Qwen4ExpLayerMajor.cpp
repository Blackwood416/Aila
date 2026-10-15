#include "Qwen4ExpBnb4Backend.hpp"
#include "../utils/EnvUtils.hpp"
#include "../profile/Profiling.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

// Same arithmetic as add_shared_expert_to_y() in Qwen4ExpBnb4Backend.cpp.
// That helper lives in an anonymous namespace in another translation unit, so
// the layer-major path carries its own copy; keep the two in sync.
void layer_major_add_shared_expert_to_y(
    Context& ctx, Tensor& y, Tensor& shared_out, Tensor& shared_gate,
    int rows, int hidden_size) {
    bf16* y_ptr = static_cast<bf16*>(y.data());
    const bf16* s_ptr = static_cast<const bf16*>(shared_out.data());
    const bf16* g_ptr = static_cast<const bf16*>(shared_gate.data());

    if ((hidden_size % 8) == 0) {
        using vec8 = sycl::vec<bf16, 8>;
        const int dim8 = hidden_size / 8;
        constexpr int wg = 64;
        const size_t num_wgs_per_row = (dim8 + wg - 1) / wg;

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
                            const float s_val = static_cast<float>(s_vec[v]);
                            const float y_val = static_cast<float>(y_vec[v]);
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
            const size_t r = idx[0] / hidden_size;
            const float g_val = static_cast<float>(g_ptr[r]);
            const float gate_act = 1.0f / (1.0f + sycl::native::exp(-g_val));
            const float s_val = static_cast<float>(s_ptr[idx[0]]);
            const float y_val = static_cast<float>(y_ptr[idx[0]]);
            y_ptr[idx[0]] = bf16(y_val + gate_act * s_val);
        });
    });
}

// Rows are copied verbatim into the four hyper-connection groups, matching the
// embedding replication in ForwardImpl. Used once per layer-major forward.
void layer_major_replicate_embedding(
    Context& ctx, const bf16* src, bf16* dst, int rows, int hidden) {
    if (rows <= 0) return;
    if ((hidden % 8) != 0) {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * hidden),
                [=](sycl::id<1> idx) {
                    const size_t t = idx[0] / hidden;
                    const size_t d = idx[0] % hidden;
                    const bf16 value = src[t * hidden + d];
                    for (int s = 0; s < 4; ++s) dst[t * 10240 + s * hidden + d] = value;
                });
        });
        return;
    }
    using vec8 = sycl::vec<bf16, 8>;
    const int dim8 = hidden / 8;
    constexpr int wg = 256;
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(rows), wg),
                              sycl::range<2>(1, wg)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                const size_t t = item.get_group(0);
                const int lid = static_cast<int>(item.get_local_id(1));
                for (int i = lid; i < dim8; i += wg) {
                    const vec8 value = *reinterpret_cast<const vec8*>(src + t * hidden + i * 8);
#pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        *reinterpret_cast<vec8*>(dst + t * 10240 + s * hidden + i * 8) = value;
                    }
                }
            });
    });
}

} // namespace

bool Qwen4ExpBnb4Backend::LayerMajorPrefillEligible(
    const Session& session, int count, const ForwardOptions& options) const {
    static const bool s_layer_major = aila::env::read_flag("AILA_Q4E_LAYER_MAJOR", false);
    if (!s_layer_major) return false;
    // Prototype shape: exactly two full 4096-token chunks. Every other shape
    // (including 1/3+ chunks, count != 8192) falls back to chunk-major in
    // ForwardImpl and is logged there once per process. TODO(layer-major):
    // generalize to N sub-chunks once the 8192 case is accepted.
    if (count != 8192) return false;
    if (options.decode_verification) return false;
    if (options.prefill_remaining < 0) return false;
    // Interior checkpoints are chunk-local; the layer-major pass does not
    // implement them yet. Fall back to chunk-major instead of dropping them.
    // TODO(layer-major): checkpoint the first 4096-row sub-chunk in place.
    if (options.checkpoint_after != 0) return false;
    if (moe_compact_tile_ != 0) return false;
    if (!adaptive_cache_.IsAllocated()) return false;
    if (cfg_.num_experts != 512) return false;
    if (!session.embeddingOverridePositions_.empty() || session.mropePosT_.valid()) return false;
    if (session.CurrentLength() + count > max_seq_len_) return false;
    // TODO(layer-major): reject while the MTP runtime is active (no backend
    // visibility yet) and honor cancellation between layers.
    return true;
}

void Qwen4ExpBnb4Backend::RunAttentionWindowAt(
    Context& ctx, Session& session, int layer, int count, int start_pos,
    int row_offset, bool allowQ8) {
    bf16* x_base = buf_.x.data_as<bf16>() + static_cast<size_t>(row_offset) * hidden_size_;
    bf16* y_base = buf_.y.data_as<bf16>() + static_cast<size_t>(row_offset) * hidden_size_;
    if (prefill_attention_tile_ == 0 || count <= prefill_attention_tile_) {
        Tensor input = Tensor::view(ctx, x_base, {count, hidden_size_});
        Tensor output = Tensor::view(ctx, y_base, {count, hidden_size_});
        RunAttentionLayerTile(ctx, session, layer, input, output, count,
                              start_pos, nullptr, 0, allowQ8);
        return;
    }
    for (int offset = 0; offset < count;) {
        int rows = std::min(prefill_attention_tile_, count - offset);
        // A one-row tail must not accidentally select decode window/sink rules.
        if (count - offset - rows == 1) --rows;
        Tensor input = Tensor::view(ctx, x_base + static_cast<size_t>(offset) * hidden_size_,
                                    {rows, hidden_size_});
        Tensor output = Tensor::view(ctx, y_base + static_cast<size_t>(offset) * hidden_size_,
                                     {rows, hidden_size_});
        RunAttentionLayerTile(ctx, session, layer, input, output, rows,
                              start_pos + offset, nullptr, 0, allowQ8);
        offset += rows;
    }
}

void Qwen4ExpBnb4Backend::EnsureLayerMajorCarries(Context& ctx, int count) {
    const int top_k = cfg_.num_experts_per_tok;
    if (lm_hidden_.valid() &&
        lm_hidden_.numel() >= static_cast<int64_t>(count) * 10240) {
        return;
    }
    // Growth is outside the decode hot path; drain before replacing carries.
    ctx.synchronize();
    lm_hidden_ = Tensor::allocate(ctx, {count, 10240}, dnnl::memory::data_type::bf16);
    lm_x_ = Tensor::allocate(ctx, {count, hidden_size_}, dnnl::memory::data_type::bf16);
    lm_y_ = Tensor::allocate(ctx, {count, hidden_size_}, dnnl::memory::data_type::bf16);
    lm_inject_ = Tensor::allocate(ctx, {count, 4}, dnnl::memory::data_type::f32);
    lm_moe_h_ = Tensor::allocate(ctx, {count, top_k, cfg_.moe_intermediate_size},
                                 dnnl::memory::data_type::bf16);
    lm_down_ = Tensor::allocate(ctx, {static_cast<int64_t>(count) * top_k, hidden_size_},
                                dnnl::memory::data_type::f32);
    lm_shared_out_ = Tensor::allocate(ctx, {count, hidden_size_},
                                      dnnl::memory::data_type::bf16);
    lm_shared_gate_ = Tensor::allocate(ctx, {count}, dnnl::memory::data_type::bf16);
    const uint64_t bytes =
        uint64_t(count) * 10240 * 2 +
        uint64_t(count) * hidden_size_ * 2 * 3 +
        uint64_t(count) * 4 * 4 +
        uint64_t(count) * top_k * cfg_.moe_intermediate_size * 2 +
        uint64_t(count) * top_k * hidden_size_ * 4 +
        uint64_t(count) * 2;
    AILA_LOG_INFO("[Q4E LayerMajor] carries count=%d bytes=%.3f GiB "
                  "(hidden/x/y/inject/moe_h/down/shared)",
                  count, static_cast<double>(bytes) / double(1ULL << 30));
}

void Qwen4ExpBnb4Backend::EnsureLayerMajorMaps(Context& ctx, int count) {
    if (layer_major_map_capacity_ >= count || runtime_seq_capacity_ >= count) return;
    const int top_k = cfg_.num_experts_per_tok;
    const size_t map_entries = static_cast<size_t>(count) * top_k;
    ctx.synchronize();
    auto grow_host_ints = [&](int*& ptr) {
        if (ptr) ctx.free_host(ptr);
        ptr = static_cast<int*>(ctx.alloc_host(map_entries * sizeof(int)));
    };
    grow_host_ints(router_host_ids_);
    if (router_host_weights_) ctx.free_host(router_host_weights_);
    router_host_weights_ = static_cast<float*>(ctx.alloc_host(map_entries * sizeof(float)));
    if (expert_map_token_host_) ctx.free_host(expert_map_token_host_);
    if (expert_map_rank_host_) ctx.free_host(expert_map_rank_host_);
    if (expert_map_slot_host_) ctx.free_host(expert_map_slot_host_);
    if (expert_map_token_dev_) ctx.free_device(expert_map_token_dev_);
    if (expert_map_rank_dev_) ctx.free_device(expert_map_rank_dev_);
    if (expert_map_slot_dev_) ctx.free_device(expert_map_slot_dev_);
    expert_map_token_host_ = static_cast<int*>(ctx.alloc_host(map_entries * sizeof(int)));
    expert_map_rank_host_ = static_cast<int*>(ctx.alloc_host(map_entries * sizeof(int)));
    expert_map_slot_host_ = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_host(map_entries * sizeof(ops::MoeExpertSlotDev)));
    expert_map_token_dev_ = static_cast<int*>(ctx.alloc_device(map_entries * sizeof(int)));
    expert_map_rank_dev_ = static_cast<int*>(ctx.alloc_device(map_entries * sizeof(int)));
    expert_map_slot_dev_ = static_cast<ops::MoeExpertSlotDev*>(
        ctx.alloc_device(map_entries * sizeof(ops::MoeExpertSlotDev)));
    layer_major_map_capacity_ = count;
    AILA_LOG_INFO("[Q4E LayerMajor] expert map mirrors grown: count=%d entries=%zu cap=%d",
                  count, map_entries, runtime_seq_capacity_);
}

Tensor& Qwen4ExpBnb4Backend::ForwardLayerMajorPrefill(
    Context& ctx, Session& session, const int* tokenIdsDevice, int count,
    bool lastTokenOnly, const ForwardOptions& options) {
    ValidateSession(session);
    constexpr int kChunks = 2;
    const int chunk = count / kChunks;
    const int top_k = cfg_.num_experts_per_tok;
    const int start_pos = session.currentLength_;
    const int cached_len = start_pos + count;
    // The prototype only serves the normal Forward() path, which reads the
    // last row's logits. ForwardLegacyLogits has prefill_remaining < 0 and is
    // rejected by LayerMajorPrefillEligible().
    if (!lastTokenOnly) {
        throw std::logic_error("Qwen4Exp layer-major prototype requires last-token logits");
    }
    if (cached_len > max_seq_len_) {
        throw std::runtime_error("Qwen4ExpBnb4Backend: context length " +
                                 std::to_string(cached_len) + " exceeds max_seq_len " +
                                 std::to_string(max_seq_len_));
    }

    static const bool s_work_summary = aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    static const bool s_fuse_shared = aila::env::read_flag("AILA_Q4E_MOE_SHARED_FUSED", false);
    using prof_clock = std::chrono::steady_clock;
    const auto prof_t0 = prof_clock::now();
    auto prof_ms = [](prof_clock::time_point a, prof_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double prof_router_sync_ms = 0.0;
    double prof_experts_ms = 0.0;
    // v4 pipeline gate + module split (see the layer-major v4 plan). The module
    // timers only run on the gated arm; the ungated path is byte-identical.
    static const bool s_pipeline = aila::env::read_flag("AILA_Q4E_LAYER_MAJOR_PIPELINE", false);
    double pip_ple_ms = 0.0, pip_halves_ms = 0.0, pip_shared_ms = 0.0;
    double pip_mix_ms = 0.0, pip_head_ms = 0.0;
    const auto adaptive_stats0 = adaptive_cache_.Stats();
    const auto cold_stats0 = expert_executor_.stats();

    AILA_LOG_INFO("[Q4E LayerMajor] begin pos=%d count=%d chunk=%d need_logits=%d remaining=%d",
                  start_pos, count, chunk, int(options.need_logits), options.prefill_remaining);

    // The whole 8192-token window is one owner of the session state. A stale
    // PLE lookahead ticket from an earlier chunk-major forward is not reused
    // on this path (TODO: PLE lookahead across the next outer chunk).
    ple_next_ticket_.valid = false;
    // v3: standard capacity scratch stays at the 4096-row chunk size; only the
    // layer-major carries below span both halves.
    ensure_runtime_buffers(ctx, chunk);
    EnsureLayerMajorMaps(ctx, count);
    EnsureLayerMajorCarries(ctx, count);
    session.consecutive_no_miss_layers_ = 0;

    lm_ids_host_.resize(static_cast<size_t>(count));
    ctx.memcpy_d2h(lm_ids_host_.data(), tokenIdsDevice,
                   static_cast<size_t>(count) * sizeof(int));
    for (int c = 0; c < kChunks; ++c) {
        EmbedTokens(ctx, tokenIdsDevice + static_cast<size_t>(c) * chunk, chunk);
        layer_major_replicate_embedding(
            ctx, buf_.x.data_as<bf16>(),
            lm_hidden_.data_as<bf16>() + static_cast<size_t>(c) * chunk * 10240,
            chunk, hidden_size_);
    }

    // 1. 48 transformer layers, layer-major. Every non-expert module runs as
    //    two 4096-row halves with the standard scratch, exactly like the
    //    chunk-major path; only the MoE input/expert output and residual stack
    //    live in the 8192-row carries so one staging union serves both halves.
    for (int L = 0; L < cfg_.num_hidden_layers; ++L) {
        auto& layer = layers_[L];
        const auto layer_t0 = prof_clock::now();
        // TODO(layer-major): check the request cancel flag here once the
        // backend receives one; today cancellation is only seen between
        // backend calls, so the whole 8192-row forward is uninterruptible.

        // 1.0. PLE on layer 1, two 4096-token calls on the residual carry.
        if (L == 1) {
            const auto pip_t0 = s_pipeline ? prof_clock::now() : prof_clock::time_point{};
            for (int c = 0; c < kChunks; ++c) {
                const int off = c * chunk;
                PleTicket ticket = ple_.prepare(
                    ctx, session.ple_state_, lm_ids_host_.data() + off, chunk,
                    &session, session.generation(), start_pos + off);
                bf16* hidden = lm_hidden_.data_as<bf16>() + static_cast<size_t>(off) * 10240;
                ple_.consume(ctx, ticket, session.ple_state_, hidden, nullptr, hidden, 0, nullptr);
                ple_.commit(ticket, session.ple_state_);
            }
            if (s_pipeline) pip_ple_ms += prof_ms(pip_t0, prof_clock::now());
        }

        sycl::event route_events[kChunks];
        const auto pip_halves_t0 = s_pipeline ? prof_clock::now() : prof_clock::time_point{};
        for (int c = 0; c < kChunks; ++c) {
            const int off = c * chunk;
            const size_t hidden_off = static_cast<size_t>(off) * 10240;
            // Attn HC on this half, using the 4096-row scratch.
            ctx.queue().memcpy(buf_.hidden4.data(),
                               lm_hidden_.data_as<bf16>() + hidden_off,
                               static_cast<size_t>(chunk) * 10240 * sizeof(bf16));
            RunHyperConnection(ctx, layer.attn_hc, chunk, true);
            // Attention/GDN stay causal and read buf_.x in place.
            RunAttentionWindowAt(ctx, session, L, chunk, start_pos + off, 0, false);
            // MLP HC: residual add + norm -> MoE input, inject into scratch.
            RunHyperConnectionCombined(ctx, layer.mlp_hc, chunk, true,
                                       buf_.y.data_as<bf16>(),
                                       buf_.hc_inject_f32.data_as<float>());
            ctx.queue().memcpy(lm_x_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                               buf_.x.data(),
                               static_cast<size_t>(chunk) * hidden_size_ * sizeof(bf16));
            ctx.queue().memcpy(lm_hidden_.data_as<bf16>() + hidden_off, buf_.hidden4.data(),
                               static_cast<size_t>(chunk) * 10240 * sizeof(bf16));
            ctx.queue().memcpy(lm_inject_.data_as<float>() + static_cast<size_t>(off) * 4,
                               buf_.hc_inject_f32.data(),
                               static_cast<size_t>(chunk) * 4 * sizeof(float));
            // Router for this half; top-k writes into the pinned host arrays.
            Tensor x_view = Tensor::view(ctx, buf_.x.data_as<bf16>(), {chunk, hidden_size_});
            Tensor scores_view = Tensor::view(ctx, buf_.router_scores.data_as<bf16>(),
                                              {chunk, cfg_.num_experts});
            DenseLinear(ctx, layer.router_gate, layer.router_w, x_view, scores_view,
                        chunk, cfg_.num_experts, hidden_size_);
            route_events[c] = ops::router_softmax_topk_device(
                ctx, scores_view.data_as<bf16>(), nullptr,
                router_host_ids_ + static_cast<size_t>(off) * top_k,
                router_host_weights_ + static_cast<size_t>(off) * top_k,
                chunk, cfg_.num_experts, top_k, true);
        }
        if (s_pipeline) pip_halves_ms += prof_ms(pip_halves_t0, prof_clock::now());

        // Shared expert per half, from the carried MoE input into the shared
        // carry; submitted before the route wait to overlap cold staging.
        const auto pip_shared_t0 = s_pipeline ? prof_clock::now() : prof_clock::time_point{};
        for (int c = 0; c < kChunks; ++c) {
            const int off = c * chunk;
            Tensor x_view = Tensor::view(ctx,
                lm_x_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                {chunk, hidden_size_});
            Tensor gate_view = Tensor::view(ctx, lm_shared_gate_.data_as<bf16>() + off,
                                            {chunk, 1});
            Tensor gate_up_view = Tensor::view(ctx, buf_.shared_gate_up_buf.data_as<bf16>(),
                                               {chunk, 2 * cfg_.shared_expert_intermediate_size});
            Tensor inter_view = Tensor::view(ctx, buf_.shared_inter_buf.data_as<bf16>(),
                                             {chunk, cfg_.shared_expert_intermediate_size});
            Tensor out_view = Tensor::view(ctx,
                lm_shared_out_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                {chunk, hidden_size_});
            DenseLinear(ctx, layer.shared_expert_gate, layer.shared_expert_gate_w,
                        x_view, gate_view, chunk, 1, hidden_size_);
            layer.shared_gate_up_proj.forward(ctx, linear_scratch_, x_view, gate_up_view, chunk);
            ops::moe_silu_gate_up_mul_flat(ctx, gate_up_view.data_as<bf16>(),
                                           inter_view.data_as<bf16>(), chunk,
                                           cfg_.shared_expert_intermediate_size);
            layer.shared_down_proj.forward(ctx, linear_scratch_, inter_view, out_view, chunk);
        }
        if (s_pipeline) pip_shared_ms += prof_ms(pip_shared_t0, prof_clock::now());

        // 1.5. Wait for both route halves, then one shared-staging expert pass
        //      over the carried 8192-row window (v2 pipelining retained).
        {
            const auto wait_begin = prof_clock::now();
            for (int c = 0; c < kChunks; ++c) {
                if (route_events[c] != sycl::event{}) route_events[c].wait_and_throw();
            }
            prof_router_sync_ms += prof_ms(wait_begin, prof_clock::now());
        }
        const auto layer_stats0 = adaptive_cache_.Stats();
        const auto experts_t0 = prof_clock::now();
        RunAdaptivePrefillLayerMajor(
            ctx, L, s_fuse_shared, count,
            lm_x_.data_as<bf16>(), lm_y_.data_as<bf16>(),
            lm_moe_h_.data_as<bf16>(), lm_down_.data_as<float>(),
            lm_shared_out_.data_as<bf16>(), lm_shared_gate_.data_as<bf16>());
        const double layer_experts_ms = prof_ms(experts_t0, prof_clock::now());
        prof_experts_ms += layer_experts_ms;
        if (s_work_summary) {
            const auto layer_stats1 = adaptive_cache_.Stats();
            AILA_LOG_INFO("[Q4E LayerMajorLayer] layer=%d staged=%llu h2d_mb=%.3f fill_ms=%.3f "
                          "bank_wait_ms=%.3f experts_ms=%.3f wall_ms=%.3f",
                L,
                static_cast<unsigned long long>(layer_stats1.cold_experts_staged - layer_stats0.cold_experts_staged),
                (layer_stats1.cold_bytes_transferred - layer_stats0.cold_bytes_transferred) / 1048576.0,
                (layer_stats1.staging_fill_ns - layer_stats0.staging_fill_ns) * 1e-6,
                (layer_stats1.staging_wait_ns - layer_stats0.staging_wait_ns) * 1e-6,
                layer_experts_ms, prof_ms(layer_t0, prof_clock::now()));
        }

        // 1.6. Per-half shared add (unless fused) and MLP residual injection.
        const auto pip_mix_t0 = s_pipeline ? prof_clock::now() : prof_clock::time_point{};
        for (int c = 0; c < kChunks; ++c) {
            const int off = c * chunk;
            if (!s_fuse_shared) {
                Tensor y_view = Tensor::view(ctx,
                    lm_y_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                    {chunk, hidden_size_});
                Tensor so_view = Tensor::view(ctx,
                    lm_shared_out_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                    {chunk, hidden_size_});
                Tensor sg_view = Tensor::view(ctx, lm_shared_gate_.data_as<bf16>() + off,
                                              {chunk, 1});
                layer_major_add_shared_expert_to_y(ctx, y_view, so_view, sg_view,
                                                   chunk, hidden_size_);
            }
            ops::hc_inject_add(
                ctx, lm_hidden_.data_as<bf16>() + static_cast<size_t>(off) * 10240,
                lm_y_.data_as<bf16>() + static_cast<size_t>(off) * hidden_size_,
                lm_inject_.data_as<float>() + static_cast<size_t>(off) * 4,
                chunk, hidden_size_, 4);
        }
        if (s_pipeline) pip_mix_ms += prof_ms(pip_mix_t0, prof_clock::now());
        // Cumulative checkpoints so the module split survives a truncated log
        // tail (take per-layer deltas of consecutive rows).
        if (s_pipeline && s_work_summary) {
            AILA_LOG_INFO("[Q4E LayerMajorPipeline] progress layer=%d ple_ms=%.3f halves_ms=%.3f "
                          "shared_ms=%.3f route_wait_ms=%.3f experts_ms=%.3f mix_ms=%.3f wall_ms=%.3f",
                L, pip_ple_ms, pip_halves_ms, pip_shared_ms, prof_router_sync_ms,
                prof_experts_ms, pip_mix_ms, prof_ms(prof_t0, prof_clock::now()));
        }
    }

    // 2. Final mixer for the last 4096-row half only; the sampler reads the
    //    last row, and the mixer is row-independent.
    if (options.need_logits) {
        const auto pip_head_t0 = s_pipeline ? prof_clock::now() : prof_clock::time_point{};
        const int off = (kChunks - 1) * chunk;
        ctx.queue().memcpy(buf_.hidden4.data(),
                           lm_hidden_.data_as<bf16>() + static_cast<size_t>(off) * 10240,
                           static_cast<size_t>(chunk) * 10240 * sizeof(bf16));
        RunHyperConnection(ctx, mixer_, chunk, false);
        const bf16* last_x = buf_.x.data_as<bf16>() +
            static_cast<size_t>(chunk - 1) * hidden_size_;
        Tensor x_last = Tensor::view(ctx, const_cast<bf16*>(last_x), {1, hidden_size_});
        if (lmHeadQ8Enabled_) {
            ops::bf16_q8_head_forward(ctx, lm_head_q8_weight_, lm_head_q8_scale_,
                                      x_last, buf_.logits, lm_head_q8_input_,
                                      lm_head_q8_input_scale_, 1, hidden_size_, cfg_.vocab_size);
            ++lmHeadQ8ForwardCount_;
            lmHeadQ8RowCount_ += 1;
        } else {
            lm_head_.forward(ctx, x_last, buf_.logits, 1);
        }
        lastTokenLogits_ = Tensor::view(ctx, buf_.logits.data(),
                                        {1, static_cast<int64_t>(cfg_.vocab_size)},
                                        buf_.logits.dtype());
        (void)lastTokenOnly;
        if (s_pipeline) pip_head_ms += prof_ms(pip_head_t0, prof_clock::now());
    }

    if (options.snapshot_policy == SnapshotPolicy::ReusablePrefix) {
        CaptureSnapshot(ctx, session, cached_len, count, options.capture_snapshot);
    }
    session.currentLength_ = cached_len;
    if (options.prefill_remaining == 0) TrimDecodeRuntimeBuffers(ctx);

    if (s_work_summary) {
        const auto& a = adaptive_cache_.Stats();
        const auto& c = expert_executor_.stats();
        AILA_LOG_INFO("[Q4E AdaptiveWork] pos=%d count=%d ple_reused=0 submitted_ms=%.3f router_wait_ms=%.3f "
            "ple_host_ms=0.000 hc_host_ms=0.000 linear_host_ms=0.000 full_host_ms=0.000 "
            "experts_host_ms=%.3f head_host_ms=0.000 fill_ms=%.3f bank_wait_ms=%.3f swap_ms=%.3f "
            "cpu_compute_ms=%.3f cpu_join_ms=%.3f cpu_tasks=%llu hits=%llu misses=%llu staged=%llu h2d_mb=%.3f layer_major=1",
            start_pos, count, prof_ms(prof_t0, prof_clock::now()), prof_router_sync_ms, prof_experts_ms,
            (a.staging_fill_ns - adaptive_stats0.staging_fill_ns) * 1e-6,
            (a.staging_wait_ns - adaptive_stats0.staging_wait_ns) * 1e-6,
            (a.swap_host_ns - adaptive_stats0.swap_host_ns) * 1e-6,
            (c.cpu_compute_ns - cold_stats0.cpu_compute_ns) * 1e-6,
            (c.cpu_join_wait_ns - cold_stats0.cpu_join_wait_ns) * 1e-6,
            static_cast<unsigned long long>(c.cpu_tasks_completed - cold_stats0.cpu_tasks_completed),
            static_cast<unsigned long long>(a.cache_hits - adaptive_stats0.cache_hits),
            static_cast<unsigned long long>(a.cache_misses - adaptive_stats0.cache_misses),
            static_cast<unsigned long long>(a.cold_experts_staged - adaptive_stats0.cold_experts_staged),
            (a.cold_bytes_transferred - adaptive_stats0.cold_bytes_transferred) / 1048576.0);
    }
    {
        const auto& a_end = adaptive_cache_.Stats();
        AILA_LOG_INFO("[Q4E LayerMajor] done pos=%d count=%d wall_ms=%.3f experts_ms=%.3f "
                      "router_wait_ms=%.3f fill_ms=%.3f bank_wait_ms=%.3f staged=%llu h2d_mb=%.3f",
            start_pos, count, prof_ms(prof_t0, prof_clock::now()), prof_experts_ms,
            prof_router_sync_ms,
            (a_end.staging_fill_ns - adaptive_stats0.staging_fill_ns) * 1e-6,
            (a_end.staging_wait_ns - adaptive_stats0.staging_wait_ns) * 1e-6,
            static_cast<unsigned long long>(a_end.cold_experts_staged - adaptive_stats0.cold_experts_staged),
            (a_end.cold_bytes_transferred - adaptive_stats0.cold_bytes_transferred) / 1048576.0);
    }
    if (s_pipeline && s_work_summary) {
        AILA_LOG_INFO("[Q4E LayerMajorPipeline] window count=%d ple_ms=%.3f halves_ms=%.3f "
                      "shared_ms=%.3f route_wait_ms=%.3f experts_ms=%.3f mix_ms=%.3f head_ms=%.3f",
            count, pip_ple_ms, pip_halves_ms, pip_shared_ms, prof_router_sync_ms,
            prof_experts_ms, pip_mix_ms, pip_head_ms);
    }

    return options.need_logits ? lastTokenLogits_ : unusedLogits_;
}

// v2: layer-major expert pass. Every cold bank is staged once and consumed by
// two 4096-row halves, so the per-expert kernel shapes match the fast
// chunk-major path while the H2D volume stays at one union per layer. Resident
// hits are processed in bounded microbatches so one large launch cannot stall
// the staging ping-pong, and the next cold bank is staged before the current
// bank is consumed.
sycl::event Qwen4ExpBnb4Backend::RunAdaptivePrefillLayerMajor(
    Context& ctx, int layer, bool fuse_shared, int count,
    const bf16* x_base, bf16* y_base, bf16* moe_h_base, float* down_acc_base,
    const bf16* shared_out_base, const bf16* shared_gate_base) {
    const int top_k = cfg_.num_experts_per_tok;
    const int half = count / 2;
    const bf16* x_dev = x_base ? x_base : buf_.x.data_as<bf16>();
    bf16* y_dev = y_base ? y_base : buf_.y.data_as<bf16>();
    bf16* h_dev = moe_h_base ? moe_h_base : buf_.moe_h.data_as<bf16>();
    float* down_dev = down_acc_base ? down_acc_base : buf_.moe_down_acc.data_as<float>();
    const bf16* shared_out_dev = shared_out_base ? shared_out_base : buf_.shared_out_buf.data_as<bf16>();
    const bf16* shared_gate_dev = shared_gate_base ? shared_gate_base : buf_.shared_gate_buf.data_as<bf16>();
    adaptive_cache_.ObserveRouting(layer, router_host_ids_, count * top_k);
    if (export_enabled_) export_policy_.Observe(layer, router_host_ids_, count, top_k, true);

    std::array<std::vector<int>, 2> half_uses;
    half_uses[0].assign(cfg_.num_experts, 0);
    half_uses[1].assign(cfg_.num_experts, 0);
    for (int i = 0; i < half * top_k; ++i) {
        const int e = router_host_ids_[i];
        if (e < 0 || e >= cfg_.num_experts) throw std::runtime_error("Invalid Qwen4Exp route");
        ++half_uses[0][e];
    }
    for (int i = half * top_k; i < count * top_k; ++i) {
        const int e = router_host_ids_[i];
        if (e < 0 || e >= cfg_.num_experts) throw std::runtime_error("Invalid Qwen4Exp route");
        ++half_uses[1][e];
    }
    if (prefill_expert_uses_.size() < static_cast<size_t>(cfg_.num_experts))
        prefill_expert_uses_.assign(cfg_.num_experts, 0);
    std::fill(prefill_expert_uses_.begin(), prefill_expert_uses_.end(), 0);
    for (int e = 0; e < cfg_.num_experts; ++e)
        prefill_expert_uses_[e] = half_uses[0][e] + half_uses[1][e];

    prefill_unique_.clear();
    const auto& metadata = host_arena_.GetLayerNestedMetadata(layer);
    for (int e = 0; e < cfg_.num_experts; ++e) {
        const int slot = adaptive_cache_.GetSlot(layer, e);
        if (prefill_expert_uses_[e] && slot >= 0) {
            prefill_unique_.push_back(e);
            adaptive_prefill_slots_[e] = adaptive_cache_.GetResidentSlotDev(slot, &metadata);
        }
    }
    const int hits = static_cast<int>(prefill_unique_.size());
    for (int e = 0; e < cfg_.num_experts; ++e) {
        if (prefill_expert_uses_[e] && adaptive_cache_.GetSlot(layer, e) < 0)
            prefill_unique_.push_back(e);
    }
    const int total_unique = static_cast<int>(prefill_unique_.size());

    static const int xmx_rows = aila::env::read_int("AILA_Q4E_GROUPED_XMX_MIN_ROWS", 0);
    const int threshold = grouped_xmx_min_rows_override_ >= 0 ? grouped_xmx_min_rows_override_ : xmx_rows;
    const int min_rows = grouped_xmx_supported_ && threshold > 0
        ? threshold : std::numeric_limits<int>::max();
    static const int configured_batch = [] {
        const int raw = aila::env::read_int("AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS", 16);
        return std::clamp(raw, 1, 64);
    }();
    const int batch = std::max(1, std::min(configured_batch,
                                           adaptive_cache_.StagingSlotsPerBuffer()));
    static bool logged_batch = false;
    if (!logged_batch) {
        logged_batch = true;
        AILA_LOG_INFO("[Q4E LayerMajor] expert batch=%d configured=%d cache_slots=%d",
                      batch, configured_batch, adaptive_cache_.StagingSlotsPerBuffer());
    }
    static const bool s_pipeline = aila::env::read_flag("AILA_Q4E_LAYER_MAJOR_PIPELINE", false);
    static bool logged_pipeline = false;
    if (s_pipeline && !logged_pipeline) {
        logged_pipeline = true;
        AILA_LOG_INFO("[Q4E LayerMajor] pipeline=1 depth=2 (cold head staged before the resident phase)");
    }

    int act_start = 0, tile_start = 0;
    auto run_batch = [&](int firstUnique, int numUnique, const sycl::event& ready) {
        sycl::event out = ready;
        for (int h = 0; h < 2; ++h) {
            int acts = act_start, tiles = tile_start;
            out = RunGroupedExperts(ctx, layer, half, firstUnique, numUnique,
                                    acts, tiles, min_rows, out,
                                    h * half * top_k, h * half,
                                    half_uses[h].data(),
                                    x_dev, h_dev, down_dev);
            if (acts < act_start) throw std::logic_error("Qwen4Exp layer-major act bookkeeping regressed");
            act_start = acts;
            tile_start = tiles;
        }
        return out;
    };

    struct PendingStage {
        aila::StagingBatch batch{};
        int first = 0;
        int n = 0;
        bool valid = false;
    };
    auto stage_one = [&](int first, int n) {
        auto staged = adaptive_cache_.StageColdExpertsAsync(
            host_arena_, *copy_q_, layer, prefill_unique_.data() + first, n,
            ping_pong_index_, &expert_executor_);
        ping_pong_index_ ^= 1;
        for (int j = 0; j < n; ++j) {
            adaptive_prefill_slots_[prefill_unique_[first + j]] =
                adaptive_cache_.GetStagingSlotDev(staged.ping_pong_index, j, &metadata);
        }
        PendingStage pending;
        pending.batch = staged;
        pending.first = first;
        pending.n = n;
        pending.valid = true;
        return pending;
    };

    const auto* resident_ready = adaptive_cache_.ResidentReady();
    sycl::event last = resident_ready ? *resident_ready : sycl::event{};

    // v4 (AILA_Q4E_LAYER_MAJOR_PIPELINE=1): the union is already known here, so
    // both ping-pong banks can be primed with the head cold batches *before* the
    // resident-hit microbatches are submitted. The head fill + H2D then overlap
    // the hit kernels instead of sitting on the critical path of the first cold
    // consumption. The loop below is the v3 one-deep look-ahead schedule, so an
    // unprimed (ungated) call keeps its exact stage/consume order.
    PendingStage fifo[2];
    int fifo_count = 0, fifo_head = 0, stage_next = hits;
    double head_stage_ms = 0.0;
    auto push_stage = [&]() {
        const int n = std::min(batch, total_unique - stage_next);
        fifo[(fifo_head + fifo_count) & 1] = stage_one(stage_next, n);
        stage_next += n;
        ++fifo_count;
    };
    if (s_pipeline) {
        const auto head_t0 = std::chrono::steady_clock::now();
        while (fifo_count < 2 && stage_next < total_unique) push_stage();
        head_stage_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - head_t0).count();
    }
    for (int start = 0; start < hits; start += batch) {
        last = run_batch(start, std::min(batch, hits - start), last);
    }
    if (hits) adaptive_cache_.RecordResidentConsumer(last);
    while (fifo_count > 0 || stage_next < total_unique) {
        while (fifo_count < 2 && stage_next < total_unique) push_stage();
        const PendingStage cur = fifo[fifo_head];
        fifo_head = (fifo_head + 1) & 1;
        --fifo_count;
        sycl::event consumed = run_batch(cur.first, cur.n, cur.batch.ready_event);
        adaptive_cache_.RecordStagingConsumer(cur.batch.ping_pong_index, consumed);
        last = consumed;
    }
    if (act_start != count * top_k) {
        throw std::logic_error("Incomplete Qwen4Exp layer-major routing map");
    }

    // One rank-ordered merge per 4096-row half, exactly like the chunk-major
    // path. All down rows are already written; each token is merged once.
    for (int h = 0; h < 2; ++h) {
        const int route_base = h * half * top_k;
        const int token_base = h * half;
        ctx.queue().memcpy(batch_weights_dev_, router_host_weights_ + route_base,
                           static_cast<size_t>(half) * top_k * sizeof(float));
        const size_t down_off = static_cast<size_t>(route_base) * hidden_size_;
        const size_t y_off = static_cast<size_t>(token_base) * hidden_size_;
        const sycl::event dep = last;
        last = ops::moe_ordered_accumulate_down_rows(
            ctx, down_dev + down_off, batch_weights_dev_,
            y_dev + y_off, half, top_k, hidden_size_,
            fuse_shared ? shared_out_dev + y_off : nullptr,
            fuse_shared ? shared_gate_dev + token_base : nullptr,
            &dep, nullptr);
    }
    static const bool s_work_summary = aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    if (s_pipeline && s_work_summary) {
        AILA_LOG_INFO("[Q4E LayerMajorPipeline] layer=%d hits=%d cold=%d batch=%d stage_head_ms=%.3f",
                      layer, hits, total_unique - hits, batch, head_stage_ms);
    }
    return last;
}
