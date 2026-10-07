#include "Qwen4ExpBnb4Backend.hpp"
#include "../ops/QsaIndexerOps.hpp"
#include "../ops/Qwen4ExpAttentionPair.hpp"
#include "../utils/EnvUtils.hpp"
#include "../profile/Profiling.hpp"
#include <algorithm>

using bf16 = sycl::ext::oneapi::bfloat16;

// Per-layer attention tile. The input/output banks belong to the entire routing
// window, while QKV, GDN intermediates and score scratch belong to this tile.
// All submitted work shares the in-order compute queue; no state is published
// until ForwardImpl completes the window. KV/GDN state is always session-owned.
void Qwen4ExpBnb4Backend::RunAttentionLayerTile(
    Context& ctx, Session& session, int L, Tensor& input, Tensor& output,
    int count, int start_pos, StateSnapshot* interior, int checkpoint_after, bool allowQ8) {
    auto& layer = layers_[L];
    auto& cache = session.layerCaches_[L];
    const int cached_len = start_pos + count;
    constexpr int full_rotary_dim = 64;
    static const bool s_small_rows_decode = aila::env::read_flag("AILA_Q4E_SMALL_ROWS_DECODE", false);
    const bool small_rows_decode = s_small_rows_decode && count >= 2 && count <= 4;
    auto project_prefill = [&](Bnb4BitLinear& projection, Tensor& x, Tensor& y) {
        if (small_rows_decode) projection.ForwardDecodeRows(ctx, linear_scratch_, x, y, count);
        else projection.forward(ctx, linear_scratch_, x, y, count);
    };
    const bool attention_q8 = allowQ8 && count == 1 && !interior && attentionNf4Q8Enabled_ &&
        !gpuNf4Q8ReferenceForTest_ && !decodeHcReferenceForTest_;
    auto project_decode = [&](Bnb4BitLinear& projection, Tensor& x, Tensor& y, int kind) {
        const int features = projection.in_features();
        if (attention_q8 && projection.SupportsDecodeQ8() && features > 0 && features % 64 == 0 &&
            buf_.attn_xq.valid() && buf_.attn_xs.valid() &&
            buf_.attn_xq.numel() >= features && buf_.attn_xs.numel() >= features / 64) {
            ops::quantize_moe_input_q8_64(ctx, x.data_as<bf16>(), buf_.attn_xq.data_as<int8_t>(),
                buf_.attn_xs.data_as<float>(), features);
            if (projection.TryForwardDecodeQ8(ctx, buf_.attn_xq.data_as<int8_t>(),
                buf_.attn_xs.data_as<float>(), y.data_as<bf16>())) {
                ++attentionNf4Q8ProjectionCounts_[kind];
                if (!attentionNf4Q8Logged_) {
                    AILA_LOG_INFO("[Q4E Attention NF4Q8] actual_enabled=1 layer=%d projection=%d input=%d output=%d activation_block=64",
                        L, kind, features, projection.out_features());
                    attentionNf4Q8Logged_ = true;
                }
                return;
            }
        }
        projection.ForwardDecodeRows(ctx, linear_scratch_, x, y, 1);
    };
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
            project_decode(layer.linear_all_proj, input, buf_.linear_all, 0);
            bf16* linear_all_ptr = static_cast<bf16*>(buf_.linear_all.data());
            const bf16* qkv_ptr = linear_all_ptr;
            const bf16* z_ptr = linear_all_ptr + linear_qkv_dim_;
            const bf16* a_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_;
            const bf16* b_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_ + linear_kv_heads_;

            static const bool s_gdn_history_fused = aila::env::read_flag("AILA_Q4E_GDN_HISTORY_FUSED", false);
            const bool defer_history = s_gdn_history_fused && allowQ8 && !attention_q8 && !interior &&
                !gpuNf4Q8ReferenceForTest_ && !decodeHcReferenceForTest_ &&
                ops::supports_default_grouped_linear_delta_fastpath(layer.geom.q_heads, layer.geom.v_heads,
                    layer.geom.head_k_dim, layer.geom.head_v_dim, layer.geom.kernel, layer.geom.conv_rows);
            const int history_row = layer.geom.conv_rows > 0 ? d_state.conv_head % layer.geom.conv_rows : 0;
            ops::run_linear_delta_decode_gpu(ctx, layer.geom, d_weights, d_state,
                                             qkv_ptr, z_ptr, a_ptr, b_ptr,
                                             static_cast<bf16*>(buf_.attn_out.data()), defer_history);

            if (defer_history && layer.linear_o_proj.TryForwardDecodeCommitHistory(ctx, buf_.attn_out, output,
                qkv_ptr, d_state.conv_state, layer.geom.qkv_dim, history_row)) {
                if (gdnHistoryFusionCount_++ == 0)
                    AILA_LOG_INFO("[Q4E GDN History Fused] actual_enabled=1 layer=%d channels=%d original_projection_math=1", L, layer.geom.qkv_dim);
            } else {
                // Includes LoRA/nonpacked fallback: the deferred ring write
                // still completes exactly once, before any subsequent reader.
                if (defer_history) ops::commit_linear_conv_history(ctx, qkv_ptr, layer.geom.qkv_dim,
                    d_state.conv_state, layer.geom.qkv_dim, layer.geom.conv_rows, history_row, 1);
                project_decode(layer.linear_o_proj, buf_.attn_out, output, 1);
            }
        } else {
            auto component_begin = [&] {
                if (!component_profile_active_for_test_) return std::chrono::steady_clock::time_point{};
                ctx.synchronize(); return std::chrono::steady_clock::now();
            };
            auto component_end = [&](int index, auto begin) {
                if (!component_profile_active_for_test_) return;
                ctx.synchronize();
                component_profile_ms_for_test_[index] += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            };
            auto component_start = component_begin();
            project_prefill(layer.linear_all_proj, input, buf_.linear_all);
            component_end(0,component_start);
            if (verification_linear_inputs_for_test_ && allowQ8 && count <= 4 && !interior)
                ctx.queue().memcpy(verification_linear_inputs_for_test_ + size_t(L) * 4 * linear_all_dim_,
                    buf_.linear_all.data(), size_t(count) * linear_all_dim_ * sizeof(bf16));
            const auto run_range = [&](int offset, int length) {
                const bf16* input = buf_.linear_all.data_as<bf16>() + static_cast<size_t>(offset) * linear_all_dim_;
                bf16* output = buf_.attn_out.data_as<bf16>() + static_cast<size_t>(offset) * linear_kv_dim_;
                if (gdn_chunk_mode_ > 0 && grouped_xmx_supported_ &&
                    ops::try_linear_delta_prefill_chunked(ctx, layer.geom, d_weights, d_state,
                        gdn_chunk_scratch_, input, output, length, gdn_chunk_mode_ == 2)) {
                    ++gdn_chunk_calls_;
                } else {
                    static const bool verify_gdn_order = aila::env::read_flag("AILA_Q4E_VERIFY_GDN_ORDER", false);
                    ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, d_weights, d_state,
                        input, output, length, verify_gdn_order && allowQ8 && count <= 4 && !interior);
                }
            };
            component_start = component_begin();
            if (interior) {
                run_range(0, checkpoint_after);
                auto& snapshot = interior->layers[L];
                snapshot.linear_conv_head = d_state.conv_head;
                // In-order D2H completes before the suffix may overwrite the recurrent state.
                ctx.memcpy_d2h_async(snapshot.linear_state_host.data(), cache.linear_state.data(),
                                     cache.linear_state.size_bytes());
                ctx.memcpy_d2h_async(snapshot.linear_conv_state_host.data(), cache.linear_conv_state.data(),
                                     cache.linear_conv_state.size_bytes());
                if (checkpoint_after < count) run_range(checkpoint_after, count - checkpoint_after);
            } else {
                run_range(0, count);
            }
            component_end(1,component_start);
            component_start = component_begin();
            project_prefill(layer.linear_o_proj, buf_.attn_out, output);
            component_end(2,component_start);
        }
        cache.linear_conv_head = d_state.conv_head;
        cache.device_state_dirty = d_state.device_state_dirty;
    } else {
        if (count == 1) {
            project_decode(layer.qkv_proj, input, buf_.full_qkv, 2);
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
                    DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, input, k_target,
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
                    DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, input, buf_.indexer_qk,
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

            project_decode(layer.o_proj, buf_.attn_out, output, 3);
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
                    DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, input, k_target,
                                count, k_dim, hidden_size_);
                } else {
                    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                    DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, input, buf_.indexer_qk,
                                count, indexer_out_dim, hidden_size_);
                }
            }

            project_prefill(layer.qkv_proj, input, buf_.full_qkv);
            static const bool verify_decode_prepare = aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_PREPARE", false);
            if (verify_decode_prepare && allowQ8 && !interior && count == 2 && start_pos > 0 && !sparse_prefill) {
                // The same per-row RMSNorm/RoPE arithmetic and KV writes as decode.
                // Reuse the packed projection and existing output banks; no USM scratch.
                for (int r = 0; r < count; ++r) {
                    bf16* packed = buf_.full_qkv.data_as<bf16>() + size_t(r) * (full_q_proj_dim_ + 2 * full_kv_dim_);
                    const int q_dim = full_q_heads_ * full_head_dim_;
                    auto qg = Tensor::view(ctx, packed, {1, full_q_proj_dim_});
                    auto k = Tensor::view(ctx, packed + full_q_proj_dim_, {1, full_kv_dim_});
                    auto v = Tensor::view(ctx, packed + full_q_proj_dim_ + full_kv_dim_, {1, full_kv_dim_});
                    auto q = Tensor::view(ctx, buf_.q.data_as<bf16>() + size_t(r) * q_dim, {1, q_dim});
                    auto z = Tensor::view(ctx, buf_.z.data_as<bf16>() + size_t(r) * q_dim, {1, q_dim});
                    ops::decode_prepare_qgkv_packed_partial(
                        ctx, qg, k, v, q, z, *layer.q_norm_weight, *layer.k_norm_weight,
                        cache.k, cache.v, start_pos + r, full_q_heads_, full_kv_heads_, full_head_dim_,
                        cfg_.rms_norm_eps, full_rotary_dim, cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                        (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                        (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                        (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                        session.mropePromptLength_, session.mropeTextPositionDelta_,
                        cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2]);
                }
            } else {
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
                project_prefill(layer.o_proj, buf_.attn_out, output);
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
                static const bool verify_decode_attention = aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_ATTN", false);
                static const bool verify_attention_pair = aila::env::read_flag("AILA_Q4E_VERIFY_ATTN_PAIR", false);
                static const int decode_window = aila::env::read_int_raw("AILA_ATTN_DECODE_WINDOW", 0);
                static const int decode_sink = aila::env::read_int_raw("AILA_ATTN_DECODE_SINK", 0);
                if (verify_decode_attention && verify_attention_pair && allowQ8 && count == 2 &&
                    start_pos+1 >= 1024 && !interior && decode_window <= 0 && decode_sink <= 0 &&
                    cache.k.dtype()==dnnl::memory::data_type::bf16 &&
                    buf_.incr_scores.numel()>=2*24*((session.Capacity()+127)/128)*272) {
                    static const bool logged=[] {
                        AILA_LOG_INFO("[Q4E VerifyAttentionPair] active=1 rows=2 bf16_kv_reuse=1 phase_scratch_reuse=1");
                        return true;
                    }();
                    (void)logged;
                    ops::q4e_attention_decode_pair(ctx,buf_.q,cache.k,cache.v,buf_.attn_out,
                        buf_.incr_scores,start_pos+1,buf_.z,
                        verification_attention_pair_bundle_ ? &*verification_attention_pair_bundle_ : nullptr);
                } else if (verify_decode_attention && allowQ8 && count <= 4 && !interior &&
                    decode_window <= 0 && decode_sink <= 0) {
                    // Verification queries share KV storage but each sees only its
                    // own causal prefix. Reuse decode's exact-merge and fused gate;
                    // the in-order queue safely reuses the single-row scratch.
                    const int q_dim = full_q_heads_ * full_head_dim_;
                    for (int row = 0; row < count; ++row) {
                        auto q_row = Tensor::view(ctx, buf_.q.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        auto z_row = Tensor::view(ctx, buf_.z.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        auto out_row = Tensor::view(ctx, buf_.attn_out.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        ops::attention_decode(ctx, q_row, cache.k, cache.v, out_row, buf_.decode_scores,
                            full_q_heads_, full_kv_heads_, full_head_dim_, start_pos + row + 1,
                            &buf_.decode_attn_partials, &z_row);
                    }
                } else if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores, count, full_q_heads_, full_kv_heads_, full_head_dim_, &buf_.z);
                } else {
                    ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores, count, start_pos, full_q_heads_, full_kv_heads_, full_head_dim_, session.Capacity(), &buf_.z);
                }
                project_prefill(layer.o_proj, buf_.attn_out, output);
            }
        }
    }
}

void Qwen4ExpBnb4Backend::RunAttentionWindow(
    Context& ctx, Session& session, int layer, int count, int start_pos,
    StateSnapshot* interior, int checkpoint_after, bool allowQ8) {
    if (prefill_attention_tile_ == 0 || count <= prefill_attention_tile_) {
        RunAttentionLayerTile(ctx, session, layer, buf_.x, buf_.y,
                              count, start_pos, interior, checkpoint_after, allowQ8);
        return;
    }
    for (int offset = 0; offset < count;) {
        int rows = std::min(prefill_attention_tile_, count - offset);
        // A one-row tail must not accidentally select decode window/sink rules.
        if (count - offset - rows == 1) --rows;
        Tensor input = Tensor::view(ctx, buf_.x.data_as<bf16>() + size_t(offset)*hidden_size_, {rows, hidden_size_});
        Tensor output = Tensor::view(ctx, buf_.y.data_as<bf16>() + size_t(offset)*hidden_size_, {rows, hidden_size_});
        const bool capture = interior && checkpoint_after > offset && checkpoint_after <= offset + rows;
        RunAttentionLayerTile(ctx, session, layer, input, output, rows, start_pos + offset,
                              capture ? interior : nullptr, capture ? checkpoint_after - offset : 0, false);
        offset += rows;
    }
}

// Keep each 4/2-token reuse unit intact when cutting compact output tiles.
// Descriptors and gate activations stay in their immutable full-window slices.
sycl::event Qwen4ExpBnb4Backend::RunCompactExpertDown(Context& ctx, int first, int acts,
    int pairs, int quads, const sycl::event& ready) {
    const int lengths[] = {quads*4,pairs*2,acts-quads*4-pairs*2};
    const int widths[] = {4,2,1};
    sycl::event last=ready;
    int offset=0;
    for(int section=0;section<3;++section)for(int done=0;done<lengths[section];) {
        const int n=std::min(moe_compact_tile_,lengths[section]-done);
        const int begin=first+offset+done;
        last=ops::moe_expert_major_prefill_down(ctx,expert_map_slot_dev_+begin,
            expert_map_token_dev_+begin,expert_map_rank_dev_+begin,nullptr,
            buf_.moe_h.data_as<bf16>(),buf_.moe_down_acc.data_as<float>(),n,
            cfg_.num_experts_per_tok,cfg_.moe_intermediate_size,hidden_size_,64,&last,
            expert_cache_.uses_nested_codes(),widths[section]==2?n/2:0,widths[section]==4?n/4:0,true);
        last=ops::moe_merge_compact_down(ctx,buf_.moe_down_acc.data_as<float>(),expert_map_token_dev_,
            buf_.moe_inverse_map.data_as<int>(),batch_weights_dev_,buf_.moe_window_sum.data_as<int64_t>(),buf_.moe_compact_invalid.data_as<int>(),
            begin,n,cfg_.num_experts_per_tok,hidden_size_,&last);
        done+=n;
        if(done==lengths[section])offset+=lengths[section];
    }
    return last;
}
