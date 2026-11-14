#include "Qwen4ExpBnb4Backend.hpp"
#include "Qwen4ExpTokenGraphStageProf.hpp"
#include "../ops/AttentionOps.hpp"
#include "../ops/QsaIndexerOps.hpp"
#include "../ops/Qwen4ExpAttentionPair.hpp"
#include "../utils/EnvUtils.hpp"
#include "../profile/Profiling.hpp"
#include <algorithm>
#include <exception>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr int kQsaGraphLayer = 3;
constexpr int kQsaGraphLayerOrdinal = 0;
constexpr int kQsaGraphBlockTopK = 512;
constexpr int kQsaGraphCompressRatio = 4;
constexpr int kQsaGraphHeadDim = 256;
constexpr int kQsaGraphIndexerHeadDim = 128;
constexpr int kQsaGraphIndexerQHeads = 4;
constexpr int kQsaGraphIndexerKvHeads = 1;
constexpr uint64_t kQsaGraphSchemaFlags =
    (uint64_t{1} << static_cast<unsigned>(aila::TokenGraphBody::QsaLayerProof)) |
    (uint64_t{1} << (8 + static_cast<unsigned>(
        aila::TokenGraphAttention::QsaSparse512BlockNative)));

bool qsa_layer_graph_enabled() {
    static const bool enabled =
        aila::env::read_flag("AILA_Q4E_QSA_LAYER_GRAPH", false);
    return enabled;
}

bool qsa_layer_graph_trace_enabled() {
    static const bool enabled =
        aila::env::read_flag("AILA_Q4E_QSA_LAYER_GRAPH_TRACE", false);
    return enabled;
}

bool same_token_graph_key(const aila::TokenGraphKey& left,
                          const aila::TokenGraphKey& right) {
    return left.backend == right.backend &&
           left.session == right.session &&
           left.session_instance_id == right.session_instance_id &&
           left.session_generation == right.session_generation &&
           left.session_state_generation == right.session_state_generation &&
           left.runtime_generation == right.runtime_generation &&
           left.runtime_capacity == right.runtime_capacity &&
           left.residency_generation == right.residency_generation &&
           left.descriptor_generation == right.descriptor_generation &&
           left.ple_workspace_generation == right.ple_workspace_generation &&
           left.ple_graph_view_generation == right.ple_graph_view_generation &&
           left.topology_generation == right.topology_generation &&
           left.topology_flags == right.topology_flags &&
           left.body == right.body &&
           left.attention == right.attention;
}

} // namespace

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
    const bool small_rows_decode =
        (s_small_rows_decode || verify_rows_alignment_ || verify_decode_kernels_) &&
        count >= 2 && count <= 4;
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
        const bool use_device_conv_head =
            count == 1 && session.graph_gdn_state_enabled_;
        if (use_device_conv_head) {
            // Device authority is sufficient for steady decode: the host
            // mirror is intentionally stale until a checkpoint boundary.
            if (!GdnDeviceHeadReadable(cache)) {
                throw std::invalid_argument(
                    "Qwen4Exp graph GDN session has no readable device head "
                    "before decode");
            }
            d_state.conv_head_dev = cache.linear_conv_head_dev;
        }

        if (count == 1) {
            project_decode(layer.linear_all_proj, input, buf_.linear_all, 0);
            bf16* linear_all_ptr = static_cast<bf16*>(buf_.linear_all.data());
            const bf16* qkv_ptr = linear_all_ptr;
            const bf16* z_ptr = linear_all_ptr + linear_qkv_dim_;
            const bf16* a_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_;
            const bf16* b_ptr = linear_all_ptr + linear_qkv_dim_ + linear_z_dim_ + linear_kv_heads_;

            static const bool s_gdn_history_fused = aila::env::read_flag("AILA_Q4E_GDN_HISTORY_FUSED", true);
            const bool defer_history =
                s_gdn_history_fused && allowQ8 && !use_device_conv_head &&
                !attention_q8 && !interior &&
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
            static const bool s_gdn_history_fused = aila::env::read_flag("AILA_Q4E_GDN_HISTORY_FUSED", true);
            static const bool verify_gdn_order = aila::env::read_flag("AILA_Q4E_VERIFY_GDN_ORDER", false);
            const bool defer_verify_history =
                s_gdn_history_fused && small_rows_decode && allowQ8 && !interior &&
                !verify_gdn_order && !gpuNf4Q8ReferenceForTest_ && !decodeHcReferenceForTest_ &&
                ops::supports_default_grouped_linear_delta_fastpath(layer.geom.q_heads, layer.geom.v_heads,
                    layer.geom.head_k_dim, layer.geom.head_v_dim, layer.geom.kernel, layer.geom.conv_rows);
            const int initial_conv_head = layer.geom.conv_rows > 0 ? d_state.conv_head % layer.geom.conv_rows : 0;
            const int save_linear_rows = std::max(1, count - 1);
            bf16* save_linear_row0 = (verification_linear_inputs_for_test_ && allowQ8 && count <= 4 && !interior)
                ? (verification_linear_inputs_for_test_ + size_t(L) * 4 * linear_all_dim_) : nullptr;
            if (save_linear_row0 && (!defer_verify_history || count < 2 || count > 4)) {
                ctx.queue().memcpy(save_linear_row0, buf_.linear_all.data(),
                    size_t(count) * linear_all_dim_ * sizeof(bf16));
                save_linear_row0 = nullptr;
            }
            const auto run_range = [&](int offset, int length, bool defer_hist) {
                const bf16* input = buf_.linear_all.data_as<bf16>() + static_cast<size_t>(offset) * linear_all_dim_;
                bf16* output = buf_.attn_out.data_as<bf16>() + static_cast<size_t>(offset) * linear_kv_dim_;
                if (gdn_chunk_mode_ > 0 && grouped_xmx_supported_ && (count > 64 || !cold_epoch_enabled_) &&
                    ops::try_linear_delta_prefill_chunked(ctx, layer.geom, d_weights, d_state,
                        gdn_chunk_scratch_, input, output, length, gdn_chunk_mode_ == 2)) {
                    ++gdn_chunk_calls_;
                } else {
                    ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, d_weights, d_state,
                        input, output, length, verify_gdn_order && allowQ8 && count <= 4 && !interior,
                        false, defer_hist);
                }
            };
            component_start = component_begin();
            if (interior) {
                run_range(0, checkpoint_after, false);
                auto& snapshot = interior->layers[L];
                snapshot.linear_conv_head = d_state.conv_head;
                // In-order D2H completes before the suffix may overwrite the recurrent state.
                ctx.memcpy_d2h_async(snapshot.linear_state_host.data(), cache.linear_state.data(),
                                     cache.linear_state.size_bytes());
                ctx.memcpy_d2h_async(snapshot.linear_conv_state_host.data(), cache.linear_conv_state.data(),
                                     cache.linear_conv_state.size_bytes());
                if (checkpoint_after < count) run_range(checkpoint_after, count - checkpoint_after, false);
            } else {
                run_range(0, count, defer_verify_history);
            }
            component_end(1,component_start);
            component_start = component_begin();
            if (defer_verify_history && layer.linear_o_proj.TryForwardDecodeCommitHistory(
                    ctx, buf_.attn_out, output, buf_.linear_all.data_as<bf16>(), d_state.conv_state,
                    layer.geom.qkv_dim, initial_conv_head, count, linear_all_dim_,
                    layer.geom.conv_rows, save_linear_row0,
                    save_linear_row0 ? save_linear_rows * linear_all_dim_ : 0)) {
                // Fused multi-row history commit + optional verification prefix save.
            } else {
                if (defer_verify_history) {
                    if (save_linear_row0) {
                        ctx.queue().memcpy(save_linear_row0, buf_.linear_all.data(),
                            size_t(save_linear_rows) * linear_all_dim_ * sizeof(bf16));
                    }
                    ops::commit_linear_conv_history(ctx, buf_.linear_all.data_as<bf16>(), linear_all_dim_,
                        d_state.conv_state, layer.geom.qkv_dim, layer.geom.conv_rows, initial_conv_head, count);
                }
                project_prefill(layer.linear_o_proj, buf_.attn_out, output);
            }
            component_end(2,component_start);
        }
        if (use_device_conv_head) {
            cache.linear_conv_head_device_authority = true;
            cache.linear_conv_head_device_synced = false;
            cache.device_state_dirty = true;
        } else {
            cache.linear_conv_head = d_state.conv_head;
            if (cache.linear_conv_head_dev != nullptr) {
                cache.linear_conv_head_device_authority = false;
                cache.linear_conv_head_device_synced = false;
            }
            cache.device_state_dirty = d_state.device_state_dirty;
        }
    } else {
        if (count == 1) {
            if (L == kQsaGraphLayer && qsa_layer_graph_enabled() &&
                ForwardQsaLayerGraph(ctx, session, input, output, start_pos)) {
                return;
            }
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
                cfg_.rope.mrope_section[2],
                cache.k_scale.valid() ? &cache.k_scale : nullptr,
                cache.v_scale.valid() ? &cache.v_scale : nullptr);

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
                active_blocks, num_active_blocks,
                cache.k_scale.valid() ? &cache.k_scale : nullptr,
                cache.v_scale.valid() ? &cache.v_scale : nullptr);

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
                    if (verify_rows_alignment_) {
                        // Verify-path alignment: one BF16 GEMV per row so the
                        // window matches sequential B1 decode bit-for-bit.
                        for (int r = 0; r < count; ++r) {
                            auto in_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                            auto k_row = Tensor::view(ctx, cache.indexer_k.data_as<bf16>() +
                                static_cast<size_t>(start_pos + r) * k_dim, {1, k_dim});
                            DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, in_row, k_row,
                                        1, k_dim, hidden_size_);
                        }
                    } else {
                        Tensor k_target = Tensor::view(
                            ctx, cache.indexer_k.data_as<bf16>() + static_cast<size_t>(start_pos) * k_dim,
                            {count, k_dim});
                        DenseLinear(ctx, layer.index_k_proj, &layer.index_k_weight, input, k_target,
                                    count, k_dim, hidden_size_);
                    }
                } else {
                    const int indexer_out_dim = (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) * cfg_.indexer_head_dim;
                    if (verify_rows_alignment_) {
                        for (int r = 0; r < count; ++r) {
                            auto in_row = Tensor::view(ctx, input.data_as<bf16>() + size_t(r) * hidden_size_, {1, hidden_size_});
                            auto out_row = Tensor::view(ctx, buf_.indexer_qk.data_as<bf16>() +
                                size_t(r) * indexer_out_dim, {1, indexer_out_dim});
                            DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, in_row, out_row,
                                        1, indexer_out_dim, hidden_size_);
                        }
                    } else {
                        DenseLinear(ctx, layer.index_qk_proj, layer.index_qk_weight, input, buf_.indexer_qk,
                                    count, indexer_out_dim, hidden_size_);
                    }
                }
            }

            project_prefill(layer.qkv_proj, input, buf_.full_qkv);
            static const bool verify_decode_prepare = aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_PREPARE", true);
            if (verify_decode_prepare && allowQ8 && !interior && count >= 2 && count <= 4 && start_pos > 0) {
                // The same per-row RMSNorm/RoPE arithmetic and KV writes as decode,
                // launched in one multi-row grid.
                bf16* packed = buf_.full_qkv.data_as<bf16>();
                const int qkv_stride = full_q_proj_dim_ + 2 * full_kv_dim_;
                const int q_dim = full_q_heads_ * full_head_dim_;
                auto qg = Tensor::view(ctx, packed, {count, full_q_proj_dim_});
                auto k = Tensor::view(ctx, packed + full_q_proj_dim_, {count, full_kv_dim_});
                auto v = Tensor::view(ctx, packed + full_q_proj_dim_ + full_kv_dim_, {count, full_kv_dim_});
                auto q = Tensor::view(ctx, buf_.q.data_as<bf16>(), {count, q_dim});
                auto z = Tensor::view(ctx, buf_.z.data_as<bf16>(), {count, q_dim});
                ops::decode_prepare_qgkv_packed_partial(
                    ctx, qg, k, v, q, z, *layer.q_norm_weight, *layer.k_norm_weight,
                    cache.k, cache.v, start_pos, full_q_heads_, full_kv_heads_, full_head_dim_,
                    cfg_.rms_norm_eps, full_rotary_dim, cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
                    (session.mropePosT_.valid() ? session.mropePosT_.data_as<int>() : nullptr),
                    (session.mropePosH_.valid() ? session.mropePosH_.data_as<int>() : nullptr),
                    (session.mropePosW_.valid() ? session.mropePosW_.data_as<int>() : nullptr),
                    session.mropePromptLength_, session.mropeTextPositionDelta_,
                    cfg_.rope.mrope_section[0], cfg_.rope.mrope_section[1], cfg_.rope.mrope_section[2],
                    cache.k_scale.valid() ? &cache.k_scale : nullptr,
                    cache.v_scale.valid() ? &cache.v_scale : nullptr,
                    count, qkv_stride);
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
                    ops::copy_to_cache(ctx, buf_.k, cache.k, count, start_pos, full_kv_heads_, full_head_dim_, session.Capacity(),
                                       cache.k_scale.valid() ? &cache.k_scale : nullptr);
                    ops::copy_to_cache(ctx, buf_.v, cache.v, count, start_pos, full_kv_heads_, full_head_dim_, session.Capacity(),
                                       cache.v_scale.valid() ? &cache.v_scale : nullptr);
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

                    // Dynamic query tiling: the block-score buffer bounds how
                    // many rows one tile may score at this window's block count.
                    // Growing the tile toward that bound cuts the three
                    // launches per tile; an explicit AILA_QSA_QUERY_TILE still
                    // wins because it may deliberately shrink the tile.
                    const int window_max_blocks = std::max(1, (start_pos + count) / compress_ratio);
                    const int max_tile_rows = static_cast<int>(buf_.qsa_tile_active_blocks.shape(0));
                    const int score_fit_rows = static_cast<int>(
                        static_cast<size_t>(buf_.qsa_tile_block_scores.numel()) /
                        static_cast<size_t>(window_max_blocks));
                    static const int s_qsa_query_tile_env =
                        aila::env::read_int_raw("AILA_QSA_QUERY_TILE", 0);
                    const int query_tile = s_qsa_query_tile_env > 0
                        ? std::min(qsa_query_tile_, max_tile_rows)
                        : std::clamp(score_fit_rows, 16, std::min(max_tile_rows, 1024));
                    for (int tile_offset = 0; tile_offset < count; tile_offset += query_tile) {
                        const int tile_count = std::min(query_tile, count - tile_offset);
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

                        static const bool s_skip_dense_scores =
                            aila::env::read_flag("AILA_QSA_SKIP_DENSE_SCORES", true);
                        if (!s_skip_dense_scores || tile_max_blocks > block_topk) {
                            ops::qsa_compute_block_scores_compressed_batched(
                                ctx, q_tile, cache.indexer_compressed_k, buf_.qsa_tile_block_scores,
                                tile_count, tile_max_blocks, compress_ratio, tile_start_pos);
                        }

                        ops::qsa_select_topk_blocks_batched(
                            ctx, buf_.qsa_tile_block_scores, buf_.qsa_tile_active_blocks,
                            tile_count, tile_max_blocks, tile_start_pos, block_topk, compress_ratio,
                            &valid_counts_tile);

                        static const bool verify_decode_attention =
                            aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_ATTN", true);
                        static const bool s_qsa_block_native =
                            aila::env::read_flag("AILA_QSA_BLOCK_NATIVE", true);
                        if (verify_decode_attention && allowQ8 && !interior && count >= 2 && count <= 4 &&
                            s_qsa_block_native && !verify_rows_alignment_) {
                            const size_t block_stride = static_cast<size_t>(block_topk + 4);
                            for (int row = 0; row < tile_count; ++row) {
                                const int row_len = tile_start_pos + row + 1;
                                const int row_blocks = row_len / compress_ratio;
                                const int row_active_blocks = std::min(row_blocks, block_topk);
                                const int* row_blocks_ptr = buf_.qsa_tile_active_blocks.data_as<int>() +
                                    static_cast<size_t>(row) * block_stride;
                                auto q_row = Tensor::view(
                                    ctx, q_full_tile.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});
                                auto z_row = Tensor::view(
                                    ctx, z_tile.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});
                                auto out_row = Tensor::view(
                                    ctx, attn_out_tile.data_as<bf16>() + static_cast<size_t>(row) * q_dim, {1, q_dim});
                                ops::attention_decode(
                                    ctx, q_row, cache.k, cache.v, out_row, buf_.decode_scores,
                                    full_q_heads_, full_kv_heads_, full_head_dim_, row_len,
                                    &buf_.decode_attn_partials, &z_row,
                                    nullptr, 0, row_blocks_ptr, row_active_blocks,
                                    cache.k_scale.valid() ? &cache.k_scale : nullptr,
                                    cache.v_scale.valid() ? &cache.v_scale : nullptr);
                            }
                        } else {
                            ops::attention_prefill_qsa_batched(
                                ctx, q_full_tile, cache.k, cache.v, buf_.qsa_tile_active_blocks, attn_out_tile,
                                tile_count, tile_start_pos, full_q_heads_, full_kv_heads_, full_head_dim_,
                                block_topk, compress_ratio, &z_tile, &valid_counts_tile,
                                &cache.k_scale, &cache.v_scale);
                        }
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
                            buf_.indexer_active_tokens.data_as<int>(), attention_active_tokens,
                            nullptr, 0,
                            cache.k_scale.valid() ? &cache.k_scale : nullptr,
                            cache.v_scale.valid() ? &cache.v_scale : nullptr);
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
                static const bool verify_decode_attention = aila::env::read_flag("AILA_Q4E_VERIFY_DECODE_ATTN", true);
                static const bool verify_attention_pair = aila::env::read_flag("AILA_Q4E_VERIFY_ATTN_PAIR", true);
                static const int decode_window = aila::env::read_int_raw("AILA_ATTN_DECODE_WINDOW", 0);
                static const int decode_sink = aila::env::read_int_raw("AILA_ATTN_DECODE_SINK", 0);
                if (verify_decode_attention && verify_attention_pair && allowQ8 && count >= 2 && count <= 4 &&
                    start_pos + 1 >= 1 && !interior && decode_window <= 0 && decode_sink <= 0 &&
                    cache.k.dtype()==dnnl::memory::data_type::bf16 &&
                    (start_pos + count - 1 < 1024 || buf_.incr_scores.numel()>=2*24*((session.Capacity()+127)/128)*272) &&
                    !verify_rows_alignment_) {
                    static const bool logged=[] {
                        AILA_LOG_INFO("[Q4E VerifyAttentionPair] active=1 rows=2 bf16_kv_reuse=1 phase_scratch_reuse=1");
                        return true;
                    }();
                    (void)logged;
                    const int q_dim = full_q_heads_ * full_head_dim_;
                    int row = 0;
                    for (; row + 1 < count; row += 2) {
                        auto q_pair = Tensor::view(ctx, buf_.q.data_as<bf16>() + size_t(row) * q_dim, {2, full_q_heads_, full_head_dim_});
                        auto z_pair = Tensor::view(ctx, buf_.z.data_as<bf16>() + size_t(row) * q_dim, {2, full_q_heads_, full_head_dim_});
                        auto out_pair = Tensor::view(ctx, buf_.attn_out.data_as<bf16>() + size_t(row) * q_dim, {2, q_dim});
                        ops::q4e_attention_decode_pair(ctx, q_pair, cache.k, cache.v, out_pair,
                            buf_.incr_scores, start_pos + row + 1, z_pair,
                            verification_attention_pair_bundle_ ? &*verification_attention_pair_bundle_ : nullptr);
                    }
                    if (row < count) {
                        auto q_row = Tensor::view(ctx, buf_.q.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        auto z_row = Tensor::view(ctx, buf_.z.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        auto out_row = Tensor::view(ctx, buf_.attn_out.data_as<bf16>() + size_t(row) * q_dim, {1, q_dim});
                        ops::attention_decode(ctx, q_row, cache.k, cache.v, out_row, buf_.decode_scores,
                            full_q_heads_, full_kv_heads_, full_head_dim_, start_pos + row + 1,
                            &buf_.decode_attn_partials, &z_row,
                            nullptr, 0, nullptr, 0,
                            cache.k_scale.valid() ? &cache.k_scale : nullptr,
                            cache.v_scale.valid() ? &cache.v_scale : nullptr);
                    }
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
                            &buf_.decode_attn_partials, &z_row,
                            nullptr, 0, nullptr, 0,
                            cache.k_scale.valid() ? &cache.k_scale : nullptr,
                            cache.v_scale.valid() ? &cache.v_scale : nullptr);
                    }
                } else if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores, count, full_q_heads_, full_kv_heads_, full_head_dim_, &buf_.z);
                } else {
                    ops::attention_prefill_cached(ctx, buf_.q, cache.k, cache.v, buf_.attn_out, buf_.incr_scores, count, start_pos, full_q_heads_, full_kv_heads_, full_head_dim_, session.Capacity(), &buf_.z,
                                                  cache.k_scale.valid() ? &cache.k_scale : nullptr,
                                                  cache.v_scale.valid() ? &cache.v_scale : nullptr);
                }
                project_prefill(layer.o_proj, buf_.attn_out, output);
            }
        }
    }
}

bool Qwen4ExpBnb4Backend::QsaLayerBlockStateEligible(
    int current_blocks, int target_blocks) noexcept {
    return current_blocks >= 0 && target_blocks >= 0 &&
           (current_blocks == target_blocks ||
            (current_blocks < target_blocks &&
             target_blocks - current_blocks == 1));
}

bool Qwen4ExpBnb4Backend::QsaLayerGraphEligible(
    Context& ctx, const Session& session, const Tensor& input,
    const Tensor& output, int count, int start_pos, bool allowQ8) const {
    const bool trace = qsa_layer_graph_trace_enabled();
    const auto reject = [&](const char* reason) {
        if (trace) {
            AILA_LOG_INFO(
                "[Q4E QsaGraph] reject=%s start_pos=%d current=%d "
                "count=%d q8=%d",
                reason, start_pos, session.CurrentLength(), count,
                allowQ8 ? 1 : 0);
        }
        return false;
    };
    const auto device = ctx.queue().get_device();
    if (!device.has(sycl::aspect::ext_oneapi_limited_graph) &&
        !device.has(sycl::aspect::ext_oneapi_graph)) {
        return reject("device_graph");
    }
    if (count != 1 || start_pos != session.CurrentLength() ||
        start_pos < 0 || allowQ8 || verify_rows_alignment_ ||
        gpuNf4Q8ReferenceForTest_ || decodeHcReferenceForTest_ ||
        layers_.size() <= kQsaGraphLayer ||
        layers_[kQsaGraphLayer].is_linear) {
        return reject("base");
    }
    if (session.mropePosT_.valid() || session.mropePosH_.valid() ||
        session.mropePosW_.valid() || session.mropePromptLength_ != 0 ||
        session.mropeTextPositionDelta_ != 0 ||
        !session.embeddingOverridePositions_.empty() ||
        !session.embeddingOverrideValues_.empty() ||
        session.embeddingOverrideHiddenSize_ != 0) {
        return reject("mrope");
    }
    if (hidden_size_ != 2560 || full_q_heads_ != 24 ||
        full_kv_heads_ != 2 || full_head_dim_ != kQsaGraphHeadDim ||
        full_q_dim_ != 24 * kQsaGraphHeadDim ||
        full_q_proj_dim_ != 2 * 24 * kQsaGraphHeadDim ||
        full_kv_dim_ != 2 * kQsaGraphHeadDim ||
        full_fused_qkv_dim_ != 13312 ||
        cfg_.indexer_n_heads != kQsaGraphIndexerQHeads ||
        cfg_.indexer_kv_heads != kQsaGraphIndexerKvHeads ||
        cfg_.indexer_head_dim != kQsaGraphIndexerHeadDim ||
        (cfg_.indexer_compress_ratio > 0 ? cfg_.indexer_compress_ratio : 4) !=
            kQsaGraphCompressRatio) {
        return reject("geometry");
    }
    static const bool compressed_k_enabled =
        aila::env::read_flag("AILA_QSA_COMPRESSED_K", true);
    static const bool block_native_enabled =
        aila::env::read_flag("AILA_QSA_BLOCK_NATIVE", true);
    static const bool dense_gemv_enabled =
        aila::env::read_flag("AILA_Q4E_DENSE_GEMV", true);
    if (!compressed_k_enabled || !block_native_enabled ||
        !dense_gemv_enabled || attentionNf4Q8Enabled_) {
        return reject("features");
    }
    const int block_topk =
        (cfg_.indexer_budget > 0 ? cfg_.indexer_budget : 2048) /
        kQsaGraphCompressRatio;
    if (block_topk != kQsaGraphBlockTopK) return reject("block_topk");

    const auto& layer = layers_[kQsaGraphLayer];
    const auto& cache = session.layerCaches_[kQsaGraphLayer];
    if (!layer.qkv_proj.supports_decode_graph() ||
        !layer.o_proj.supports_decode_graph() ||
        !layer.q_norm_weight || !layer.k_norm_weight ||
        !layer.index_qk_weight || !layer.index_q_norm_weight ||
        !layer.index_k_norm_weight ||
        !layer.index_qk_weight->valid() ||
        layer.index_qk_weight->dtype() != dnnl::memory::data_type::bf16 ||
        layer.index_qk_weight->ndim() != 2 ||
        layer.index_qk_weight->shape(0) != 640 ||
        layer.index_qk_weight->shape(1) != hidden_size_ ||
        !layer.q_norm_weight->valid() ||
        layer.q_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
        layer.q_norm_weight->ndim() != 1 ||
        layer.q_norm_weight->shape(0) != full_head_dim_ ||
        !layer.k_norm_weight->valid() ||
        layer.k_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
        layer.k_norm_weight->ndim() != 1 ||
        layer.k_norm_weight->shape(0) != full_head_dim_ ||
        layer.index_q_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
        layer.index_q_norm_weight->ndim() != 1 ||
        layer.index_q_norm_weight->shape(0) != kQsaGraphIndexerHeadDim ||
        layer.index_k_norm_weight->dtype() != dnnl::memory::data_type::bf16 ||
        layer.index_k_norm_weight->ndim() != 1 ||
        layer.index_k_norm_weight->shape(0) != kQsaGraphIndexerHeadDim) {
        return reject("layer_weights");
    }
    if (!cache.k.valid() || !cache.v.valid() ||
        cache.k.dtype() != dnnl::memory::data_type::bf16 ||
        cache.v.dtype() != dnnl::memory::data_type::bf16 ||
        cache.k.ndim() != 3 || cache.v.ndim() != 3 ||
        cache.k.shape(0) != full_kv_heads_ ||
        cache.v.shape(0) != full_kv_heads_ ||
        cache.k.shape(1) != session.Capacity() ||
        cache.v.shape(1) != session.Capacity() ||
        cache.k.shape(2) != full_head_dim_ ||
        cache.v.shape(2) != full_head_dim_ ||
        cache.k_scale.valid() || cache.v_scale.valid() ||
        !cache.indexer_k.valid() ||
        cache.indexer_k.dtype() != dnnl::memory::data_type::bf16 ||
        cache.indexer_k.ndim() != 2 ||
        cache.indexer_k.shape(0) != session.Capacity() ||
        cache.indexer_k.shape(1) != kQsaGraphIndexerHeadDim ||
        !cache.indexer_compressed_k.valid() ||
        cache.indexer_compressed_k.dtype() != dnnl::memory::data_type::bf16 ||
        cache.indexer_compressed_k.ndim() != 2 ||
        cache.indexer_compressed_k.shape(1) != kQsaGraphIndexerHeadDim) {
        return reject("cache_layout");
    }

    const int visible_len = start_pos + 1;
    const int complete_blocks = visible_len / kQsaGraphCompressRatio;
    const int max_blocks =
        static_cast<int>(cache.indexer_compressed_k.shape(0));
    if (complete_blocks <= kQsaGraphBlockTopK ||
        complete_blocks > max_blocks || visible_len > session.Capacity() ||
        !QsaLayerBlockStateEligible(
            cache.num_compressed_blocks, complete_blocks)) {
        return reject("block_state");
    }
    if (input.data() != buf_.x.data() || output.data() != buf_.y.data() ||
        input.dtype() != dnnl::memory::data_type::bf16 ||
        output.dtype() != dnnl::memory::data_type::bf16 ||
        input.numel() < hidden_size_ || output.numel() < hidden_size_ ||
        !buf_.full_qkv.valid() || !buf_.q.valid() || !buf_.z.valid() ||
        !buf_.attn_out.valid() || !buf_.indexer_qk.valid() ||
        !buf_.indexer_q.valid() || !buf_.indexer_block_scores.valid() ||
        !buf_.indexer_active_blocks.valid() ||
        !buf_.decode_attn_partials.valid()) {
        return reject("io_identity");
    }
    if (buf_.full_qkv.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.full_qkv.shape(1) < full_fused_qkv_dim_ ||
        buf_.q.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.q.shape(1) < full_q_dim_ ||
        buf_.z.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.z.shape(1) < full_q_dim_ ||
        buf_.attn_out.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.attn_out.shape(1) < full_q_dim_ ||
        buf_.indexer_qk.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.indexer_qk.shape(1) < 640 ||
        buf_.indexer_q.dtype() != dnnl::memory::data_type::bf16 ||
        buf_.indexer_q.shape(1) < 512 ||
        buf_.indexer_block_scores.dtype() != dnnl::memory::data_type::f32 ||
        buf_.indexer_block_scores.ndim() != 1 ||
        buf_.indexer_block_scores.shape(0) < max_blocks ||
        buf_.indexer_active_blocks.dtype() != dnnl::memory::data_type::s32 ||
        buf_.indexer_active_blocks.ndim() != 1 ||
        buf_.indexer_active_blocks.shape(0) <
            kQsaGraphBlockTopK + kQsaGraphCompressRatio ||
        buf_.decode_attn_partials.dtype() != dnnl::memory::data_type::f32 ||
        buf_.decode_attn_partials.ndim() != 3 ||
        buf_.decode_attn_partials.shape(0) != full_q_heads_ ||
        buf_.decode_attn_partials.shape(1) < 17 ||
        buf_.decode_attn_partials.shape(2) < 272) {
        return reject("buffer_shape");
    }
    return true;
}

aila::TokenGraphKey Qwen4ExpBnb4Backend::MakeQsaLayerGraphKey(
    const Session& session) const {
    auto key = MakeTokenGraphKey(
        session, aila::TokenGraphBody::QsaLayerProof,
        aila::TokenGraphAttention::QsaSparse512BlockNative,
        kQsaGraphSchemaFlags);
    // The L3 body only bakes pointer/layout identity. K/V values, the
    // compressed-block count and PLE/expert state are replay inputs or
    // unrelated session state.
    key.session_state_generation = 0;
    key.residency_generation = 0;
    key.descriptor_generation = 0;
    key.ple_workspace_generation = 0;
    key.ple_graph_view_generation = 0;
    return key;
}

std::size_t Qwen4ExpBnb4Backend::RecordQsaLayerGraphBody(
    Context& ctx, Session& session, int layer_index, int layer_ordinal,
    Tensor& input, Tensor& output, int* control) {
    if (layer_index < 0 ||
        layer_index >= static_cast<int>(layers_.size()) ||
        layers_[layer_index].is_linear ||
        layer_ordinal < 0 ||
        layer_ordinal >= kQsaGraphLayerOrdinal + 12) {
        throw std::invalid_argument(
            "Qwen4Exp full-attention graph body has invalid layer geometry");
    }
    auto& layer = layers_[layer_index];
    auto& cache = session.layerCaches_[layer_index];
    const int indexer_out_dim =
        (cfg_.indexer_n_heads + cfg_.indexer_kv_heads) *
        cfg_.indexer_head_dim;
    const int indexer_q_dim =
        cfg_.indexer_n_heads * cfg_.indexer_head_dim;
    const int max_blocks =
        static_cast<int>(cache.indexer_compressed_k.shape(0));

    std::size_t captured_nodes = 0;
    auto captured_node = [&captured_nodes]() noexcept {
        ++captured_nodes;
    };
    layer.qkv_proj.ForwardDecodeRows(
        ctx, linear_scratch_, input, buf_.full_qkv, 1);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kQkvProj);
    ops::bf16_gemv_bf16(
        ctx, *layer.index_qk_weight, input.data_as<bf16>(),
        buf_.indexer_qk.data_as<bf16>(), indexer_out_dim,
        hidden_size_);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kIndexerQk);

    bf16* fused_ptr = buf_.full_qkv.data_as<bf16>();
    Tensor q_gate_view = Tensor::view(
        ctx, fused_ptr, {1, full_q_proj_dim_});
    Tensor k_view = Tensor::view(
        ctx, fused_ptr + full_q_proj_dim_,
        {1, full_kv_dim_});
    Tensor v_view = Tensor::view(
        ctx, fused_ptr + full_q_proj_dim_ + full_kv_dim_,
        {1, full_kv_dim_});
    Tensor indexer_qk_row = Tensor::view(
        ctx, buf_.indexer_qk.data_as<bf16>(),
        {1, indexer_out_dim});
    Tensor indexer_q_row = Tensor::view(
        ctx, buf_.indexer_q.data_as<bf16>(),
        {1, indexer_q_dim});

    ops::decode_prepare_qgkv_packed_partial_graph(
        ctx, q_gate_view, k_view, v_view, buf_.q, buf_.z,
        *layer.q_norm_weight, *layer.k_norm_weight,
        cache.k, cache.v, control,
        full_q_heads_, full_kv_heads_, full_head_dim_,
        cfg_.rms_norm_eps, 64, cfg_.rope.rope_theta,
        cfg_.rope.mrope_interleaved, nullptr, nullptr, nullptr,
        0, 0, cfg_.rope.mrope_section[0],
        cfg_.rope.mrope_section[1],
        cfg_.rope.mrope_section[2], nullptr, nullptr,
        layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kPrepareQkv);
    ops::qsa_prepare_q_and_cache_k_graph(
        ctx, indexer_qk_row, *layer.index_q_norm_weight,
        cache.indexer_k, indexer_q_row,
        control, cfg_.rope.rope_theta,
        cfg_.rope.mrope_interleaved, nullptr, nullptr, nullptr,
        0, 0, cfg_.rope.mrope_section[0],
        cfg_.rope.mrope_section[1],
        cfg_.rope.mrope_section[2],
        layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kPrepareIndexerQ);
    ops::qsa_compress_one_completed_block_graph(
        ctx, cache.indexer_k, *layer.index_k_norm_weight,
        cache.indexer_compressed_k, control,
        cfg_.rope.rope_theta, cfg_.rope.mrope_interleaved,
        nullptr, nullptr, nullptr, 0, 0,
        cfg_.rope.mrope_section[0],
        cfg_.rope.mrope_section[1],
        cfg_.rope.mrope_section[2],
        layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kCompressBlock);
    ops::qsa_compute_block_scores_compressed_graph(
        ctx, indexer_q_row, cache.indexer_compressed_k,
        buf_.indexer_block_scores, control,
        max_blocks, layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kBlockScores);
    ops::qsa_select_topk_blocks_graph(
        ctx, buf_.indexer_block_scores,
        buf_.indexer_active_blocks, control,
        kQsaGraphBlockTopK, kQsaGraphCompressRatio, max_blocks,
        layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kSelectBlocks);
    ops::attention_validate_qsa_active_blocks_graph(
        ctx, cache.k, cache.v, buf_.indexer_active_blocks,
        control, full_kv_heads_,
        layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kValidateBlocks);
    ops::attention_decode_exact_head256_block_native_graph(
        ctx, buf_.q, cache.k, cache.v, buf_.attn_out,
        buf_.decode_attn_partials, buf_.indexer_active_blocks,
        control, full_q_heads_,
        full_kv_heads_, &buf_.z, layer_ordinal);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kSparseAttend);
    layer.o_proj.ForwardDecodeRows(
        ctx, linear_scratch_, buf_.attn_out, output, 1);
    captured_node();
    aila::token_graph_stage_prof::mark(
        aila::token_graph_stage_prof::kOProj);
    return captured_nodes;
}

std::size_t Qwen4ExpBnb4Backend::RecordSingleFullAttentionLayerGraphBody(
    Context& ctx, Session& session, int layer, int layer_ordinal,
    Tensor& input, Tensor& output, int* control) {
    return RecordQsaLayerGraphBody(
        ctx, session, layer, layer_ordinal, input, output, control);
}

bool Qwen4ExpBnb4Backend::ForwardQsaLayerGraph(
    Context& ctx, Session& session, Tensor& input, Tensor& output,
    int start_pos) {
    const bool trace = qsa_layer_graph_trace_enabled();
    if (session.token_graph_poisoned()) {
        throw std::logic_error(
            "Qwen4Exp L3 graph session is poisoned; recreate the session");
    }
    if (!QsaLayerGraphEligible(
            ctx, session, input, output, 1, start_pos, false)) {
        if (trace) {
            AILA_LOG_INFO(
                "[Q4E QsaGraph] eligible=0 start_pos=%d current=%d "
                "blocks=%d",
                start_pos, session.CurrentLength(),
                session.layerCaches_[kQsaGraphLayer].num_compressed_blocks);
        }
        return false;
    }
    if (trace) {
        AILA_LOG_INFO(
            "[Q4E QsaGraph] eligible=1 start_pos=%d current=%d blocks=%d",
            start_pos, session.CurrentLength(),
            session.layerCaches_[kQsaGraphLayer].num_compressed_blocks);
    }
    ensure_runtime_buffers(ctx, 1);
    const auto runtime_phase = token_graph_runtime_.phase();
    if (runtime_phase == aila::TokenGraphPhase::Capturing ||
        runtime_phase == aila::TokenGraphPhase::InFlight) {
        (void)AbortAndPoisonTokenGraph(session);
        if (!ReleaseQsaLayerGraph(&ctx, false)) {
            throw std::runtime_error(
                "Qwen4Exp L3 graph could not drain its previous executable");
        }
        throw std::runtime_error(
            "Qwen4Exp L3 graph cannot capture while another graph owns the backend");
    }

    const auto key = MakeQsaLayerGraphKey(session);
    const bool reuse =
        runtime_phase == aila::TokenGraphPhase::Ready &&
        same_token_graph_key(token_graph_runtime_.key(), key);
    if (!reuse) {
        ctx.synchronize();
        if (!ReleaseQsaLayerGraph(&ctx, true)) {
            (void)AbortAndPoisonTokenGraph(session);
            throw std::runtime_error(
                "Qwen4Exp L3 graph could not release its previous executable");
        }
        if (!token_graph_runtime_.BeginCapture(key)) {
            (void)AbortAndPoisonTokenGraph(session);
            throw std::runtime_error(
                "Qwen4Exp L3 graph failed to enter capture");
        }
        try {
            if (!qsa_layer_graph_.control.valid()) {
                qsa_layer_graph_.control = Tensor::allocate(
                    ctx, {ops::graph_control::words},
                    dnnl::memory::data_type::s32);
            }
            qsa_layer_graph_.control_host.fill(0);
            qsa_layer_graph_.control_host[ops::graph_control::start_pos] =
                start_pos;
            qsa_layer_graph_.control_host[
                ops::graph_control_layer_blocks_slot(kQsaGraphLayerOrdinal)] =
                session.layerCaches_[kQsaGraphLayer].num_compressed_blocks;
            ctx.memcpy_h2d(
                qsa_layer_graph_.control.data(),
                qsa_layer_graph_.control_host.data(),
                qsa_layer_graph_.control.size_bytes());
            ctx.synchronize();

            int* control = qsa_layer_graph_.control.data_as<int>();

            sycl::ext::oneapi::experimental::command_graph<
                sycl::ext::oneapi::experimental::graph_state::modifiable>
                recording(ctx.queue().get_context(), ctx.queue().get_device());
            recording.begin_recording(ctx.queue());
            std::size_t captured_nodes = 0;
            try {
                captured_nodes = RecordSingleFullAttentionLayerGraphBody(
                    ctx, session, kQsaGraphLayer, kQsaGraphLayerOrdinal,
                    input, output, control);
                if (captured_nodes == 0) {
                    throw std::runtime_error(
                        "Qwen4Exp L3 graph capture recorded no nodes");
                }
                recording.end_recording(ctx.queue());
            } catch (...) {
                try {
                    recording.end_recording(ctx.queue());
                } catch (...) {
                }
                throw;
            }
            qsa_layer_graph_.executable = std::make_unique<
                QsaLayerGraphState::Executable>(recording.finalize());
            qsa_layer_graph_.node_count = captured_nodes;
            if (!MarkReady()) {
                throw std::runtime_error(
                    "Qwen4Exp L3 graph failed to publish its executable");
            }
            if (trace) {
                AILA_LOG_INFO(
                    "[Q4E QsaGraph] capture_ready start_pos=%d nodes=%zu",
                    start_pos, captured_nodes);
            }
        } catch (...) {
            (void)AbortAndPoisonTokenGraph(session);
            (void)ReleaseQsaLayerGraph(&ctx, false);
            throw;
        }
    }

    if (!qsa_layer_graph_.executable || !qsa_layer_graph_.control.valid()) {
        (void)AbortAndPoisonTokenGraph(session);
        throw std::runtime_error(
            "Qwen4Exp L3 graph is missing its executable or control table");
    }

    const int blocks_slot =
        ops::graph_control_layer_blocks_slot(kQsaGraphLayerOrdinal);
    // The graph may advance the selected device counter by one completed block.
    // Validate the post-replay target, not the pre-replay control value.
    const int target_blocks =
        (start_pos + 1) / kQsaGraphCompressRatio;
    qsa_layer_graph_.control_host.fill(0);
    qsa_layer_graph_.control_host[ops::graph_control::start_pos] = start_pos;
    qsa_layer_graph_.control_host[blocks_slot] =
        session.layerCaches_[kQsaGraphLayer].num_compressed_blocks;
    (void)ctx.memcpy_h2d_async(
        qsa_layer_graph_.control.data(),
        qsa_layer_graph_.control_host.data(),
        qsa_layer_graph_.control.size_bytes());

    try {
        auto graph_event = ctx.queue().ext_oneapi_graph(
            *qsa_layer_graph_.executable);
        const uint32_t sequence = token_graph_runtime_.next_sequence();
        if (!Launch(session, sequence, graph_event)) {
            throw std::runtime_error(
                "Qwen4Exp L3 graph launch was rejected");
        }
        graph_event.wait_and_throw();
        ctx.memcpy_d2h(
            qsa_layer_graph_.control_host.data(),
            qsa_layer_graph_.control.data(),
            qsa_layer_graph_.control.size_bytes());
        const bool gate_valid =
            qsa_layer_graph_.control_host[ops::graph_control::abort] == 0 &&
            qsa_layer_graph_.control_host[ops::graph_control::start_pos] ==
                start_pos &&
            qsa_layer_graph_.control_host[blocks_slot] == target_blocks;
        if (trace) {
            AILA_LOG_INFO(
                "[Q4E QsaGraph] replay start_pos=%d blocks=%d target=%d "
                "gate=%d",
                start_pos, qsa_layer_graph_.control_host[blocks_slot],
                target_blocks, gate_valid ? 1 : 0);
        }
        if (!Complete(sequence, gate_valid)) {
            throw std::runtime_error(
                "Qwen4Exp L3 graph completion gate rejected");
        }
        session.layerCaches_[kQsaGraphLayer].num_compressed_blocks =
            qsa_layer_graph_.control_host[blocks_slot];
        return true;
    } catch (...) {
        (void)AbortAndPoisonTokenGraph(session);
        (void)ReleaseQsaLayerGraph(&ctx, false);
        throw;
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
            true,widths[section]==2?n/2:0,widths[section]==4?n/4:0,true);
        last=ops::moe_merge_compact_down(ctx,buf_.moe_down_acc.data_as<float>(),expert_map_token_dev_,
            buf_.moe_inverse_map.data_as<int>(),batch_weights_dev_,buf_.moe_window_sum.data_as<int64_t>(),buf_.moe_compact_invalid.data_as<int>(),
            begin,n,cfg_.num_experts_per_tok,hidden_size_,&last);
        done+=n;
        if(done==lengths[section])offset+=lengths[section];
    }
    return last;
}
