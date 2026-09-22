#pragma once

#include "HybridLayerExecution.hpp"
#include "ops/Ops.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace aila::models {

// Both Hybrid storage backends instantiate this layer schedule. Projection
// policy selection is static; the policy retains NF4, BF16/JM and FFN fast
// paths without type erasure or model-name dispatch in the layer loop.
struct HybridLayerExecutor {
    template <typename Backend, typename ProjectionPolicy>
    static void Execute(Backend& model, const typename Backend::LayerPlan& plan,
                        const typename Backend::LayerExecution& execution,
                        const ProjectionPolicy& projection) {
        using bf16 = sycl::ext::oneapi::bfloat16;
        auto& ctx = execution.context;
        auto& buf = execution.buffers;
        auto& layer = *plan.layer;
        auto& cache = execution.cache;
        const auto& time_stage = execution.profiler;
        const int i = plan.cacheIndex;
        const int seq_len = execution.tokenCount;
        const int start_pos = execution.startPosition;
        const int cached_len = execution.cachedLength;
        const int full_rotary_dim = execution.fullRotaryDimension;
        const bool decode = execution.phase == HybridExecutionPhase::Decode;
        bf16* hidden_ptr_for_residual = projection.ResidualHidden();
        const bool debug_layer_stats = execution.debugLayerStats;
        const int debug_layer_detail_idx = execution.debugLayerDetail;
        const bool debug_linear_compare = execution.debugLinearCompare;
        const int debug_linear_compare_layer = execution.debugLinearCompareLayer;
        const bool use_host_grouped_linear = execution.useHostGroupedLinear;
        auto log_row_stats = [&](const char* tag, Tensor& t, int row, int width) {
            if (row < 0) return;
            bf16* ptr = static_cast<bf16*>(t.data()) + (size_t)row * width;
            std::vector<bf16> h_row((size_t)width);
            ctx.memcpy_d2h(h_row.data(), ptr, h_row.size() * sizeof(bf16));
            double mean = 0.0;
            double abs_mean = 0.0;
            float max_abs = 0.0f;
            for (int d = 0; d < width; ++d) {
                float v = static_cast<float>(h_row[(size_t)d]);
                mean += v;
                abs_mean += std::abs(v);
                max_abs = std::max(max_abs, std::abs(v));
            }
            mean /= (double)width;
            abs_mean /= (double)width;
            float v0 = static_cast<float>(h_row[0]);
            float v1 = static_cast<float>(h_row[1]);
            float v2 = static_cast<float>(h_row[2]);
            float v3 = static_cast<float>(h_row[3]);
            AILA_LOG_INFO("[Q35L0] %s mean=%.6f abs_mean=%.6f max_abs=%.6f v0=%.6f v1=%.6f v2=%.6f v3=%.6f",
                          tag, mean, abs_mean, max_abs, v0, v1, v2, v3);
        };

        // Entry and exit retain the normalized input / residual stream contract.
        int dbg_row = seq_len - 1;
        bool debug_this_layer = (debug_layer_detail_idx >= 0 && debug_layer_detail_idx == i);
        if (debug_this_layer && seq_len > 0) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "layer%d_input_ln_out", i);
            log_row_stats(tag, buf.normed, dbg_row, model.hidden_size_);
        }

        if (plan.kind != HybridLayerKind::FullAttention) {
            if (model.use_delta_linear_) {
                if (decode) {
                    time_stage(ProfileStage::LinearProj, [&] {
                        projection.template Forward<HybridProjectionSlot::LinearAll>(ctx, layer, buf.normed, buf.linear_all, seq_len);
                    });
                    time_stage(ProfileStage::LinearDelta, [&] {
                        bf16* fused_ptr = static_cast<bf16*>(buf.linear_all.data());
                        Tensor qkv_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)model.linear_qkv_dim_});
                        Tensor z_view = Tensor::view(ctx, fused_ptr + model.linear_qkv_dim_, {1, (int64_t)model.linear_z_dim_});
                        Tensor a_view = Tensor::view(ctx, fused_ptr + model.linear_qkv_dim_ + model.linear_z_dim_,
                                                     {1, (int64_t)model.linear_kv_heads_});
                        Tensor b_view = Tensor::view(ctx, fused_ptr + model.linear_qkv_dim_ + model.linear_z_dim_ + model.linear_kv_heads_,
                                                     {1, (int64_t)model.linear_kv_heads_});
                        if (use_host_grouped_linear) {
                            model.run_linear_delta_host(ctx, layer, cache, qkv_view, z_view, a_view, b_view, 1);
                            return;
                        }
                        bool compare_this_layer = debug_linear_compare &&
                                                  (debug_linear_compare_layer < 0 || debug_linear_compare_layer == i);
                        if (compare_this_layer) {
                            model.debug_compare_linear_delta_decode(ctx, i, layer, cache, qkv_view, z_view, a_view, b_view);
                        } else {
                            model.run_linear_delta_decode_gpu(ctx, layer, cache, qkv_view, z_view, a_view, b_view);
                        }
                    });
                } else {
                    time_stage(ProfileStage::LinearProj, [&] {
                        projection.template Forward<HybridProjectionSlot::LinearAll>(ctx, layer, buf.normed, buf.linear_all, seq_len);
                    });
                    time_stage(ProfileStage::LinearDelta, [&] {
                        if (use_host_grouped_linear) {
                            bf16* fused_ptr = static_cast<bf16*>(buf.linear_all.data());
                            bf16* qkv_ptr = static_cast<bf16*>(buf.qkv.data());
                            bf16* z_ptr = static_cast<bf16*>(buf.z.data());
                            bf16* a_ptr = static_cast<bf16*>(buf.a.data());
                            bf16* b_ptr = static_cast<bf16*>(buf.b.data());
                            const int row_width = model.linear_all_dim_;
                            const int qkv_width = model.linear_qkv_dim_;
                            const int z_width = model.linear_z_dim_;
                            const int ab_width = model.linear_kv_heads_;
                            ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(seq_len) * static_cast<size_t>(row_width)),
                                [=](sycl::id<1> idx) {
                                    size_t linear_idx = idx[0];
                                    int row = static_cast<int>(linear_idx / static_cast<size_t>(row_width));
                                    int col = static_cast<int>(linear_idx % static_cast<size_t>(row_width));
                                    bf16 v = fused_ptr[linear_idx];
                                    if (col < qkv_width) {
                                        qkv_ptr[static_cast<size_t>(row) * qkv_width + col] = v;
                                    } else if (col < qkv_width + z_width) {
                                        z_ptr[static_cast<size_t>(row) * z_width + (col - qkv_width)] = v;
                                    } else if (col < qkv_width + z_width + ab_width) {
                                        a_ptr[static_cast<size_t>(row) * ab_width + (col - qkv_width - z_width)] = v;
                                    } else {
                                        b_ptr[static_cast<size_t>(row) * ab_width + (col - qkv_width - z_width - ab_width)] = v;
                                    }
                                });
                            model.run_linear_delta_host(ctx, layer, cache,
                                                  buf.qkv, buf.z, buf.a, buf.b,
                                                  seq_len);
                            return;
                        }
                        model.run_linear_delta_prefill_gpu_batched(ctx, layer, cache,
                                                             buf.linear_all, buf.attn_out, seq_len);
                    });
                }
            } else {
                time_stage(ProfileStage::LinearProj, [&] {
                    projection.template Forward<HybridProjectionSlot::LinearQkv>(ctx, layer, buf.normed, buf.qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    ops::split_qkv(ctx, buf.qkv, buf.q, buf.k, buf.v,
                                   seq_len, model.linear_q_dim_, model.linear_q_dim_);
                });

                time_stage(ProfileStage::QkNormRope, [&] {
                    std::vector<bf16> hq((size_t)seq_len * model.linear_q_dim_);
                    std::vector<bf16> hk((size_t)seq_len * model.linear_q_dim_);
                    std::vector<float> fq((size_t)seq_len * model.linear_q_dim_);
                    std::vector<float> fk((size_t)seq_len * model.linear_q_dim_);
                    ctx.memcpy_d2h(hq.data(), buf.q.data(), hq.size() * sizeof(bf16));
                    ctx.memcpy_d2h(hk.data(), buf.k.data(), hk.size() * sizeof(bf16));
                    for (size_t z = 0; z < fq.size(); ++z) {
                        fq[z] = static_cast<float>(hq[z]);
                        fk[z] = static_cast<float>(hk[z]);
                    }
                    model.head_l2_norm_inplace(fq, seq_len, model.linear_q_heads_, model.linear_head_dim_, model.cfg_.rms_norm_eps);
                    model.head_l2_norm_inplace(fk, seq_len, model.linear_q_heads_, model.linear_head_dim_, model.cfg_.rms_norm_eps);
                    for (size_t z = 0; z < fq.size(); ++z) {
                        hq[z] = bf16(fq[z]);
                        hk[z] = bf16(fk[z]);
                    }
                    ctx.memcpy_h2d(buf.q.data(), hq.data(), hq.size() * sizeof(bf16));
                    ctx.memcpy_h2d(buf.k.data(), hk.data(), hk.size() * sizeof(bf16));
                });

                time_stage(ProfileStage::KvCache, [&] {
                    ops::copy_to_cache(ctx, buf.k, cache.k, seq_len, start_pos,
                                       model.linear_q_heads_, model.linear_head_dim_, execution.cacheCapacity);
                    ops::copy_to_cache(ctx, buf.v, cache.v, seq_len, start_pos,
                                       model.linear_kv_heads_, model.cfg_.linear_value_head_dim, execution.cacheCapacity);
                });

                time_stage(ProfileStage::Attention, [&] {
                    if (decode) {
                        ops::attention_decode(ctx, buf.q, cache.k, cache.v, buf.attn_out, buf.decode_scores,
                                              model.linear_q_heads_, model.linear_kv_heads_, model.linear_head_dim_, cached_len);
                    } else if (start_pos == 0) {
                        ops::attention_prefill(ctx, buf.q, buf.k, buf.v, buf.attn_out, buf.scores,
                                               seq_len, model.linear_q_heads_, model.linear_kv_heads_, model.linear_head_dim_);
                    } else {
                        ops::attention_prefill_cached(ctx, buf.q, cache.k, cache.v, buf.attn_out, buf.incr_scores,
                                                      seq_len, start_pos, model.linear_q_heads_, model.linear_kv_heads_,
                                                      model.linear_head_dim_, execution.cacheCapacity);
                    }
                });

                time_stage(ProfileStage::AttnGate, [&] {
                    projection.template Forward<HybridProjectionSlot::LinearGate>(ctx, layer, buf.normed, buf.z, seq_len);
                    ops::sigmoid_mul(ctx, buf.attn_out, buf.z, buf.attn_out, seq_len * model.linear_kv_dim_);
                });
            }
            time_stage(ProfileStage::LinearOProj, [&] {
                projection.template Forward<HybridProjectionSlot::LinearOutput>(ctx, layer, buf.attn_out, buf.gate, seq_len);
            });
        } else {
            Tensor q_decode_view;
            Tensor q_gate_decode_view;
            Tensor k_decode_view;
            Tensor v_decode_view;
            Tensor* q_for_attn = &buf.q;
            Tensor* k_for_attn = &buf.k;
            Tensor* v_for_attn = &buf.v;

            if (decode) {
                time_stage(ProfileStage::FullQkvProj, [&] {
                    projection.template Forward<HybridProjectionSlot::FullQkv>(ctx, layer, buf.normed, buf.full_qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    bf16* fused_ptr = static_cast<bf16*>(buf.full_qkv.data());
                    if (model.cfg_.attn_output_gate) {
                        q_gate_decode_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)model.full_q_proj_dim_});
                        q_for_attn = &buf.q;
                    } else {
                        q_decode_view = Tensor::view(ctx, fused_ptr, {1, (int64_t)model.full_q_dim_});
                        q_for_attn = &q_decode_view;
                    }
                    k_decode_view = Tensor::view(ctx, fused_ptr + model.full_q_proj_dim_, {1, (int64_t)model.full_kv_dim_});
                    v_decode_view = Tensor::view(ctx, fused_ptr + model.full_q_proj_dim_ + model.full_kv_dim_, {1, (int64_t)model.full_kv_dim_});
                    k_for_attn = &k_decode_view;
                    v_for_attn = &v_decode_view;
                });
            } else {
                time_stage(ProfileStage::FullQkvProj, [&] {
                    projection.template Forward<HybridProjectionSlot::FullQkv>(ctx, layer, buf.normed, buf.full_qkv, seq_len);
                });
                time_stage(ProfileStage::FullSplit, [&] {
                    ops::split_qkv(ctx, buf.full_qkv, buf.qkv, buf.k, buf.v,
                                   seq_len, model.full_q_proj_dim_, model.full_kv_dim_);
                    if (model.cfg_.attn_output_gate) {
                        ops::split_q_gate(ctx, buf.qkv, buf.q, buf.z,
                                          seq_len, model.full_q_heads_, model.full_head_dim_);
                    } else {
                        ops::copy_tensor(ctx, buf.qkv, buf.q, seq_len * model.full_q_dim_);
                    }
                });
            }
            if (debug_this_layer && seq_len > 0 && model.cfg_.attn_output_gate) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_gate_z", i);
                log_row_stats(tag, buf.z, dbg_row, model.full_q_dim_);
            }

            time_stage(ProfileStage::QkNormRope, [&] {
                if (decode) {
                    if (model.cfg_.attn_output_gate) {
                        ops::decode_prepare_qgkv_packed_partial(ctx, q_gate_decode_view, *k_for_attn, *v_for_attn,
                                                                buf.q, buf.z,
                                                                *layer.q_norm_weight, *layer.k_norm_weight,
                                                                cache.k, cache.v, start_pos,
                                                                model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_,
                                                                model.cfg_.rms_norm_eps, full_rotary_dim,
                                                                model.cfg_.rope.rope_theta,
                                                                model.cfg_.rope.mrope_interleaved,
                                                                execution.mropePositionT, execution.mropePositionH, execution.mropePositionW,
                                                                execution.mropePromptLength, execution.mropeTextPositionDelta,
                                                                model.cfg_.rope.mrope_section[0],
                                                                model.cfg_.rope.mrope_section[1],
                                                                model.cfg_.rope.mrope_section[2]);
                    } else {
                        ops::decode_prepare_qkv_partial(ctx, *q_for_attn, *k_for_attn, *v_for_attn,
                                                        *layer.q_norm_weight, *layer.k_norm_weight,
                                                        cache.k, cache.v, start_pos,
                                                        model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_,
                                                        model.cfg_.rms_norm_eps, full_rotary_dim,
                                                        model.cfg_.rope.rope_theta,
                                                        model.cfg_.rope.mrope_interleaved,
                                                        execution.mropePositionT, execution.mropePositionH, execution.mropePositionW,
                                                        execution.mropePromptLength, execution.mropeTextPositionDelta,
                                                        model.cfg_.rope.mrope_section[0],
                                                        model.cfg_.rope.mrope_section[1],
                                                        model.cfg_.rope.mrope_section[2]);
                    }
                } else {
                    ops::head_rms_norm(ctx, buf.q, *layer.q_norm_weight,
                                       model.cfg_.rms_norm_eps, seq_len, model.full_q_heads_, model.full_head_dim_);
                    ops::head_rms_norm(ctx, *k_for_attn, *layer.k_norm_weight,
                                       model.cfg_.rms_norm_eps, seq_len, model.full_kv_heads_, model.full_head_dim_);

                    ops::apply_rope_partial(ctx, buf.q, *k_for_attn, seq_len, start_pos,
                                            model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_,
                                            full_rotary_dim, model.cfg_.rope.rope_theta,
                                            model.cfg_.rope.mrope_interleaved,
                                            execution.mropePositionT, execution.mropePositionH, execution.mropePositionW,
                                            execution.mropePromptLength, execution.mropeTextPositionDelta,
                                            model.cfg_.rope.mrope_section[0],
                                            model.cfg_.rope.mrope_section[1],
                                            model.cfg_.rope.mrope_section[2]);
                }
            });
            if (debug_this_layer && seq_len > 0) {
                char tag_q[64];
                char tag_k[64];
                char tag_v[64];
                std::snprintf(tag_q, sizeof(tag_q), "layer%d_full_q_after_rope", i);
                std::snprintf(tag_k, sizeof(tag_k), "layer%d_full_k_after_rope", i);
                std::snprintf(tag_v, sizeof(tag_v), "layer%d_full_v", i);
                log_row_stats(tag_q, *q_for_attn, dbg_row, model.full_q_dim_);
                log_row_stats(tag_k, *k_for_attn, dbg_row, model.full_kv_dim_);
                log_row_stats(tag_v, *v_for_attn, dbg_row, model.full_kv_dim_);
            }

            time_stage(ProfileStage::KvCache, [&] {
                if (seq_len > 1) {
                    ops::copy_to_cache(ctx, *k_for_attn, cache.k, seq_len, start_pos,
                                       model.full_kv_heads_, model.full_head_dim_, execution.cacheCapacity);
                    ops::copy_to_cache(ctx, *v_for_attn, cache.v, seq_len, start_pos,
                                       model.full_kv_heads_, model.full_head_dim_, execution.cacheCapacity);
                }
            });

            time_stage(ProfileStage::Attention, [&] {
                if (decode) {
                    ops::attention_decode(ctx, *q_for_attn, cache.k, cache.v, buf.attn_out, buf.decode_scores,
                                          model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_, cached_len,
                                          &buf.decode_attn_partials);
                } else if (start_pos == 0) {
                    ops::attention_prefill(ctx, buf.q, buf.k, buf.v, buf.attn_out, buf.scores,
                                           seq_len, model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_);
                } else {
                    ops::attention_prefill_cached(ctx, buf.q, cache.k, cache.v, buf.attn_out, buf.incr_scores,
                                                  seq_len, start_pos, model.full_q_heads_, model.full_kv_heads_,
                                                  model.full_head_dim_, execution.cacheCapacity);
                }
            });
            if (debug_this_layer && seq_len > 0) {
                char tag[64];
                std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_pre_gate", i);
                log_row_stats(tag, buf.attn_out, dbg_row, model.full_q_dim_);
            }

            if (model.cfg_.attn_output_gate) {
                time_stage(ProfileStage::AttnGate, [&] {
                    ops::sigmoid_mul(ctx, buf.attn_out, buf.z, buf.attn_out, seq_len * model.full_q_dim_);
                });
                if (debug_this_layer && seq_len > 0) {
                    char tag[64];
                    std::snprintf(tag, sizeof(tag), "layer%d_full_attn_out_post_gate", i);
                    log_row_stats(tag, buf.attn_out, dbg_row, model.full_q_dim_);
                }
            }
            time_stage(ProfileStage::FullOProj, [&] {
                projection.template Forward<HybridProjectionSlot::FullOutput>(ctx, layer, buf.attn_out, buf.gate, seq_len);
            });
        }
        if (debug_this_layer && seq_len > 0) {
            char tag[64];
            std::snprintf(tag, sizeof(tag), "layer%d_token_mixer_out", i);
            log_row_stats(tag, buf.gate, dbg_row, model.hidden_size_);
        }

        time_stage(ProfileStage::PostAttnNorm, [&] {
            if (hidden_ptr_for_residual) {
                ops::rms_norm(ctx, buf.gate, *layer.post_attn_ln_weight,
                              model.cfg_.rms_norm_eps, buf.normed, seq_len, model.hidden_size_);
            } else {
                ops::fused_add_rms_norm(ctx, buf.hidden, buf.gate, *layer.post_attn_ln_weight,
                                        model.cfg_.rms_norm_eps, buf.normed, seq_len, model.hidden_size_);
            }
        });
        if (debug_this_layer && seq_len > 0) {
            char tag1[64];
            char tag2[64];
            std::snprintf(tag1, sizeof(tag1), "layer%d_after_attn_residual", i);
            std::snprintf(tag2, sizeof(tag2), "layer%d_post_attn_ln_out", i);
            log_row_stats(tag1, buf.hidden, dbg_row, model.hidden_size_);
            log_row_stats(tag2, buf.normed, dbg_row, model.hidden_size_);
        }

        if (decode) {
            // Each storage policy retains its existing fused decode path and
            // fallback. Activation and projection profiling boundaries are unchanged.
            bool fused_ok = false;
            time_stage(ProfileStage::FfnProj, [&] {
                fused_ok = projection.TryFusedGateUp(ctx, layer, buf.normed, buf.gate, model.ff_dim_);
                if (!fused_ok) {
                    projection.template Forward<HybridProjectionSlot::FfnGateUp>(ctx, layer, buf.normed, buf.gate_up, seq_len);
                }
            });
            if (fused_ok) {
                time_stage(ProfileStage::FfnAct, [&] { });
            } else {
                time_stage(ProfileStage::FfnAct, [&] {
                    ops::fused_gate_up_swiglu(ctx, buf.gate_up, buf.gate, model.ff_dim_);
                });
            }
            time_stage(ProfileStage::DownProj, [&] {
                projection.template Forward<HybridProjectionSlot::FfnDown>(ctx, layer, buf.gate, buf.up, seq_len);
            });
        } else {
            time_stage(ProfileStage::FfnProj, [&] {
                projection.template Forward<HybridProjectionSlot::FfnGateUp>(ctx, layer, buf.normed, buf.gate_up, seq_len);
            });
            time_stage(ProfileStage::FfnAct, [&] {
                ops::split_gate_up(ctx, buf.gate_up, buf.gate, buf.up, seq_len, model.ff_dim_);
                ops::swiglu(ctx, buf.gate, buf.up, buf.gate, seq_len * model.ff_dim_);
            });
            time_stage(ProfileStage::DownProj, [&] {
                projection.template Forward<HybridProjectionSlot::FfnDown>(ctx, layer, buf.gate, buf.up, seq_len);
            });
        }

        time_stage(ProfileStage::PostMlpNorm, [&] {
            if (hidden_ptr_for_residual) {
                ops::rms_norm(ctx, buf.up, *plan.postMlpNormWeight,
                              model.cfg_.rms_norm_eps, buf.normed, seq_len, model.hidden_size_);
            } else {
                ops::fused_add_rms_norm(ctx, buf.hidden, buf.up, *plan.postMlpNormWeight,
                                        model.cfg_.rms_norm_eps, buf.normed, seq_len, model.hidden_size_);
            }
        });
        if (debug_this_layer && seq_len > 0) {
            char tag1[64];
            char tag2[64];
            std::snprintf(tag1, sizeof(tag1), "layer%d_after_mlp_residual", i);
            std::snprintf(tag2, sizeof(tag2), "layer%d_next_input_ln_out", i);
            log_row_stats(tag1, buf.hidden, dbg_row, model.hidden_size_);
            log_row_stats(tag2, buf.normed, dbg_row, model.hidden_size_);
        }

        if (debug_layer_stats && seq_len > 0) {
            const int row = seq_len - 1;
            bf16* hidden_ptr = static_cast<bf16*>(buf.hidden.data()) + (size_t)row * model.hidden_size_;
            std::vector<bf16> h_row((size_t)model.hidden_size_);
            ctx.memcpy_d2h(h_row.data(), hidden_ptr, h_row.size() * sizeof(bf16));

            double mean = 0.0;
            double abs_mean = 0.0;
            float max_abs = 0.0f;
            for (int d = 0; d < model.hidden_size_; ++d) {
                float v = static_cast<float>(h_row[(size_t)d]);
                mean += v;
                abs_mean += std::abs(v);
                max_abs = std::max(max_abs, std::abs(v));
            }
            mean /= (double)model.hidden_size_;
            abs_mean /= (double)model.hidden_size_;
            float v0 = static_cast<float>(h_row[0]);
            float v1 = static_cast<float>(h_row[1]);
            float v2 = static_cast<float>(h_row[2]);
            float v3 = static_cast<float>(h_row[3]);
            AILA_LOG_INFO("[Q35LayerStat] layer=%d mean=%.6f abs_mean=%.6f max_abs=%.6f v0=%.6f v1=%.6f v2=%.6f v3=%.6f",
                          i, mean, abs_mean, max_abs, v0, v1, v2, v3);
        }

    }
};

} // namespace aila::models
