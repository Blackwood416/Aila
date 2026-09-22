#pragma once

#include "HybridLayerExecution.hpp"
#include "core/Tensor.hpp"
#include "ops/Ops.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aila::models {

// Decode-only batch view: each row names an independent session and consumes
// one token at that session's own position. Projection rows are dense, while
// attention and DeltaNet use the original single-sequence kernels and state.
// No padding row can write a KV slot, convolution head or recurrent matrix.
struct HybridDecodeBatchExecutor {
    template <typename Backend>
    static bool Supported(const Backend& model) noexcept {
        // These old diagnostic modes assume a single sequence/row. Preserve
        // their complete original path by making such sessions ineligible.
        static const bool profile = aila::env::read_flag("AILA_PROFILE_Q35_DECODE", false);
        static const bool stats = aila::env::read_flag("AILA_DEBUG_Q35_LAYER_STATS", false);
        static const int detail = aila::env::read_int_raw("AILA_DEBUG_Q35_LAYER_DETAIL", -1);
        static const bool compare = aila::env::read_flag("AILA_DEBUG_Q35_LINEAR_COMPARE", false);
        static const bool fusedResidual = aila::env::read_flag("AILA_FUSE_RESIDUAL_ADD", false);
        return model.decoderPlan_ && !profile && !stats && detail < 0 && !compare && !fusedResidual &&
            std::none_of(model.decoderPlan_->layers.begin(), model.decoderPlan_->layers.end(),
                         [](const auto& plan) { return plan.kind == HybridLayerKind::LegacyLinearAttention; });
    }

    template <typename Backend, typename Projection>
    static Tensor& Execute(Backend& model, Context& ctx,
                           const std::vector<typename Backend::Session*>& sessions,
                           const int* tokenIdsDevice, const Projection& projection,
                           bool useHostGroupedLinear) {
        using bf16 = sycl::ext::oneapi::bfloat16;
        using Slot = HybridProjectionSlot;
        const int count = static_cast<int>(sessions.size());
        if (count < 1 || count > 4 || !tokenIdsDevice || &ctx != model.context_) {
            throw std::invalid_argument("Hybrid decode batch requires the model context, token IDs and one to four rows");
        }
        for (size_t i = 0; i < sessions.size(); ++i) {
            if (!sessions[i] || std::find(sessions.begin(), sessions.begin() + i, sessions[i]) != sessions.begin() + i) {
                throw std::invalid_argument("Hybrid decode batch requires distinct non-null sessions");
            }
            model.ValidateSession(*sessions[i]);
            if (sessions[i]->currentLength_ >= sessions[i]->Capacity()) {
                throw std::invalid_argument("Hybrid decode batch exceeds session capacity");
            }
        }
        if (count == 1) return model.Forward(ctx, *sessions[0], tokenIdsDevice, 1);
        if (!Supported(model)) throw std::invalid_argument("Hybrid model/debug configuration does not support decode batching");

        model.ensure_runtime_buffers(ctx, count);
        auto& buf = model.buf_;
        if (!model.batchLogits_.valid()) {
            model.batchLogits_ = Tensor::allocate(ctx, {4, static_cast<int64_t>(model.cfg_.vocab_size)});
        }
        model.batchLogitsView_ = Tensor::view(ctx, model.batchLogits_.data(),
            {count, static_cast<int64_t>(model.cfg_.vocab_size)}, model.batchLogits_.dtype());
        auto row = [&](Tensor& storage, int index, int width) {
            // Storage capacity may have a larger width. Each producing kernel
            // writes logical-width packed rows; capacity is never a row stride.
            return Tensor::view(ctx, storage.template data_as<bf16>() + static_cast<size_t>(index) * width,
                                {1, static_cast<int64_t>(width)}, storage.dtype());
        };
        auto normalizeRows = [&](Tensor& input, Tensor& weight, Tensor& output) {
            for (int r = 0; r < count; ++r) {
                auto source = row(input, r, model.hidden_size_);
                auto target = row(output, r, model.hidden_size_);
                ops::rms_norm(ctx, source, weight, model.cfg_.rms_norm_eps, target, 1, model.hidden_size_);
            }
        };
        auto addNormalizeRows = [&](Tensor& input, Tensor& residual, Tensor& weight, Tensor& output) {
            for (int r = 0; r < count; ++r) {
                auto source = row(input, r, model.hidden_size_);
                auto added = row(residual, r, model.hidden_size_);
                auto target = row(output, r, model.hidden_size_);
                // Decode and prefill have different reduction/rounding
                // boundaries. Each independent sequence remains decode.
                ops::fused_add_rms_norm(ctx, source, added, weight, model.cfg_.rms_norm_eps,
                                        target, 1, model.hidden_size_);
            }
        };
        ops::embedding_lookup(ctx, *model.embed_weight_, tokenIdsDevice, count, buf.hidden, model.hidden_size_);
        for (int r = 0; r < count; ++r) {
            auto& session = *sessions[static_cast<size_t>(r)];
            if (session.embeddingOverrideHiddenSize_ != model.hidden_size_) continue;
            for (size_t i = 0; i < session.embeddingOverridePositions_.size(); ++i) {
                if (session.embeddingOverridePositions_[i] != session.currentLength_) continue;
                auto* source = session.embeddingOverrideValues_.data() + i * static_cast<size_t>(model.hidden_size_);
                auto* target = buf.hidden.template data_as<bf16>() + static_cast<size_t>(r) * model.hidden_size_;
                ctx.queue().memcpy(target, source, static_cast<size_t>(model.hidden_size_) * sizeof(bf16));
            }
        }
        normalizeRows(buf.hidden, *model.decoderPlan_->initialNormWeight, buf.normed);
        int rotary = std::max(2, static_cast<int>(std::floor(model.full_head_dim_ * model.cfg_.rope.partial_rotary_factor)));
        rotary = std::min(model.full_head_dim_, rotary);
        if (rotary & 1) --rotary;
        if (rotary <= 0) rotary = std::min(2, model.full_head_dim_);

        for (const auto& plan : model.decoderPlan_->layers) {
            auto& layer = *plan.layer;
            if (plan.kind == HybridLayerKind::DeltaNet) {
                projection.template ForwardDecodeRows<Slot::LinearAll>(ctx, layer, buf.normed, buf.linear_all, count);
                for (int r = 0; r < count; ++r) {
                    auto& cache = sessions[static_cast<size_t>(r)]->layerCaches_[static_cast<size_t>(plan.cacheIndex)];
                    auto* fused = buf.linear_all.template data_as<bf16>() + static_cast<size_t>(r) * model.linear_all_dim_;
                    auto qkv = Tensor::view(ctx, fused, {1, static_cast<int64_t>(model.linear_qkv_dim_)});
                    auto z = Tensor::view(ctx, fused + model.linear_qkv_dim_, {1, static_cast<int64_t>(model.linear_z_dim_)});
                    auto a = Tensor::view(ctx, fused + model.linear_qkv_dim_ + model.linear_z_dim_,
                                          {1, static_cast<int64_t>(model.linear_kv_heads_)});
                    auto b = Tensor::view(ctx, fused + model.linear_qkv_dim_ + model.linear_z_dim_ + model.linear_kv_heads_,
                                          {1, static_cast<int64_t>(model.linear_kv_heads_)});
                    auto out = row(buf.attn_out, r, model.linear_kv_dim_);
                    if (useHostGroupedLinear) model.run_linear_delta_host(ctx, layer, cache, qkv, z, a, b, 1, out);
                    else model.run_linear_delta_decode_gpu(ctx, layer, cache, qkv, z, a, b, out);
                }
                projection.template ForwardDecodeRows<Slot::LinearOutput>(ctx, layer, buf.attn_out, buf.gate, count);
            } else {
                projection.template ForwardDecodeRows<Slot::FullQkv>(ctx, layer, buf.normed, buf.full_qkv, count);
                for (int r = 0; r < count; ++r) {
                    auto& session = *sessions[static_cast<size_t>(r)];
                    auto& cache = session.layerCaches_[static_cast<size_t>(plan.cacheIndex)];
                    auto* fused = buf.full_qkv.template data_as<bf16>() + static_cast<size_t>(r) * model.full_fused_qkv_dim_;
                    auto packedQ = Tensor::view(ctx, fused, {1, static_cast<int64_t>(model.full_q_proj_dim_)});
                    auto k = Tensor::view(ctx, fused + model.full_q_proj_dim_, {1, static_cast<int64_t>(model.full_kv_dim_)});
                    auto v = Tensor::view(ctx, fused + model.full_q_proj_dim_ + model.full_kv_dim_,
                                          {1, static_cast<int64_t>(model.full_kv_dim_)});
                    auto q = model.cfg_.attn_output_gate ? row(buf.q, r, model.full_q_dim_) :
                        Tensor::view(ctx, fused, {1, static_cast<int64_t>(model.full_q_dim_)});
                    auto z = row(buf.z, r, model.full_q_dim_);
                    const auto* posT = static_cast<const int*>(session.mropePosT_.data());
                    const auto* posH = static_cast<const int*>(session.mropePosH_.data());
                    const auto* posW = static_cast<const int*>(session.mropePosW_.data());
                    if (model.cfg_.attn_output_gate) {
                        ops::decode_prepare_qgkv_packed_partial(ctx, packedQ, k, v, q, z,
                            *layer.q_norm_weight, *layer.k_norm_weight, cache.k, cache.v, session.currentLength_,
                            model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_, model.cfg_.rms_norm_eps,
                            rotary, model.cfg_.rope.rope_theta, model.cfg_.rope.mrope_interleaved,
                            posT, posH, posW, session.mropePromptLength_, session.mropeTextPositionDelta_,
                            model.cfg_.rope.mrope_section[0], model.cfg_.rope.mrope_section[1], model.cfg_.rope.mrope_section[2]);
                    } else {
                        ops::decode_prepare_qkv_partial(ctx, q, k, v,
                            *layer.q_norm_weight, *layer.k_norm_weight, cache.k, cache.v, session.currentLength_,
                            model.full_q_heads_, model.full_kv_heads_, model.full_head_dim_, model.cfg_.rms_norm_eps,
                            rotary, model.cfg_.rope.rope_theta, model.cfg_.rope.mrope_interleaved,
                            posT, posH, posW, session.mropePromptLength_, session.mropeTextPositionDelta_,
                            model.cfg_.rope.mrope_section[0], model.cfg_.rope.mrope_section[1], model.cfg_.rope.mrope_section[2]);
                    }

                }
                {
                    // One multi-row decode launch for all independent sequences instead
                    // of one per row. Each row keeps the single-row kernel body and
                    // reduction order; attention_decode_rows falls back to the original
                    // per-row dispatch whenever a row would select another kernel.
                    // With the output gate, q lives in buf.q at full_q_dim_ stride;
                    // otherwise it sits inside the fused QKV row.
                    Tensor& attentionQuery =
                        model.cfg_.attn_output_gate ? buf.q : buf.full_qkv;
                    const int attentionStride = model.cfg_.attn_output_gate
                        ? model.full_q_dim_ : model.full_fused_qkv_dim_;
                    ops::DecodeRowGeometry attentionRows[4];
                    int maxAttentionLen = 0;
                    for (int r = 0; r < count; ++r) {
                        auto& cache = sessions[static_cast<size_t>(r)]
                            ->layerCaches_[static_cast<size_t>(plan.cacheIndex)];
                        attentionRows[r].k_cache = &cache.k;
                        attentionRows[r].v_cache = &cache.v;
                        attentionRows[r].cached_len =
                            sessions[static_cast<size_t>(r)]->currentLength_ + 1;
                        maxAttentionLen =
                            std::max(maxAttentionLen, attentionRows[r].cached_len);
                    }
                    ops::attention_decode_rows(ctx, attentionQuery, attentionStride,
                                               buf.attn_out, buf.decode_scores,
                                               attentionRows, count,
                                               model.full_q_heads_, model.full_kv_heads_,
                                               model.full_head_dim_, maxAttentionLen,
                                               &buf.decode_attn_partials);
                    if (model.cfg_.attn_output_gate) {
                        for (int r = 0; r < count; ++r) {
                            auto out = row(buf.attn_out, r, model.full_q_dim_);
                            auto z = row(buf.z, r, model.full_q_dim_);
                            ops::sigmoid_mul(ctx, out, z, out, model.full_q_dim_);
                        }
                    }
                }                projection.template ForwardDecodeRows<Slot::FullOutput>(ctx, layer, buf.attn_out, buf.gate, count);
            }
            addNormalizeRows(buf.hidden, buf.gate, *layer.post_attn_ln_weight, buf.normed);
            // Decode's fused gate/up keeps its original weight and activation
            // rounding boundaries. A prefill-shaped BF16 gate/up projection
            // changes those semantics, even with the same mathematical SiLU.
            // Backends that can fuse several rows in one launch do so here; the
            // per-row path remains the fallback and is numerically equivalent.
            if (!projection.TryFusedGateUpRows(ctx, layer, buf.normed, buf.gate, count)) {
                for (int r = 0; r < count; ++r) {
                    auto input = row(buf.normed, r, model.hidden_size_);
                    auto gate = row(buf.gate, r, model.ff_dim_);
                    if (!projection.TryFusedGateUp(ctx, layer, input, gate, model.ff_dim_)) {
                        auto gateUp = row(buf.gate_up, r, 2 * model.ff_dim_);
                        projection.template Forward<Slot::FfnGateUp>(ctx, layer, input, gateUp, 1);
                        ops::fused_gate_up_swiglu(ctx, gateUp, gate, model.ff_dim_);
                    }
                }
            }
            projection.template ForwardDecodeRows<Slot::FfnDown>(ctx, layer, buf.gate, buf.up, count);
            addNormalizeRows(buf.hidden, buf.up, *plan.postMlpNormWeight, buf.normed);
        }
        // One LM-head row and one sampling state per sequence. B is never
        // interpreted as a prompt suffix with only the final row retained.
        for (int r = 0; r < count; ++r) {
            auto hidden = row(buf.normed, r, model.hidden_size_);
            auto logits = row(model.batchLogits_, r, model.cfg_.vocab_size);
            model.lm_head_.forward(ctx, hidden, logits, 1);
        }
        for (auto* session : sessions) model.CaptureSnapshot(ctx, *session, session->currentLength_ + 1, 1);
        for (auto* session : sessions) ++session->currentLength_;
        return model.batchLogitsView_;
    }
};

} // namespace aila::models
