#include "Qwen4ExpMtpDraft.hpp"

#include "../ops/QsaIndexerOps.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
// Fixed MTP expert-cache geometry: 64 slots of 2534400 bytes. The draft never
// scans all 512 source experts and never sizes the cache from the environment.
constexpr size_t kMtpExpertSlotBytes = 2534400;
constexpr int kMtpExpertSlots = 64;
} // namespace

Qwen4ExpMtpDraft::Qwen4ExpMtpDraft(Context& ctx, Qwen4ExpBnb4Backend& target, ModelWeights& weights)
    : ctx_(ctx), target_(target), weights_(weights) {
    try {
        normEmbedding_ = Norm("mtp.pre_fc_norm_embedding.weight");
        normHidden_ = Norm("mtp.pre_fc_norm_hidden.weight");
        Quant("mtp.fc_embedding.weight", fcEmbedding_);
        Quant("mtp.fc_hidden.weight", fcHidden_);
        Require(fcEmbedding_.in_features() == 2560 && fcEmbedding_.out_features() == 2560 &&
                fcHidden_.in_features() == 2560 && fcHidden_.out_features() == 2560,
                "Unexpected MTP FC geometry");
        Hc("mtp.layers.0.attn_hyper_connection.", layer_.attn_hc, true);
        Hc("mtp.layers.0.mlp_hyper_connection.", layer_.mlp_hc, true);
        Hc("mtp.hyper_connection_mixer.", mixer_, false);
        Bnb4BitLinear q, k, v, g, u;
        Quant("mtp.layers.0.self_attn.q_proj.weight", q);
        Quant("mtp.layers.0.self_attn.k_proj.weight", k);
        Quant("mtp.layers.0.self_attn.v_proj.weight", v);
        std::string error;
        Require(layer_.qkv_proj.init_fused_rows(ctx_, q, k, v, &error), error);
        for (const auto* name : {"q_proj", "k_proj", "v_proj"})
            EraseFusedSource(std::string("mtp.layers.0.self_attn.") + name + ".weight");
        Quant("mtp.layers.0.self_attn.o_proj.weight", layer_.o_proj);
        layer_.q_norm_weight = Norm("mtp.layers.0.self_attn.q_norm.weight", true);
        layer_.k_norm_weight = Norm("mtp.layers.0.self_attn.k_norm.weight", true);
        layer_.index_qk_weight = &weights_.upload("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", ctx_);
        layer_.index_k_weight = Tensor::view(ctx_, layer_.index_qk_weight->data_as<bf16>() + 512 * 2560,
                                             {128, 2560});
        layer_.index_k_norm_weight = Norm("mtp.layers.0.self_attn.indexer.k_layernorm.weight", true);
        layer_.router_w = &weights_.upload("mtp.layers.0.mlp.gate.weight", ctx_);
        layer_.shared_expert_gate_w = &weights_.upload("mtp.layers.0.mlp.shared_expert_gate.weight", ctx_);
        Quant("mtp.layers.0.mlp.shared_expert.gate_proj.weight", g);
        Quant("mtp.layers.0.mlp.shared_expert.up_proj.weight", u);
        Require(layer_.shared_gate_up_proj.init_fused_rows(ctx_, g, u, &error), error);
        EraseFusedSource("mtp.layers.0.mlp.shared_expert.gate_proj.weight");
        EraseFusedSource("mtp.layers.0.mlp.shared_expert.up_proj.weight");
        Quant("mtp.layers.0.mlp.shared_expert.down_proj.weight", layer_.shared_down_proj);

        embed_ = Allocate(1, 2560);
        embedNorm_ = Allocate(1, 2560);
        embedFc_ = Allocate(1, 2560);
        hiddenNorm_ = Allocate(4, 2560);
        hostEmbed_.resize(2560);
        embeddingTable_ = weights_.host_view("model.language_model.embed_tokens.weight");
        Require(embeddingTable_.valid() && embeddingTable_.dtype == DT_BF16, "MTP shared embedding missing");

        // One MoE layer, a separate trace suffix, and a fixed 64-slot GPU
        // budget. The cache's own bounded staging and fences handle misses; no
        // CPU expert math and no all-expert warmup happen here.
        Require(experts_.init(ctx_, weights_, 1, kMtpExpertSlots * kMtpExpertSlotBytes, 512, 2560, 640, 64,
                              &error, true, ".mtp"),
                error);
        Bnb4BitHostWeightRef gu, dn;
        Require(LoadBnb4BitHostRef(weights_, "mtp.layers.0.mlp.experts.gate_up_proj", gu, &error, 2560), error);
        Require(LoadBnb4BitHostRef(weights_, "mtp.layers.0.mlp.experts.down_proj", dn, &error, 640), error);
        experts_.set_layer_expert_ref(0, gu, dn);
        Require(experts_.num_slots() == kMtpExpertSlots, "MTP expert slot budget mismatch");
        ctx_.synchronize();
    } catch (...) {
        // Drain any submitted work before members (and their USM) unwind.
        try {
            experts_.wait_prefetch();
            experts_.wait_copy();
            ctx_.synchronize();
        } catch (...) {}
        throw;
    }
}

Qwen4ExpMtpDraft::~Qwen4ExpMtpDraft() {
    try {
        experts_.wait_prefetch();
        experts_.wait_copy();
        ctx_.synchronize();
    } catch (...) {}
}

void Qwen4ExpMtpDraft::Require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

Tensor Qwen4ExpMtpDraft::Allocate(int rows, int cols, dnnl::memory::data_type type) {
    return Tensor::allocate(ctx_, {rows, cols}, type);
}

Tensor* Qwen4ExpMtpDraft::Norm(const std::string& name, bool addOneBf16) {
    auto v = weights_.host_view(name);
    Require(v.valid() && v.dtype == DT_BF16, "MTP norm missing: " + name);
    std::vector<bf16> src(v.bytes / 2);
    std::memcpy(src.data(), v.ptr, v.bytes);
    if (addOneBf16) {
        for (auto& x : src) x = bf16(float(x) + 1.0f);
        auto t = Allocate(1, int(src.size()));
        ctx_.memcpy_h2d(t.data(), src.data(), v.bytes);
        weights_.put(name, std::move(t));
    } else {
        std::vector<float> f(src.size());
        std::transform(src.begin(), src.end(), f.begin(), [](bf16 x) { return float(x); });
        auto t = Allocate(1, int(f.size()), dnnl::memory::data_type::f32);
        ctx_.memcpy_h2d(t.data(), f.data(), f.size() * sizeof(float));
        weights_.put(name, std::move(t));
    }
    return &weights_.get(name);
}

void Qwen4ExpMtpDraft::Quant(const std::string& name, Bnb4BitLinear& projection) {
    for (const auto* suffix :
         {"", ".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4"})
        if (weights_.has_host(name + suffix)) weights_.upload(name + suffix, ctx_);
    Bnb4BitWeightRef ref;
    std::string error;
    Require(LoadBnb4BitWeightRef(ctx_, weights_, name, ref, &error), error);
    Require(projection.init(ctx_, ref, &error), error);
}

void Qwen4ExpMtpDraft::EraseFusedSource(const std::string& name) {
    // Fused projections own their packed rows/scales. Their quant_map can still
    // refer to the first source, matching the main backend loader.
    for (const auto* suffix :
         {"", ".absmax", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4"})
        if (weights_.has(name + suffix)) weights_.erase(name + suffix);
}

void Qwen4ExpMtpDraft::Hc(const std::string& prefix, Qwen4ExpHyperConnectionLayer& hc, bool inject) {
    hc.hc_norm_weight = Norm(prefix + "hc_norm.weight");
    hc.mix_down_w = &weights_.upload(prefix + "input_mix_weight_down.weight", ctx_);
    hc.mix_up_w = &weights_.upload(prefix + "input_mix_weight_up.weight", ctx_);
    if (inject) hc.inject_w = &weights_.upload(prefix + "block_inject_weight.weight", ctx_);
    // Decode uses raw BF16 GEMV; no unused oneDNN primitives for these HCs.
}

std::unique_ptr<Qwen4ExpMtpDraft::State> Qwen4ExpMtpDraft::CreateState(int capacity) {
    Require(capacity > 0 && capacity <= 2048 && capacity <= target_.cfg_.indexer_budget,
            "MTP draft requires dense-QSA text positions within the model indexer budget");
    auto state = std::make_unique<State>();
    state->cache.k = Tensor::allocate(ctx_, {2, capacity, 256}, dnnl::memory::data_type::bf16);
    state->cache.v = Tensor::allocate(ctx_, {2, capacity, 256}, dnnl::memory::data_type::bf16);
    state->cache.indexer_k = Allocate(capacity, 128);
    // Floor block count with a minimum of one, matching the target backend's
    // Session allocation (max(1, capacity / 4)).
    state->cache.indexer_compressed_k = Allocate(std::max(1, capacity / 4), 128);
    state->cache.num_compressed_blocks = 0;
    return state;
}

void Qwen4ExpMtpDraft::Step(State& state, const sycl::ext::oneapi::bfloat16* hidden, int token,
                            int position, bool kvOnly) {
    auto& cache = state.cache;
    const int capacity = static_cast<int>(cache.k.shape(1));
    Require(position >= 0 && position < capacity, "MTP position outside draft capacity");
    auto& b = target_.buf_;
    auto& scratch = target_.linear_scratch_;
    Require(token >= 0 && token < target_.cfg_.vocab_size, "MTP token out of range");
    std::memcpy(hostEmbed_.data(), embeddingTable_.ptr + size_t(token) * 2560 * sizeof(bf16),
                2560 * sizeof(bf16));
    // This blocking copy also protects the reusable host row on exceptions.
    ctx_.memcpy_h2d(embed_.data(), hostEmbed_.data(), 2560 * sizeof(bf16));
    ops::grouped_rms_norm(ctx_, embed_.data_as<bf16>(), normEmbedding_->data_as<float>(), 1e-6f,
                          embedNorm_.data_as<bf16>(), 1, 1, 2560, true);
    fcEmbedding_.ForwardDecodeRows(ctx_, scratch, embedNorm_, embedFc_, 1);
    // Global 10240-wide norm, then the same FC on each 2560-wide stream.
    ops::grouped_rms_norm(ctx_, hidden, normHidden_->data_as<float>(), 1e-6f,
                          hiddenNorm_.data_as<bf16>(), 1, 1, 10240, true);
    fcHidden_.ForwardDecodeRows(ctx_, scratch, hiddenNorm_, b.hidden4, 4);
    auto* h = b.hidden4.data_as<bf16>();
    auto* e = embedFc_.data_as<bf16>();
    ctx_.queue().parallel_for(sycl::range<1>(10240), [=](sycl::id<1> i) {
        h[i] = bf16(float(h[i]) + float(e[i % 2560]));
    });
    target_.RunHyperConnection(ctx_, layer_.attn_hc, 1, true);
    layer_.qkv_proj.ForwardDecodeRows(ctx_, scratch, b.x, b.full_qkv, 1);
    auto* p = b.full_qkv.data_as<bf16>();
    auto q = Tensor::view(ctx_, p, {1, 12288});
    auto k = Tensor::view(ctx_, p + 12288, {1, 512});
    auto v = Tensor::view(ctx_, p + 12800, {1, 512});
    ops::decode_prepare_qgkv_packed_partial(ctx_, q, k, v, b.q, b.z, *layer_.q_norm_weight,
                                            *layer_.k_norm_weight, cache.k, cache.v, position, 24, 2, 256,
                                            1e-6f, 64, 10000000.0f, true, nullptr, nullptr, nullptr, 0, 0, 11,
                                            11, 10,
                                            cache.k_scale.valid() ? &cache.k_scale : nullptr,
                                            cache.v_scale.valid() ? &cache.v_scale : nullptr);
    ops::bf16_gemv_bf16(ctx_, layer_.index_k_weight, b.x.data_as<bf16>(),
                        cache.indexer_k.data_as<bf16>() + size_t(position) * 128, 128, 2560);
    if ((position + 1) % 4 == 0) {
        ops::qsa_compress_key_blocks(ctx_, cache.indexer_k, *layer_.index_k_norm_weight,
                                     cache.indexer_compressed_k, position / 4, 1, 10000000.0f, true, nullptr,
                                     nullptr, nullptr, 0, 0, 11, 11, 10);
    }
    // Keep the original fused QKV and indexer arithmetic. Catch-up only needs
    // these persistent rows; neither attention nor MoE feeds them.
    if (kvOnly) return;
    ops::attention_decode(ctx_, b.q, cache.k, cache.v, b.attn_out, b.decode_scores, 24, 2, 256, position + 1,
                          &b.decode_attn_partials, &b.z, nullptr, 0, nullptr, 0,
                          cache.k_scale.valid() ? &cache.k_scale : nullptr,
                          cache.v_scale.valid() ? &cache.v_scale : nullptr);
    layer_.o_proj.ForwardDecodeRows(ctx_, scratch, b.attn_out, b.y, 1);
    target_.RunHyperConnectionCombined(ctx_, layer_.mlp_hc, 1, true, b.y.data_as<bf16>(),
                                       b.hc_inject_f32.data_as<float>());
    ops::bf16_gemv_bf16(ctx_, *layer_.router_w, b.x.data_as<bf16>(), b.router_scores.data_as<bf16>(), 512,
                        2560);
    auto route = ops::router_softmax_topk_device(ctx_, b.router_scores.data_as<bf16>(), nullptr,
                                                 target_.router_host_ids_, target_.router_host_weights_, 1, 512,
                                                 10, true);
    ops::bf16_gemv_bf16(ctx_, *layer_.shared_expert_gate_w, b.x.data_as<bf16>(),
                        b.shared_gate_buf.data_as<bf16>(), 1, 2560);
    if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(ctx_, layer_.shared_gate_up_proj, b.x,
                                                          b.shared_inter_buf, 640)) {
        layer_.shared_gate_up_proj.ForwardDecodeRows(ctx_, scratch, b.x, b.shared_gate_up_buf, 1);
        ops::moe_silu_gate_up_mul_flat(ctx_, b.shared_gate_up_buf.data_as<bf16>(),
                                       b.shared_inter_buf.data_as<bf16>(), 1, 640);
    }
    layer_.shared_down_proj.ForwardDecodeRows(ctx_, scratch, b.shared_inter_buf, b.shared_out_buf, 1);
    route.wait_and_throw();
    auto ready = experts_.acquire(0, target_.router_host_ids_, 10, slotIds_.data());
    experts_.get_slots_dev(slotIds_.data(), 10, slots_.data());
    ops::moe_decode_gate_up_swiglu_grouped(ctx_, slots_.data(), nullptr, b.x.data_as<bf16>(),
                                           b.moe_h.data_as<bf16>(), 10, 2560, 640, 64, ready);
    auto done = ops::moe_decode_down_weighted_accumulate(
        ctx_, slots_.data(), nullptr, target_.router_host_weights_, b.moe_h.data_as<bf16>(), nullptr,
        b.y.data_as<bf16>(), 10, 640, 2560, 64, b.shared_out_buf.data_as<bf16>(),
        b.shared_gate_buf.data_as<bf16>());
    experts_.record_consumer_reader_fences(slotIds_.data(), 10, done);
    ops::hc_inject_add(ctx_, b.hidden4.data_as<bf16>(), b.y.data_as<bf16>(),
                       b.hc_inject_f32.data_as<float>(), 1, 2560, 4);
    target_.RunHyperConnection(ctx_, mixer_, 1, false);
    target_.lm_head_.forward(ctx_, b.x, b.logits, 1);
}
