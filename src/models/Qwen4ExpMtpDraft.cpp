#include "Qwen4ExpMtpDraft.hpp"

#include "../ops/QsaIndexerOps.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#endif

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
        layer_.index_q_norm_weight = Norm("mtp.layers.0.self_attn.indexer.q_layernorm.weight", true);
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
        draftH_ = Allocate(kMtpExpertsPerStep, kMtpIntermediate);
        draftDown_ = Allocate(kMtpExpertsPerStep, kMtpHidden, dnnl::memory::data_type::f32);
        hostEmbed_ = static_cast<bf16*>(ctx_.alloc_host(
            static_cast<size_t>(kMtpHostEmbedRing) * static_cast<size_t>(kMtpHidden) * sizeof(bf16)));
        embeddingTable_ = weights_.host_view("model.language_model.embed_tokens.weight");
        Require(embeddingTable_.valid() && embeddingTable_.dtype == DT_BF16, "MTP shared embedding missing");

        InitExpertCache();
        ctx_.synchronize();
    } catch (...) {
        try {
            if (copy_q_) copy_q_->wait_and_throw();
            ctx_.synchronize();
        } catch (...) {}
        ReleaseExpertCache();
        FreeHostEmbed();
        throw;
    }
}

Qwen4ExpMtpDraft::~Qwen4ExpMtpDraft() {
    try {
        if (copy_q_) copy_q_->wait_and_throw();
        ctx_.synchronize();
    } catch (...) {}
    ReleaseExpertCache();
    FreeHostEmbed();
}

void Qwen4ExpMtpDraft::FreeHostEmbed() noexcept {
    if (!hostEmbed_) return;
    for (auto& ev : hostEmbedEvents_) {
        try {
            if (ev != sycl::event{}) ev.wait_and_throw();
        } catch (...) {}
        ev = sycl::event{};
    }
    try { ctx_.free_host(hostEmbed_); } catch (...) {}
    hostEmbed_ = nullptr;
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
    for (const auto* suffix :
         {"", ".absmax", ".nested_absmax", ".nested_quant_map", ".quant_state.bitsandbytes__nf4"})
        if (weights_.has(name + suffix)) weights_.erase(name + suffix);
}

void Qwen4ExpMtpDraft::Hc(const std::string& prefix, Qwen4ExpHyperConnectionLayer& hc, bool inject) {
    hc.hc_norm_weight = Norm(prefix + "hc_norm.weight");
    hc.mix_down_w = &weights_.upload(prefix + "input_mix_weight_down.weight", ctx_);
    hc.mix_up_w = &weights_.upload(prefix + "input_mix_weight_up.weight", ctx_);
    if (inject) hc.inject_w = &weights_.upload(prefix + "block_inject_weight.weight", ctx_);
}

void Qwen4ExpMtpDraft::InitExpertCache() {
    std::string error;
    Require(LoadBnb4BitHostRef(weights_, "mtp.layers.0.mlp.experts.gate_up_proj", gu_, &error, kMtpHidden),
            error);
    Require(LoadBnb4BitHostRef(weights_, "mtp.layers.0.mlp.experts.down_proj", dn_, &error, kMtpIntermediate),
            error);
    Require(gu_.quant_state.blocksize == kMtpBlocksize && dn_.quant_state.blocksize == kMtpBlocksize,
            "MTP experts must be NF4 with blocksize 64");
    Require(gu_.quant_state.nested_blocksize == kMtpNestedBlocksize &&
                dn_.quant_state.nested_blocksize == kMtpNestedBlocksize,
            "MTP experts must use nested quantization with blocksize 256");
    Require(gu_.nested_absmax.valid() && dn_.nested_absmax.valid() &&
                gu_.nested_quant_map.numel() == 256 && dn_.nested_quant_map.numel() == 256,
            "MTP expert nested scales are missing");

    layout_.gu_packed_bytes = size_t(2 * kMtpIntermediate) * size_t(kMtpHidden / 2);
    layout_.gu_absmax_bytes = size_t(2 * kMtpIntermediate) * size_t(kMtpHidden / kMtpBlocksize);
    layout_.dn_packed_bytes = size_t(kMtpHidden) * size_t(kMtpIntermediate / 2);
    layout_.dn_absmax_bytes = size_t(kMtpHidden) * size_t(kMtpIntermediate / kMtpBlocksize);
    layout_.gu_packed_offset = 0;
    layout_.gu_absmax_offset = layout_.gu_packed_bytes;
    layout_.dn_packed_offset = layout_.gu_absmax_offset + layout_.gu_absmax_bytes;
    layout_.dn_absmax_offset = layout_.dn_packed_offset + layout_.dn_packed_bytes;
    layout_.slot_bytes = layout_.dn_absmax_offset + layout_.dn_absmax_bytes;
    Require(layout_.slot_bytes == kMtpExpertSlotBytes, "MTP expert slot geometry mismatch");

    const size_t experts = size_t(kMtpExperts);
    Require(gu_.packed_weight.bytes == experts * layout_.gu_packed_bytes &&
                dn_.packed_weight.bytes == experts * layout_.dn_packed_bytes,
            "MTP expert packed weight size mismatch");
    Require(gu_.absmax.bytes == experts * layout_.gu_absmax_bytes &&
                dn_.absmax.bytes == experts * layout_.dn_absmax_bytes,
            "MTP expert absmax code size mismatch (compact nested format required)");

    gu_nested_per_expert_ = layout_.gu_absmax_bytes / size_t(kMtpNestedBlocksize);
    dn_nested_per_expert_ = layout_.dn_absmax_bytes / size_t(kMtpNestedBlocksize);
    Require(gu_.nested_absmax.bytes == size_t(gu_.nested_absmax.numel()) * sizeof(float) &&
                dn_.nested_absmax.bytes == size_t(dn_.nested_absmax.numel()) * sizeof(float) &&
                gu_.nested_absmax.numel() >= int64_t(experts * gu_nested_per_expert_) &&
                dn_.nested_absmax.numel() >= int64_t(experts * dn_nested_per_expert_),
            "MTP expert nested_absmax does not cover 512 experts");
    gu_nested_offset_ = gu_.quant_state.nested_offset;
    dn_nested_offset_ = dn_.quant_state.nested_offset;

    copy_q_ = std::make_unique<sycl::queue>(
        ctx_.queue().get_context(), ctx_.queue().get_device(),
        sycl::property_list{sycl::property::queue::in_order()});

    slot_pool_dev_ = static_cast<uint8_t*>(ctx_.alloc_device(size_t(kMtpExpertSlots) * layout_.slot_bytes));
    slot_staging_host_ =
        static_cast<uint8_t*>(ctx_.alloc_host(size_t(kMtpExpertsPerStep) * layout_.slot_bytes));
    gu_nested_absmax_dev_ =
        static_cast<float*>(ctx_.alloc_device(size_t(gu_.nested_absmax.numel()) * sizeof(float)));
    dn_nested_absmax_dev_ =
        static_cast<float*>(ctx_.alloc_device(size_t(dn_.nested_absmax.numel()) * sizeof(float)));
    gu_nested_quant_map_dev_ = static_cast<float*>(ctx_.alloc_device(256 * sizeof(float)));
    dn_nested_quant_map_dev_ = static_cast<float*>(ctx_.alloc_device(256 * sizeof(float)));

    ctx_.memcpy_h2d(gu_nested_absmax_dev_, gu_.nested_absmax.ptr, gu_.nested_absmax.bytes);
    ctx_.memcpy_h2d(dn_nested_absmax_dev_, dn_.nested_absmax.ptr, dn_.nested_absmax.bytes);
    ctx_.memcpy_h2d(gu_nested_quant_map_dev_, gu_.nested_quant_map.ptr, 256 * sizeof(float));
    ctx_.memcpy_h2d(dn_nested_quant_map_dev_, dn_.nested_quant_map.ptr, 256 * sizeof(float));

    const auto touch_span = [](const uint8_t* ptr, size_t bytes) {
        if (!ptr || bytes == 0) return;
#ifdef _WIN32
        WIN32_MEMORY_RANGE_ENTRY entry{const_cast<uint8_t*>(ptr), bytes};
        (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#endif
        volatile uint8_t sink = 0;
        for (size_t off = 0; off < bytes; off += 4096) {
            sink = static_cast<uint8_t>(sink ^ ptr[off]);
        }
        sink = static_cast<uint8_t>(sink ^ ptr[bytes - 1]);
        (void)sink;
    };
    touch_span(gu_.packed_weight.ptr, gu_.packed_weight.bytes);
    touch_span(gu_.absmax.ptr, gu_.absmax.bytes);
    touch_span(dn_.packed_weight.ptr, dn_.packed_weight.bytes);
    touch_span(dn_.absmax.ptr, dn_.absmax.bytes);

    const float* gu_nested_absmax_host = reinterpret_cast<const float*>(gu_.nested_absmax.ptr);
    const float* dn_nested_absmax_host = reinterpret_cast<const float*>(dn_.nested_absmax.ptr);
    const float* gu_nested_qmap_host = reinterpret_cast<const float*>(gu_.nested_quant_map.ptr);
    const float* dn_nested_qmap_host = reinterpret_cast<const float*>(dn_.nested_quant_map.ptr);
    for (int e = 0; e < kMtpExperts; ++e) {
        auto& hs = host_slots_[size_t(e)];
        hs = ops::MoeExpertSlotHost{};
        hs.gate_up_packed = gu_.packed_weight.ptr + size_t(e) * layout_.gu_packed_bytes;
        hs.down_packed = dn_.packed_weight.ptr + size_t(e) * layout_.dn_packed_bytes;
        hs.gate_up_nested.codes = gu_.absmax.ptr + size_t(e) * layout_.gu_absmax_bytes;
        hs.down_nested.codes = dn_.absmax.ptr + size_t(e) * layout_.dn_absmax_bytes;
        hs.gate_up_nested.nested_absmax = gu_nested_absmax_host + size_t(e) * gu_nested_per_expert_;
        hs.down_nested.nested_absmax = dn_nested_absmax_host + size_t(e) * dn_nested_per_expert_;
        hs.gate_up_nested.nested_quant_map = gu_nested_qmap_host;
        hs.down_nested.nested_quant_map = dn_nested_qmap_host;
        hs.gate_up_nested.offset = gu_nested_offset_;
        hs.down_nested.offset = dn_nested_offset_;
        hs.gate_up_nested.blocksize = kMtpNestedBlocksize;
        hs.down_nested.blocksize = kMtpNestedBlocksize;
    }

    for (auto& s : slot_pool_) s = SlotEntry{};
    for (int i = 0; i < kMtpExpertsPerStep; ++i) staging_copy_valid_[i] = false;
    tick_ = 0;
    for (int s = 0; s < kMtpExpertSlots; ++s) {
        StageExpert(slot_staging_host_, s);
        ctx_.memcpy_h2d(slot_pool_dev_ + size_t(s) * layout_.slot_bytes,
                        slot_staging_host_, layout_.slot_bytes);
        slot_pool_[size_t(s)].expert = s;
        slot_pool_[size_t(s)].last_used = ++tick_;
    }
}

void Qwen4ExpMtpDraft::ReleaseExpertCache() noexcept {
    if (copy_q_) {
        try { copy_q_->wait_and_throw(); } catch (...) {}
    }
    for (auto& s : slot_pool_) s = SlotEntry{};
    if (slot_pool_dev_) {
        try { ctx_.free_device(slot_pool_dev_); } catch (...) {}
        slot_pool_dev_ = nullptr;
    }
    if (slot_staging_host_) {
        try { ctx_.free_host(slot_staging_host_); } catch (...) {}
        slot_staging_host_ = nullptr;
    }
    if (gu_nested_absmax_dev_) {
        try { ctx_.free_device(gu_nested_absmax_dev_); } catch (...) {}
        gu_nested_absmax_dev_ = nullptr;
    }
    if (dn_nested_absmax_dev_) {
        try { ctx_.free_device(dn_nested_absmax_dev_); } catch (...) {}
        dn_nested_absmax_dev_ = nullptr;
    }
    if (gu_nested_quant_map_dev_) {
        try { ctx_.free_device(gu_nested_quant_map_dev_); } catch (...) {}
        gu_nested_quant_map_dev_ = nullptr;
    }
    if (dn_nested_quant_map_dev_) {
        try { ctx_.free_device(dn_nested_quant_map_dev_); } catch (...) {}
        dn_nested_quant_map_dev_ = nullptr;
    }
    copy_q_.reset();
}

void Qwen4ExpMtpDraft::StageExpert(uint8_t* dst, int expert) const {
    const size_t e = size_t(expert);
    std::memcpy(dst + layout_.gu_packed_offset, gu_.packed_weight.ptr + e * layout_.gu_packed_bytes,
                layout_.gu_packed_bytes);
    std::memcpy(dst + layout_.gu_absmax_offset, gu_.absmax.ptr + e * layout_.gu_absmax_bytes,
                layout_.gu_absmax_bytes);
    std::memcpy(dst + layout_.dn_packed_offset, dn_.packed_weight.ptr + e * layout_.dn_packed_bytes,
                layout_.dn_packed_bytes);
    std::memcpy(dst + layout_.dn_absmax_offset, dn_.absmax.ptr + e * layout_.dn_absmax_bytes,
                layout_.dn_absmax_bytes);
}

void Qwen4ExpMtpDraft::BuildSlotDev(int slot, int expert, ops::MoeExpertSlotDev& out) const {
    out = ops::MoeExpertSlotDev{};
    if (slot < 0 || slot >= kMtpExpertSlots || expert < 0 || expert >= kMtpExperts) return;
    const uint8_t* base = slot_pool_dev_ + size_t(slot) * layout_.slot_bytes;
    out.gate_up_packed = base + layout_.gu_packed_offset;
    out.down_packed = base + layout_.dn_packed_offset;
    out.gate_up_nested.codes = base + layout_.gu_absmax_offset;
    out.down_nested.codes = base + layout_.dn_absmax_offset;
    out.gate_up_nested.nested_absmax = gu_nested_absmax_dev_ + size_t(expert) * gu_nested_per_expert_;
    out.down_nested.nested_absmax = dn_nested_absmax_dev_ + size_t(expert) * dn_nested_per_expert_;
    out.gate_up_nested.nested_quant_map = gu_nested_quant_map_dev_;
    out.down_nested.nested_quant_map = dn_nested_quant_map_dev_;
    out.gate_up_nested.offset = gu_nested_offset_;
    out.down_nested.offset = dn_nested_offset_;
    out.gate_up_nested.blocksize = kMtpNestedBlocksize;
    out.down_nested.blocksize = kMtpNestedBlocksize;
}

sycl::event Qwen4ExpMtpDraft::AcquireExperts(const int* expert_ids, int count, int* out_slots) {
    Require(expert_ids != nullptr && out_slots != nullptr && count > 0 && count <= kMtpExpertsPerStep,
            "MTP expert acquire bounds");
    Require(copy_q_ && slot_pool_dev_ && slot_staging_host_, "MTP expert cache is not initialized");

    std::array<int, kMtpExpertsPerStep> requested{};
    std::array<bool, kMtpExpertSlots> reserved{};
    for (int i = 0; i < count; ++i) {
        const int e = expert_ids[i];
        Require(e >= 0 && e < kMtpExperts, "MTP expert id out of range");
        requested[i] = e;
        out_slots[i] = -1;
        for (int j = 0; j < i; ++j) {
            if (requested[j] == e) {
                out_slots[i] = out_slots[j];
                break;
            }
        }
        if (out_slots[i] >= 0) {
            reserved[out_slots[i]] = true;
            continue;
        }
        for (int s = 0; s < kMtpExpertSlots; ++s) {
            if (slot_pool_[s].expert != e) continue;
            slot_pool_[s].last_used = ++tick_;
            out_slots[i] = s;
            reserved[s] = true;
            break;
        }
    }

    sycl::event last{};
    bool copied = false;
    for (int i = 0; i < count; ++i) {
        if (out_slots[i] >= 0) continue;
        int victim = -1;
        uint64_t oldest = 0;
        for (int s = 0; s < kMtpExpertSlots; ++s) {
            if (reserved[s]) continue;
            if (victim < 0 || slot_pool_[s].last_used < oldest) {
                victim = s;
                oldest = slot_pool_[s].last_used;
            }
        }
        Require(victim >= 0, "MTP expert slot starvation");
        reserved[victim] = true;
        if (slot_pool_[victim].reader_valid) {
            slot_pool_[victim].reader.wait_and_throw();
            slot_pool_[victim].reader_valid = false;
        }
        if (staging_copy_valid_[i]) {
            staging_copy_[i].wait_and_throw();
            staging_copy_valid_[i] = false;
        }
        uint8_t* staged = slot_staging_host_ + size_t(i) * layout_.slot_bytes;
        StageExpert(staged, requested[i]);
        last = copy_q_->memcpy(slot_pool_dev_ + size_t(victim) * layout_.slot_bytes, staged,
                               layout_.slot_bytes);
        staging_copy_[i] = last;
        staging_copy_valid_[i] = true;
        copied = true;
        slot_pool_[victim].expert = requested[i];
        slot_pool_[victim].last_used = ++tick_;
        out_slots[i] = victim;
    }

    for (int i = 0; i < count; ++i) BuildSlotDev(out_slots[i], requested[i], slots_[i]);
    return copied ? last : sycl::event{};
}

void Qwen4ExpMtpDraft::RecordSlotReaders(const int* slots, int count, const sycl::event& consumer) {
    for (int i = 0; i < count; ++i) {
        const int s = slots[i];
        if (s < 0 || s >= kMtpExpertSlots) continue;
        slot_pool_[s].reader = consumer;
        slot_pool_[s].reader_valid = true;
    }
}

std::unique_ptr<Qwen4ExpMtpDraft::State> Qwen4ExpMtpDraft::CreateState(int capacity) {
    Require(capacity > 0 && capacity <= target_.max_seq_len(),
            "MTP draft capacity exceeds the target sequence budget");
    auto state = std::make_unique<State>();
    state->cache.k = Tensor::allocate(ctx_, {2, capacity, 256}, dnnl::memory::data_type::bf16);
    state->cache.v = Tensor::allocate(ctx_, {2, capacity, 256}, dnnl::memory::data_type::bf16);
    state->cache.indexer_k = Allocate(capacity, 128);
    state->cache.indexer_compressed_k = Allocate(std::max(1, capacity / 4), 128);
    state->cache.num_compressed_blocks = 0;
    return state;
}

void Qwen4ExpMtpDraft::Step(State& state, const sycl::ext::oneapi::bfloat16* hidden, int token,
                            int position, bool kvOnly,
                            void (*onBeforeCpuCold)(void*), void* onBeforeCpuColdCtx) {
    auto& cache = state.cache;
    const int capacity = static_cast<int>(cache.k.shape(1));
    Require(position >= 0 && position < capacity, "MTP position outside draft capacity");
    auto& b = target_.buf_;
    auto& scratch = target_.linear_scratch_;
    Require(token >= 0 && token < target_.cfg_.vocab_size, "MTP token out of range");
    // The previous step's async row upload must have finished reading the
    // staging block before it is overwritten. Waiting on that copy keeps the
    // compute queue running, unlike the blocking ctx_.memcpy_h2d it replaces.
    const size_t ringSlot = static_cast<size_t>((hostEmbedTick_++) % kMtpHostEmbedRing);
    const size_t fenceSlot = ringSlot | 15u;
    bf16* stagedRow = hostEmbed_ + ringSlot * static_cast<size_t>(kMtpHidden);
    if (hostEmbedEvents_[fenceSlot] != sycl::event{}) {
        hostEmbedEvents_[fenceSlot].wait_and_throw();
        hostEmbedEvents_[fenceSlot] = sycl::event{};
    }
    std::memcpy(stagedRow, embeddingTable_.ptr + size_t(token) * 2560 * sizeof(bf16),
                2560 * sizeof(bf16));
    ops::grouped_rms_norm(ctx_, stagedRow, normEmbedding_->data_as<float>(), 1e-6f,
                          embedNorm_.data_as<bf16>(), 1, 1, 2560, true);
    if ((ringSlot & 15u) == 15u)
        hostEmbedEvents_[ringSlot] = ctx_.queue().ext_oneapi_submit_barrier();
    fcEmbedding_.ForwardDecodeRows(ctx_, scratch, embedNorm_, embedFc_, 1);
    ops::grouped_rms_norm(ctx_, hidden, normHidden_->data_as<float>(), 1e-6f,
                          hiddenNorm_.data_as<bf16>(), 1, 1, 10240, true);
    fcHidden_.ForwardDecodeRows(ctx_, scratch, hiddenNorm_, b.hidden4, 4);
    auto* h = b.hidden4.data_as<bf16>();
    auto* e = embedFc_.data_as<bf16>();
    ctx_.queue().parallel_for(sycl::range<1>(10240), [=](sycl::id<1> i) {
        h[i] = bf16(float(h[i]) + float(e[i % 2560]));
    });
    target_.RunHyperConnectionCombined(ctx_, layer_.attn_hc, 1, true, nullptr, nullptr);
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
    const int* active_blocks = nullptr;
    int num_active_blocks = 0;
    const bool sparse_qsa = !kvOnly && ((position + 1) / 4 > 512) &&
                            layer_.index_q_norm_weight != nullptr &&
                            cache.indexer_compressed_k.valid();
    if (!sparse_qsa) {
        ops::bf16_gemv_bf16(ctx_, layer_.index_k_weight, b.x.data_as<bf16>(),
                            cache.indexer_k.data_as<bf16>() + size_t(position) * 128, 128, 2560);
        if ((position + 1) % 4 == 0) {
            ops::qsa_compress_key_blocks(ctx_, cache.indexer_k, *layer_.index_k_norm_weight,
                                         cache.indexer_compressed_k, position / 4, 1, 10000000.0f, true, nullptr,
                                         nullptr, nullptr, 0, 0, 11, 11, 10);
            cache.num_compressed_blocks = (position + 1) / 4;
        }
        if (kvOnly) return;
    } else {
        ops::bf16_gemv_bf16(ctx_, *layer_.index_qk_weight, b.x.data_as<bf16>(),
                            b.indexer_qk.data_as<bf16>(), 640, 2560);
        num_active_blocks = ops::qsa_prepare_causal_query_and_select_blocks(
            ctx_, b.indexer_qk, *layer_.index_q_norm_weight, *layer_.index_k_norm_weight,
            cache.indexer_k, b.indexer_q, b.indexer_block_scores, b.indexer_active_blocks,
            position, 512, 4, 10000000.0f, true, nullptr, nullptr, nullptr, 0, 0, 11, 11, 10,
            &cache.indexer_compressed_k, &cache.num_compressed_blocks);
        if (num_active_blocks > 0) {
            active_blocks = b.indexer_active_blocks.data_as<int>();
        }
    }
    ops::attention_decode(ctx_, b.q, cache.k, cache.v, b.attn_out, b.decode_scores, 24, 2, 256, position + 1,
                          &b.decode_attn_partials, &b.z, nullptr, 0,
                          active_blocks, num_active_blocks,
                          cache.k_scale.valid() ? &cache.k_scale : nullptr,
                          cache.v_scale.valid() ? &cache.v_scale : nullptr);
    layer_.o_proj.ForwardDecodeRows(ctx_, scratch, b.attn_out, b.y, 1);
    target_.RunHyperConnectionCombined(ctx_, layer_.mlp_hc, 1, true, b.y.data_as<bf16>(),
                                       b.hc_inject_f32.data_as<float>());
    const bool use_cpu_cold = target_.expert_executor_.cpu_overlap_enabled() &&
                              target_.expert_executor_.x_host_ptr() != nullptr;
    ops::bf16_moe_front_fused(ctx_, layer_.router_w->data_as<bf16>(),
                              layer_.shared_expert_gate_w->data_as<bf16>(), b.x.data_as<bf16>(),
                              b.router_scores.data_as<bf16>(), b.shared_gate_buf.data_as<bf16>(),
                              nullptr, nullptr, 2560, 512,
                              use_cpu_cold ? target_.expert_executor_.x_host_ptr() : nullptr);
    auto route = ops::router_softmax_topk_device(ctx_, b.router_scores.data_as<bf16>(), nullptr,
                                                 target_.router_host_ids_, target_.router_host_weights_, 1, 512,
                                                 10, true);
    try { (void)route.get_info<sycl::info::event::command_execution_status>(); } catch (...) {}
    if (!Bnb4BitLinear::try_forward_decode_gate_up_swiglu(ctx_, layer_.shared_gate_up_proj, b.x,
                                                          b.shared_inter_buf, 640)) {
        layer_.shared_gate_up_proj.ForwardDecodeRows(ctx_, scratch, b.x, b.shared_gate_up_buf, 1);
        ops::moe_silu_gate_up_mul_flat(ctx_, b.shared_gate_up_buf.data_as<bf16>(),
                                       b.shared_inter_buf.data_as<bf16>(), 1, 640);
    }
    layer_.shared_down_proj.ForwardDecodeRows(ctx_, scratch, b.shared_inter_buf, b.shared_out_buf, 1);
    route.wait_and_throw();
    if (use_cpu_cold) {
        target_.expert_executor_.MarkComputeQueueSyncedAfterRoute();
        std::array<ops::MoeExpertSlotDev, kMtpExpertsPerStep> gpu_slots{};
        std::array<int, kMtpExpertsPerStep> gpu_ranks{};
        std::array<ops::MoeExpertSlotHost, kMtpExpertsPerStep> cpu_sources{};
        std::array<int, kMtpExpertsPerStep> cpu_ids{}, cpu_ranks{};
        int gpu_count = 0, cpu_count = 0;
        for (int r = 0; r < kMtpExpertsPerStep; ++r) {
            const int e = target_.router_host_ids_[r];
            Require(e >= 0 && e < kMtpExperts, "MTP expert id out of range");
            int hit_slot = -1;
            for (int s = 0; s < kMtpExpertSlots; ++s) {
                if (slot_pool_[size_t(s)].expert == e) {
                    hit_slot = s;
                    slot_pool_[size_t(s)].last_used = ++tick_;
                    break;
                }
            }
            if (hit_slot >= 0) {
                BuildSlotDev(hit_slot, e, gpu_slots[size_t(gpu_count)]);
                gpu_ranks[size_t(gpu_count)] = r;
                ++gpu_count;
            } else {
                cpu_sources[size_t(cpu_count)] = host_slots_[size_t(e)];
                cpu_ids[size_t(cpu_count)] = e;
                cpu_ranks[size_t(cpu_count)] = r;
                ++cpu_count;
            }
        }
        sycl::event last_gpu{};
        bool gpu_live = false;
        if (gpu_count > 0) {
            last_gpu = ops::moe_decode_gate_up_swiglu_grouped(
                ctx_, gpu_slots.data(), nullptr, b.x.data_as<bf16>(),
                draftH_.data_as<bf16>(), gpu_count, 2560, 640, 64,
                nullptr, nullptr, true, true);
            last_gpu = ops::moe_decode_down_to_rank_f32(
                ctx_, gpu_slots.data(), gpu_ranks.data(), draftH_.data_as<bf16>(),
                draftDown_.data_as<float>(), gpu_count, kMtpExpertsPerStep, 640, 2560, 64,
                nullptr, nullptr, &last_gpu, true, true);
            gpu_live = true;
        }
        if (cpu_count > 0) {
            if (onBeforeCpuCold) {
                onBeforeCpuCold(onBeforeCpuColdCtx);
                onBeforeCpuCold = nullptr;
            }
            last_gpu = ctx_.queue().ext_oneapi_submit_barrier();
            gpu_live = true;
            try { (void)last_gpu.get_info<sycl::info::event::command_execution_status>(); } catch (...) {}
            const auto* cpu_rows = target_.expert_executor_.ExecuteDraftCold(
                cpu_sources.data(), cpu_ids.data(), cpu_ranks.data(), cpu_count);
            ops::MoeCpuHostDownRows rows_with_weights = *cpu_rows;
            rows_with_weights.has_host_weights = true;
            for (int r = 0; r < kMtpExpertsPerStep; ++r) {
                rows_with_weights.host_weights[size_t(r)] = target_.router_host_weights_[r];
            }
            auto done = ops::moe_ordered_accumulate_cpu_host_rows(
                ctx_, draftDown_.data_as<float>(), nullptr, b.y.data_as<bf16>(),
                1, kMtpExpertsPerStep, 2560, rows_with_weights,
                b.shared_out_buf.data_as<bf16>(), b.shared_gate_buf.data_as<bf16>(),
                gpu_live ? &last_gpu : nullptr);
            target_.expert_executor_.RecordDraftMergeEvent(done);
        } else {
            ops::MoeCpuHostDownRows empty_rows{};
            empty_rows.values = target_.expert_executor_.cpu_out_host_ptr();
            empty_rows.rows = 1;
            empty_rows.rank_to_row.fill(-1);
            empty_rows.has_host_weights = true;
            for (int r = 0; r < kMtpExpertsPerStep; ++r) {
                empty_rows.host_weights[size_t(r)] = target_.router_host_weights_[r];
            }
            (void)ops::moe_ordered_accumulate_cpu_host_rows(
                ctx_, draftDown_.data_as<float>(), nullptr, b.y.data_as<bf16>(),
                1, kMtpExpertsPerStep, 2560, empty_rows,
                b.shared_out_buf.data_as<bf16>(), b.shared_gate_buf.data_as<bf16>(),
                gpu_live ? &last_gpu : nullptr);
        }
    } else {
        slot_ready_ = AcquireExperts(target_.router_host_ids_, kMtpExpertsPerStep, slotIds_.data());
        slot_ready_valid_ = (slot_ready_ != sycl::event{});
        ops::moe_decode_gate_up_swiglu_grouped(ctx_, slots_.data(), nullptr, b.x.data_as<bf16>(),
                                               b.moe_h.data_as<bf16>(), kMtpExpertsPerStep, 2560, 640, 64,
                                               slot_ready_valid_ ? &slot_ready_ : nullptr);
        auto done = ops::moe_decode_down_weighted_accumulate(
            ctx_, slots_.data(), nullptr, target_.router_host_weights_, b.moe_h.data_as<bf16>(), nullptr,
            b.y.data_as<bf16>(), kMtpExpertsPerStep, 640, 2560, 64, b.shared_out_buf.data_as<bf16>(),
            b.shared_gate_buf.data_as<bf16>());
        RecordSlotReaders(slotIds_.data(), kMtpExpertsPerStep, done);
    }
    target_.RunHyperConnectionCombined(ctx_, mixer_, 1, false,
                                       b.y.data_as<bf16>(), b.hc_inject_f32.data_as<float>());
    target_.ForwardLmHead(ctx_, b.x, b.logits, 1);
}
