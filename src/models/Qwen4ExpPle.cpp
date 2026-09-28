#include "Qwen4ExpPle.hpp"
#include "../utils/EnvUtils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static constexpr float kDefaultNf4QuantMap[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.4407079219818115f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

Qwen4ExpPle::~Qwen4ExpPle() {
    cleanup();
}

Qwen4ExpPle::Qwen4ExpPle(Qwen4ExpPle&& other) noexcept {
    *this = std::move(other);
}

Qwen4ExpPle& Qwen4ExpPle::operator=(Qwen4ExpPle&& other) noexcept {
    if (this == &other) return *this;
    cleanup();

    ctx_ = other.ctx_;
    ple_layer_id_ = other.ple_layer_id_;
    max_tokens_ = other.max_tokens_;
    eps_ = other.eps_;

    layer_multipliers_ = other.layer_multipliers_;
    ngram_heads_vocab_sizes_ = other.ngram_heads_vocab_sizes_;
    ngram_heads_offsets_ = other.ngram_heads_offsets_;

    shards_ = std::move(other.shards_);
    rows_per_shard_ = other.rows_per_shard_;

    key_proj_ = std::move(other.key_proj_);
    value_proj_ = std::move(other.value_proj_);
    norm_key_w_ = std::move(other.norm_key_w_);
    norm_query_w_ = std::move(other.norm_query_w_);
    norm_conv_w_ = std::move(other.norm_conv_w_);
    owned_conv1d_w_ = std::move(other.owned_conv1d_w_);
    conv1d_w_ptr_ = other.conv1d_w_ptr_;

    scratch_key_ = std::move(other.scratch_key_);
    scratch_val_ = std::move(other.scratch_val_);

    key_buf_ = std::move(other.key_buf_);
    value_buf_ = std::move(other.value_buf_);
    key_normed_ = std::move(other.key_normed_);
    query_normed_ = std::move(other.query_normed_);
    gv_buf_ = std::move(other.gv_buf_);
    gv_normed_ = std::move(other.gv_normed_);
    initialized_ = other.initialized_;

    other.cleanup();
    return *this;
}

void Qwen4ExpPle::cleanup() {
    if (worker_thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            stop_worker_ = true;
            worker_cv_.notify_all();
        }
        worker_thread_.join();
    }
    task_queue_.clear();
    stop_worker_ = false;
    current_task_id_ = 0;
    completed_task_id_ = 0;
    next_staging_slot_ = 0;
    slot_task_ids_.fill(0);

    if (copy_q_) {
        try { copy_q_->wait_and_throw(); } catch (...) {}
        copy_q_.reset();
    }
    if (ctx_) {
        try {
            ctx_->synchronize();
        } catch (...) {}
        for (int s = 0; s < kNumStagingSlots; ++s) {
            if (pinned_staging_embed_slots_[s]) {
                ctx_->free_host(pinned_staging_embed_slots_[s]);
                pinned_staging_embed_slots_[s] = nullptr;
            }
            dev_embed_slots_[s] = Tensor();
            slot_ngram_ids_[s].clear();
            slot_input_ids_[s].clear();
            slot_history_[s].clear();
            slot_h2d_events_[s] = sycl::event{};
        }
    }
    pinned_staging_embed_ = nullptr;
    ctx_ = nullptr;
    conv1d_w_ptr_ = nullptr;
    staging_embed_bytes_ = 0;
    shards_.clear();
    rows_per_shard_ = 0;
    dev_embed_ = Tensor();
    key_buf_ = Tensor();
    value_buf_ = Tensor();
    key_normed_ = Tensor();
    query_normed_ = Tensor();
    gv_buf_ = Tensor();
    gv_normed_ = Tensor();
    norm_key_w_ = Tensor();
    norm_query_w_ = Tensor();
    norm_conv_w_ = Tensor();
    owned_conv1d_w_ = Tensor();
    initialized_ = false;
}

namespace {

bool upload_and_load_bnb(Context& ctx, ModelWeights& weights, const std::string& name,
                         Bnb4BitWeightRef& ref, std::string* err, int64_t in_hint) {
    if (weights.has_host(name)) {
        weights.upload(name, ctx);
        if (weights.has_host(name + ".absmax")) weights.upload(name + ".absmax", ctx);
        if (weights.has_host(name + ".quant_map")) weights.upload(name + ".quant_map", ctx);
        if (weights.has_host(name + ".nested_absmax")) weights.upload(name + ".nested_absmax", ctx);
        if (weights.has_host(name + ".nested_quant_map")) weights.upload(name + ".nested_quant_map", ctx);
        if (weights.has_host(name + ".quant_state.bitsandbytes__nf4")) weights.upload(name + ".quant_state.bitsandbytes__nf4", ctx);
    }
    return LoadBnb4BitWeightRef(ctx, weights, name, ref, err, in_hint);
}

void load_norm_as_f32(Context& ctx, ModelWeights& weights, const std::string& name, Tensor& out_tensor) {
    out_tensor = Tensor::allocate(ctx, {Qwen4ExpPle::HC_HIDDEN_SIZE}, dnnl::memory::data_type::f32);
    if (weights.has_host(name)) {
        auto view = weights.host_view(name);
        const bf16* src = reinterpret_cast<const bf16*>(view.ptr);
        std::vector<float> f32_vals(Qwen4ExpPle::HC_HIDDEN_SIZE);
        for (int i = 0; i < Qwen4ExpPle::HC_HIDDEN_SIZE; ++i) {
            f32_vals[i] = static_cast<float>(src[i]);
        }
        ctx.memcpy_h2d(out_tensor.data(), f32_vals.data(), f32_vals.size() * sizeof(float));
    } else if (weights.has(name)) {
        Tensor& t = weights.get(name);
        std::vector<bf16> src(Qwen4ExpPle::HC_HIDDEN_SIZE);
        ctx.memcpy_d2h(src.data(), t.data(), src.size() * sizeof(bf16));
        std::vector<float> f32_vals(Qwen4ExpPle::HC_HIDDEN_SIZE);
        for (int i = 0; i < Qwen4ExpPle::HC_HIDDEN_SIZE; ++i) {
            f32_vals[i] = static_cast<float>(src[i]);
        }
        ctx.memcpy_h2d(out_tensor.data(), f32_vals.data(), f32_vals.size() * sizeof(float));
    } else {
        ctx.queue().memset(out_tensor.data(), 0, out_tensor.size_bytes()).wait_and_throw();
    }
}

} // namespace

bool Qwen4ExpPle::init(Context& ctx,
                       ModelWeights& weights,
                       int ple_layer_id,
                       int max_tokens,
                       float eps,
                       std::string* error_message) {
    cleanup();

    ctx_ = &ctx;
    ple_layer_id_ = ple_layer_id;
    max_tokens_ = max_tokens > 0 ? max_tokens : 2048;
    eps_ = eps;

    const std::string prefix = "model.language_model.layers." + std::to_string(ple_layer_id_) + ".ple.";

    // 1. Layer multipliers
    std::string mult_key = prefix + "ple_embedding.layer_multipliers";
    if (weights.has_host(mult_key)) {
        auto view = weights.host_view(mult_key);
        std::memcpy(layer_multipliers_.data(), view.ptr, 3 * sizeof(int64_t));
    } else if (weights.has(mult_key)) {
        ctx.memcpy_d2h(layer_multipliers_.data(), weights.get(mult_key).data(), 3 * sizeof(int64_t));
    } else {
        layer_multipliers_ = {23703573157769LL, 20109073645365LL, 8052911324071LL};
    }

    // 2. Ngram heads vocab sizes
    std::string vocab_key = prefix + "ple_embedding.ngram_heads_vocab_sizes";
    if (weights.has_host(vocab_key)) {
        auto view = weights.host_view(vocab_key);
        std::memcpy(ngram_heads_vocab_sizes_.data(), view.ptr, 16 * sizeof(int64_t));
    } else if (weights.has(vocab_key)) {
        ctx.memcpy_d2h(ngram_heads_vocab_sizes_.data(), weights.get(vocab_key).data(), 16 * sizeof(int64_t));
    } else {
        ngram_heads_vocab_sizes_ = {
            20000003LL, 20000023LL, 20000033LL, 20000047LL, 20000059LL, 20000063LL, 20000069LL, 20000077LL,
            20000081LL, 20000093LL, 20000107LL, 20000147LL, 20000153LL, 20000159LL, 20000161LL, 20000171LL
        };
    }

    // 3. Ngram heads offsets
    std::string offset_key = prefix + "ple_embedding.ngram_heads_offsets";
    if (weights.has_host(offset_key)) {
        auto view = weights.host_view(offset_key);
        std::memcpy(ngram_heads_offsets_.data(), view.ptr, 16 * sizeof(int64_t));
    } else if (weights.has(offset_key)) {
        ctx.memcpy_d2h(ngram_heads_offsets_.data(), weights.get(offset_key).data(), 16 * sizeof(int64_t));
    } else {
        int64_t cur = 0;
        for (int h = 0; h < 16; ++h) {
            ngram_heads_offsets_[h] = cur;
            cur += ngram_heads_vocab_sizes_[h];
        }
    }

    // 4. Index 128 embedding shards from host_view
    shards_.clear();
    shards_.reserve(NUM_SHARDS);
    for (int s = 0; s < NUM_SHARDS; ++s) {
        std::string shard_w = prefix + "ple_embedding.ngram_embedding.shard_" + std::to_string(s) + ".weight";
        if (!weights.has_host(shard_w)) break;

        auto w_view = weights.host_view(shard_w);
        auto a_view = weights.host_view(shard_w + ".absmax");
        auto q_view = weights.host_view(shard_w + ".quant_map");

        PleShardRef ref;
        ref.packed_weight = w_view.ptr;
        ref.absmax = reinterpret_cast<const float*>(a_view.ptr);
        ref.quant_map = reinterpret_cast<const float*>(q_view.ptr);
        ref.rows = w_view.shape.size() > 0 ? (w_view.shape[0] / 80) : 0;
        shards_.push_back(ref);
    }
    if (!shards_.empty()) {
        rows_per_shard_ = shards_[0].rows;
    }

    // 5. Linear projections key_proj & value_proj
    static const bool s_q4e_async_dequant = aila::env::read_flag("AILA_Q4E_ASYNC_DEQUANT", true);
    static const bool s_q4e_fused_prefill = aila::env::read_flag("AILA_Q4E_FUSED_PREFILL", false);
    std::string key_name = prefix + "key_proj.weight";
    std::string val_name = prefix + "value_proj.weight";
    if (weights.has_host(key_name) || weights.has(key_name)) {
        Bnb4BitWeightRef key_ref;
        if (!upload_and_load_bnb(ctx, weights, key_name, key_ref, error_message, EMBED_DIM)) {
            return false;
        }
        if (!key_proj_.init(ctx, key_ref, error_message)) {
            return false;
        }
        if (s_q4e_async_dequant) {
            key_proj_.set_allow_async_dequant(true);
        }
        if (s_q4e_fused_prefill) {
            key_proj_.set_allow_dense_xmx(true);
        }
    }
    if (weights.has_host(val_name) || weights.has(val_name)) {
        Bnb4BitWeightRef val_ref;
        if (!upload_and_load_bnb(ctx, weights, val_name, val_ref, error_message, EMBED_DIM)) {
            return false;
        }
        if (!value_proj_.init(ctx, val_ref, error_message)) {
            return false;
        }
        if (s_q4e_async_dequant) {
            value_proj_.set_allow_async_dequant(true);
        }
        if (s_q4e_fused_prefill) {
            value_proj_.set_allow_dense_xmx(true);
        }
    }

    // 6. Layer norms
    load_norm_as_f32(ctx, weights, prefix + "norm_key.weight", norm_key_w_);
    load_norm_as_f32(ctx, weights, prefix + "norm_query.weight", norm_query_w_);
    load_norm_as_f32(ctx, weights, prefix + "norm_conv.weight", norm_conv_w_);

    // 7. Conv1d weight [10240, 1, 4] bf16
    std::string conv_name = prefix + "conv1d.weight";
    if (weights.has_host(conv_name)) {
        weights.upload(conv_name, ctx);
    }
    if (weights.has(conv_name)) {
        conv1d_w_ptr_ = static_cast<const bf16*>(weights.get(conv_name).data());
    } else {
        owned_conv1d_w_ = Tensor::allocate(ctx, {HC_HIDDEN_SIZE, CONV_KERNEL_SIZE}, dnnl::memory::data_type::bf16);
        ctx.queue().memset(owned_conv1d_w_.data(), 0, owned_conv1d_w_.size_bytes()).wait_and_throw();
        conv1d_w_ptr_ = static_cast<const bf16*>(owned_conv1d_w_.data());
    }

    // 8. Staging host buffers (double-buffered) and device tensors for DDR3 -> PCIe DMA
    staging_embed_bytes_ = static_cast<size_t>(max_tokens_) * EMBED_DIM * sizeof(bf16);
    try {
        for (int s = 0; s < kNumStagingSlots; ++s) {
            pinned_staging_embed_slots_[s] = ctx.alloc_host(staging_embed_bytes_);
            dev_embed_slots_[s] = Tensor::allocate(ctx, {max_tokens_, EMBED_DIM}, dnnl::memory::data_type::bf16);
            slot_ngram_ids_[s].resize(static_cast<size_t>(max_tokens_) * NUM_HEADS);
            slot_input_ids_[s].resize(static_cast<size_t>(max_tokens_));
            slot_history_[s].resize(static_cast<size_t>(max_tokens_) + Qwen4ExpPleState::NGRAM_CONTEXT_LEN);
            slot_h2d_events_[s] = sycl::event{};
        }
        pinned_staging_embed_ = pinned_staging_embed_slots_[0];
        dev_embed_ = Tensor::view(ctx, dev_embed_slots_[0].data(), {max_tokens_, EMBED_DIM});
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Qwen4ExpPle pinned staging alloc failed: ") + e.what();
        cleanup();
        return false;
    }

    // 9. Pre-allocated device tensors for forward execution (hot path zero-alloc)
    key_buf_ = Tensor::allocate(ctx, {max_tokens_, HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
    value_buf_ = Tensor::allocate(ctx, {max_tokens_, EMBED_DIM}, dnnl::memory::data_type::bf16);
    key_normed_ = Tensor::allocate(ctx, {max_tokens_, HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
    query_normed_ = Tensor::allocate(ctx, {max_tokens_, HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
    gv_buf_ = Tensor::allocate(ctx, {max_tokens_, HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);
    gv_normed_ = Tensor::allocate(ctx, {max_tokens_, HC_HIDDEN_SIZE}, dnnl::memory::data_type::bf16);

    scratch_key_.ensure(ctx, max_tokens_, EMBED_DIM, HC_HIDDEN_SIZE, false);
    scratch_val_.ensure(ctx, max_tokens_, EMBED_DIM, EMBED_DIM, false);

    try {
        copy_q_ = std::make_unique<sycl::queue>(
            ctx.queue().get_context(), ctx.queue().get_device(),
            sycl::property_list{sycl::property::queue::in_order()});
    } catch (...) {
        copy_q_.reset();
    }

    stop_worker_ = false;
    task_queue_.clear();
    current_task_id_ = 0;
    completed_task_id_ = 0;
    next_staging_slot_ = 0;
    slot_task_ids_.fill(0);
    static const bool s_ple_async = aila::env::read_flag("AILA_Q4E_PLE_ASYNC", true);
    if (s_ple_async) {
        worker_thread_ = std::thread([this] { worker_loop(); });
    }

    initialized_ = true;
    return true;
}

void Qwen4ExpPle::set_shards(const std::vector<PleShardRef>& shards, int64_t rows_per_shard) {
    shards_ = shards;
    rows_per_shard_ = rows_per_shard;
}

void Qwen4ExpPle::set_multipliers_and_vocab(
    const std::array<int64_t, 3>& multipliers,
    const std::array<int64_t, 16>& vocab_sizes,
    const std::array<int64_t, 16>& offsets) {
    layer_multipliers_ = multipliers;
    ngram_heads_vocab_sizes_ = vocab_sizes;
    ngram_heads_offsets_ = offsets;
}

void Qwen4ExpPle::compute_ngram_ids(
    const int* token_history,
    int total_tokens,
    int num_tokens,
    const int64_t* layer_multipliers,
    const int64_t* vocab_sizes,
    const int64_t* offsets,
    int64_t* out_ids,
    int eos_token_id) {

    if (total_tokens <= 0 || num_tokens <= 0) return;

    int last_eos = -1;
    const int start_t = total_tokens - num_tokens;

    for (int i = 0; i < total_tokens; ++i) {
        const int prev_eos = last_eos;
        if (token_history[i] == eos_token_id) {
            last_eos = i;
        }
        if (i < start_t) continue;

        const int t = i - start_t;
        const int segment_start = prev_eos + 1;
        const int pos_in_segment = i - segment_start;

        const int64_t s0 = token_history[i];
        const int64_t s1 = (pos_in_segment >= 1 && i >= 1) ? token_history[i - 1] : eos_token_id;
        const int64_t s2 = (pos_in_segment >= 2 && i >= 2) ? token_history[i - 2] : eos_token_id;

        const uint64_t m0_u = static_cast<uint64_t>(s0) * static_cast<uint64_t>(layer_multipliers[0]);
        const uint64_t m1_u = static_cast<uint64_t>(s1) * static_cast<uint64_t>(layer_multipliers[1]);
        const uint64_t m2_u = static_cast<uint64_t>(s2) * static_cast<uint64_t>(layer_multipliers[2]);

        const int64_t m0 = static_cast<int64_t>(m0_u);
        const int64_t m1 = static_cast<int64_t>(m1_u);
        const int64_t m2 = static_cast<int64_t>(m2_u);

        const int64_t mixed2 = m0 ^ m1;
        const int64_t mixed3 = m0 ^ m1 ^ m2;

        int64_t* row = out_ids + t * NUM_HEADS;

        for (int h = 0; h < HEADS_PER_NGRAM; ++h) {
            int64_t rem = mixed2 % vocab_sizes[h];
            if (rem < 0) rem += vocab_sizes[h];
            row[h] = rem + offsets[h];
        }
        for (int h = HEADS_PER_NGRAM; h < NUM_HEADS; ++h) {
            int64_t rem = mixed3 % vocab_sizes[h];
            if (rem < 0) rem += vocab_sizes[h];
            row[h] = rem + offsets[h];
        }
    }
}

void Qwen4ExpPle::compute_ngram_ids_pure(
    const Qwen4ExpPleState& state,
    const int* input_ids,
    int num_tokens,
    int64_t* out_ids,
    std::array<int, 2>& candidate_prev_tokens) const {
    if (num_tokens <= 0) return;

    if (num_tokens >= 2) {
        candidate_prev_tokens[0] = input_ids[num_tokens - 2];
        candidate_prev_tokens[1] = input_ids[num_tokens - 1];
    } else if (num_tokens == 1) {
        candidate_prev_tokens[0] = state.prev_tokens[1];
        candidate_prev_tokens[1] = input_ids[0];
    } else {
        candidate_prev_tokens = state.prev_tokens;
    }

    constexpr int kSmallStack = 16;
    if (num_tokens + 2 <= kSmallStack) {
        std::array<int, kSmallStack> stack_history;
        stack_history[0] = state.prev_tokens[0];
        stack_history[1] = state.prev_tokens[1];
        for (int i = 0; i < num_tokens; ++i) {
            stack_history[2 + i] = input_ids[i];
        }
        compute_ngram_ids(stack_history.data(), num_tokens + 2, num_tokens,
                          layer_multipliers_.data(), ngram_heads_vocab_sizes_.data(),
                          ngram_heads_offsets_.data(), out_ids, EOS_TOKEN_ID);
    } else {
        std::vector<int> history(num_tokens + 2);
        history[0] = state.prev_tokens[0];
        history[1] = state.prev_tokens[1];
        for (int i = 0; i < num_tokens; ++i) {
            history[2 + i] = input_ids[i];
        }
        compute_ngram_ids(history.data(), static_cast<int>(history.size()), num_tokens,
                          layer_multipliers_.data(), ngram_heads_vocab_sizes_.data(),
                          ngram_heads_offsets_.data(), out_ids, EOS_TOKEN_ID);
    }
}

void Qwen4ExpPle::compute_ngram_ids(
    Qwen4ExpPleState& state,
    const int* input_ids,
    int num_tokens,
    int64_t* out_ids) const {
    std::array<int, 2> candidate_next;
    compute_ngram_ids_pure(state, input_ids, num_tokens, out_ids, candidate_next);
    state.prev_tokens = candidate_next;
}

void Qwen4ExpPle::dequant_to_staging(const int64_t* ids, int num_tokens, int slot) {
    if (num_tokens <= 0 || slot < 0 || slot >= kNumStagingSlots) return;

    bf16* staging = static_cast<bf16*>(pinned_staging_embed_slots_[slot]);

    for (int t = 0; t < num_tokens; ++t) {
        bf16* tok_embed = staging + t * EMBED_DIM;
        const int64_t* head_ids = ids + t * NUM_HEADS;

        for (int h = 0; h < NUM_HEADS; ++h) {
            int64_t id = head_ids[h];
            bf16* h_out = tok_embed + h * HEAD_DIM;

            int shard_idx = 0;
            int64_t row_in_shard = 0;
            if (rows_per_shard_ > 0 && !shards_.empty()) {
                shard_idx = static_cast<int>(id / rows_per_shard_);
                row_in_shard = id % rows_per_shard_;
                if (shard_idx >= static_cast<int>(shards_.size())) {
                    shard_idx = static_cast<int>(shards_.size()) - 1;
                }
            }

            if (shards_.empty() || !shards_[shard_idx].packed_weight) {
                std::fill(h_out, h_out + HEAD_DIM, bf16(0.0f));
                continue;
            }

            const auto& shard = shards_[shard_idx];
            const uint8_t* packed_row = shard.packed_weight + row_in_shard * 80;
            const float* absmax_row = shard.absmax + row_in_shard * 5;
            const float* qmap = shard.quant_map ? shard.quant_map : kDefaultNf4QuantMap;

            for (int b = 0; b < 80; ++b) {
                uint8_t byte = packed_row[b];
                uint8_t hi = (byte >> 4) & 0x0F;
                uint8_t lo = byte & 0x0F;

                int elem0 = 2 * b;
                int scale0 = elem0 / 32;
                h_out[elem0] = bf16(qmap[hi] * absmax_row[scale0]);

                int elem1 = 2 * b + 1;
                int scale1 = elem1 / 32;
                h_out[elem1] = bf16(qmap[lo] * absmax_row[scale1]);
            }
        }
    }
}

void Qwen4ExpPle::execute_prepare_slot(int slot, int num_tokens, const std::array<int, 2>& prev_tokens) {
    if (num_tokens <= 0 || !ctx_ || slot < 0 || slot >= kNumStagingSlots) return;

    auto& hist = slot_history_[slot];
    if (hist.size() < static_cast<size_t>(num_tokens + 2)) {
        hist.resize(num_tokens + 2);
    }
    hist[0] = prev_tokens[0];
    hist[1] = prev_tokens[1];
    std::memcpy(hist.data() + 2, slot_input_ids_[slot].data(), static_cast<size_t>(num_tokens) * sizeof(int));

    if (slot_ngram_ids_[slot].size() < static_cast<size_t>(num_tokens * NUM_HEADS)) {
        slot_ngram_ids_[slot].resize(static_cast<size_t>(num_tokens * NUM_HEADS));
    }

    compute_ngram_ids(hist.data(), num_tokens + 2, num_tokens,
                      layer_multipliers_.data(), ngram_heads_vocab_sizes_.data(),
                      ngram_heads_offsets_.data(), slot_ngram_ids_[slot].data(), EOS_TOKEN_ID);

    dequant_to_staging(slot_ngram_ids_[slot].data(), num_tokens, slot);

    const size_t bytes = static_cast<size_t>(num_tokens) * EMBED_DIM * sizeof(bf16);
    sycl::event evt = copy_q_
        ? copy_q_->memcpy(dev_embed_slots_[slot].data(), pinned_staging_embed_slots_[slot], bytes)
        : ctx_->memcpy_h2d_async(dev_embed_slots_[slot].data(), pinned_staging_embed_slots_[slot], bytes);
    slot_h2d_events_[slot] = evt;
}

void Qwen4ExpPle::worker_loop() {
    while (true) {
        WorkerTask task;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            worker_cv_.wait(lock, [this] { return stop_worker_ || !task_queue_.empty(); });
            if (stop_worker_ && task_queue_.empty()) break;
            task = task_queue_.front();
            task_queue_.pop_front();
        }

        try {
            execute_prepare_slot(task.slot, task.num_tokens, task.prev_tokens_in);
        } catch (...) {
            // Guard against unhandled exception terminating the worker thread
        }

        {
            std::lock_guard<std::mutex> lock(done_mutex_);
            completed_task_id_ = std::max(completed_task_id_, task.task_id);
            done_cv_.notify_all();
        }
    }
}

void Qwen4ExpPle::wait_task(uint64_t task_id) {
    if (task_id == 0) return;
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [this, task_id] {
        return completed_task_id_ >= task_id || stop_worker_;
    });
}

void Qwen4ExpPle::lookup_and_dequant(
    Context& ctx,
    const int64_t* ids,
    int num_tokens,
    bf16* out_embed_dev) {
    if (num_tokens <= 0) return;
    dequant_to_staging(ids, num_tokens, 0);
    ctx.memcpy_h2d_async(out_embed_dev, pinned_staging_embed_slots_[0], static_cast<size_t>(num_tokens) * EMBED_DIM * sizeof(bf16));
}

PleTicket Qwen4ExpPle::prepare(
    Context& ctx,
    const Qwen4ExpPleState& state,
    const int* input_ids,
    int num_tokens,
    const void* session_ptr,
    uint64_t generation,
    int start_pos) {
    PleTicket ticket;
    if (num_tokens <= 0) return ticket;

    int slot = (next_staging_slot_++) % kNumStagingSlots;
    ticket.staging_slot = slot;
    ticket.session_ptr = session_ptr;
    ticket.generation = generation;
    ticket.start_pos = start_pos;
    ticket.num_tokens = num_tokens;
    ticket.prev_tokens_in = state.prev_tokens;

    if (num_tokens >= 2) {
        ticket.prev_tokens_next[0] = input_ids[num_tokens - 2];
        ticket.prev_tokens_next[1] = input_ids[num_tokens - 1];
    } else if (num_tokens == 1) {
        ticket.prev_tokens_next[0] = state.prev_tokens[1];
        ticket.prev_tokens_next[1] = input_ids[0];
    } else {
        ticket.prev_tokens_next = state.prev_tokens;
    }

    // 1. If a prior task targeted this slot, wait for its CPU execution to complete
    uint64_t prior_slot_task = slot_task_ids_[slot];
    if (prior_slot_task > 0) {
        wait_task(prior_slot_task);
    }

    // 2. Retain H2D source-reader completion: ensure any previous transfer from this slot is done
    if (slot_h2d_events_[slot] != sycl::event{}) {
        try { slot_h2d_events_[slot].wait_and_throw(); } catch (...) {}
        slot_h2d_events_[slot] = sycl::event{};
    }

    if (slot_input_ids_[slot].size() < static_cast<size_t>(num_tokens)) {
        slot_input_ids_[slot].resize(static_cast<size_t>(num_tokens));
    }
    std::memcpy(slot_input_ids_[slot].data(), input_ids, static_cast<size_t>(num_tokens) * sizeof(int));

    static const bool s_ple_async = aila::env::read_flag("AILA_Q4E_PLE_ASYNC", true);
    if (s_ple_async && worker_thread_.joinable()) {
        uint64_t tid = 0;
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            tid = ++current_task_id_;
            WorkerTask task;
            task.task_id = tid;
            task.slot = slot;
            task.num_tokens = num_tokens;
            task.prev_tokens_in = ticket.prev_tokens_in;
            task_queue_.push_back(task);
            worker_cv_.notify_one();
        }
        ticket.task_id = tid;
        slot_task_ids_[slot] = tid;
    } else {
        execute_prepare_slot(slot, num_tokens, ticket.prev_tokens_in);
        ticket.h2d_event = slot_h2d_events_[slot];
        ticket.task_id = 0;
        slot_task_ids_[slot] = 0;
    }

    ticket.valid = true;
    return ticket;
}

void Qwen4ExpPle::consume(
    Context& ctx,
    PleTicket& ticket,
    Qwen4ExpPleState& state,
    const bf16* hidden_states,
    bf16* out,
    bf16* residual) {
    if (!ticket.valid || ticket.num_tokens <= 0) return;

    if (ticket.task_id > 0) {
        wait_task(ticket.task_id);
        ticket.h2d_event = slot_h2d_events_[ticket.staging_slot];
    }

    if (ticket.h2d_event != sycl::event{}) {
        try { ticket.h2d_event.wait_and_throw(); } catch (...) {}
    }

    const int num_tokens = ticket.num_tokens;
    Tensor& dev_embed = dev_embed_slots_[ticket.staging_slot];

    // 3. Projections
    if (num_tokens == 1) {
        key_proj_.ForwardDecodeRows(ctx, scratch_key_, dev_embed, key_buf_, 1);
        value_proj_.ForwardDecodeRows(ctx, scratch_val_, dev_embed, value_buf_, 1);
    } else {
        key_proj_.forward(ctx, scratch_key_, dev_embed, key_buf_, num_tokens);
        value_proj_.forward(ctx, scratch_val_, dev_embed, value_buf_, num_tokens);
    }

    // 4. Grouped RMSNorm
    ops::grouped_rms_norm(ctx, static_cast<const bf16*>(key_buf_.data()),
                          static_cast<const float*>(norm_key_w_.data()), eps_,
                          static_cast<bf16*>(key_normed_.data()),
                          num_tokens, HC_COUNT, EMBED_DIM, true);

    ops::grouped_rms_norm(ctx, hidden_states,
                          static_cast<const float*>(norm_query_w_.data()), eps_,
                          static_cast<bf16*>(query_normed_.data()),
                          num_tokens, HC_COUNT, EMBED_DIM, true);

    // 5. PLE gate-value computation
    ops::ple_gate_value(ctx, static_cast<const bf16*>(key_normed_.data()),
                        static_cast<const bf16*>(query_normed_.data()),
                        static_cast<const bf16*>(value_buf_.data()),
                        static_cast<bf16*>(gv_buf_.data()),
                        num_tokens, HC_COUNT, EMBED_DIM);

    // 6. Norm conv
    ops::grouped_rms_norm(ctx, static_cast<const bf16*>(gv_buf_.data()),
                          static_cast<const float*>(norm_conv_w_.data()), eps_,
                          static_cast<bf16*>(gv_normed_.data()),
                          num_tokens, HC_COUNT, EMBED_DIM, true);

    // 7. Dilated Conv1D + SiLU + Residual Add
    ops::ple_dilated_conv_silu_add(ctx, static_cast<const bf16*>(gv_normed_.data()),
                                  static_cast<const bf16*>(gv_buf_.data()),
                                  static_cast<float*>(state.conv_state.data()),
                                  state.conv_head,
                                  conv1d_w_ptr_,
                                  out,
                                  num_tokens, HC_HIDDEN_SIZE, CONV_KERNEL_SIZE, CONV_DILATION,
                                  residual);
}

void Qwen4ExpPle::commit(
    PleTicket& ticket,
    Qwen4ExpPleState& state) {
    if (!ticket.valid) return;
    state.prev_tokens = ticket.prev_tokens_next;
    ticket.valid = false;
}

void Qwen4ExpPle::forward(
    Context& ctx,
    Qwen4ExpPleState& state,
    const int* input_ids,
    int num_tokens,
    const bf16* hidden_states,
    bf16* out,
    bf16* residual) {
    PleTicket ticket = prepare(ctx, state, input_ids, num_tokens);
    consume(ctx, ticket, state, hidden_states, out, residual);
    commit(ticket, state);
}

void Qwen4ExpPle::init_state(Context& ctx, Qwen4ExpPleState& state) const {
    state.prev_tokens = {EOS_TOKEN_ID, EOS_TOKEN_ID};
    state.conv_head = 0;
    if (!state.conv_state.valid()) {
        state.conv_state = Tensor::allocate(ctx, {CONV_STATE_LEN, HC_HIDDEN_SIZE}, dnnl::memory::data_type::f32);
    }
    ctx.queue().memset(state.conv_state.data(), 0, state.conv_state.size_bytes()).wait_and_throw();
}

void Qwen4ExpPle::capture_snapshot(Context& ctx, const Qwen4ExpPleState& state, Qwen4ExpPleSnapshot& snapshot) const {
    if (snapshot.conv_state.size() != CONV_STATE_LEN * HC_HIDDEN_SIZE) {
        throw std::invalid_argument("Qwen4ExpPle: snapshot storage is not initialized");
    }
    snapshot.prev_tokens = state.prev_tokens;
    snapshot.conv_head = state.conv_head;
    if (state.conv_state.valid()) {
        ctx.memcpy_d2h(snapshot.conv_state.data(), state.conv_state.data(), snapshot.conv_state.size() * sizeof(float));
    } else {
        std::fill(snapshot.conv_state.begin(), snapshot.conv_state.end(), 0.0f);
    }
}

void Qwen4ExpPle::restore_snapshot(Context& ctx, Qwen4ExpPleState& state, const Qwen4ExpPleSnapshot& snapshot) const {
    state.prev_tokens = snapshot.prev_tokens;
    state.conv_head = snapshot.conv_head;
    if (!state.conv_state.valid()) {
        state.conv_state = Tensor::allocate(ctx, {CONV_STATE_LEN, HC_HIDDEN_SIZE}, dnnl::memory::data_type::f32);
    }
    if (!snapshot.conv_state.empty()) {
        ctx.memcpy_h2d(state.conv_state.data(), snapshot.conv_state.data(), snapshot.conv_state.size() * sizeof(float));
    } else {
        ctx.queue().memset(state.conv_state.data(), 0, state.conv_state.size_bytes()).wait_and_throw();
    }
}
