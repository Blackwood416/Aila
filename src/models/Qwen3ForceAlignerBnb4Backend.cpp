#include "Qwen3ForceAlignerBnb4Backend.hpp"
#include "profile/Profiling.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

bool Qwen3ForceAlignerBnb4Backend::load(Context& ctx, ModelWeights& weights,
                                         const ModelSpec& spec,
                                         int max_seq_len, std::string* error_message) {
    classify_num_ = spec.classify_num;
    if (classify_num_ <= 0) {
        if (error_message) *error_message = "Qwen3ForceAlignerBnb4Backend: classify_num not set";
        return false;
    }

    // Load the full transformer backbone via parent (using NF4 quantized weights)
    if (!Qwen3ASRBnb4Backend::load(ctx, weights, spec, max_seq_len, error_message)) {
        return false;
    }

    // Replace lm_head with classify_head (dense, not quantized — same as parent's lm_head).
    // Parent's transpose_weight already transposed from [classify_num, H] to [H, classify_num].
    {
        Tensor& src = weights.get("thinker.lm_head.weight");
        classify_head_.init(ctx, src, hidden_size_, classify_num_, true);
    }

    // Re-allocate logits buffer for classify_num
    buf_.logits = Tensor::allocate(ctx, {1, classify_num_});

    AILA_LOG_INFO("[ForceAlignerBnb4] Backend loaded: classify_num=%d hidden=%d layers=%zu",
                  classify_num_, hidden_size_, layers_.size());

    return true;
}

Tensor& Qwen3ForceAlignerBnb4Backend::forward_all(Context& ctx,
                                                    const int* token_ids_device,
                                                    int seq_len) {
    if (seq_len <= 0)
        throw std::runtime_error("Qwen3ForceAlignerBnb4Backend::forward_all: seq_len must be positive");

    // Run the full transformer backbone via parent::forward()
    Qwen3ASRBnb4Backend::forward(ctx, token_ids_device, seq_len);

    // Resize all_logits_ buffer
    if (!all_logits_.valid() || all_logits_.shape(0) < static_cast<int64_t>(seq_len)) {
        all_logits_ = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), classify_num_});
    }

    // Copy exact [seq_len, H] from buf_.normed (which may have larger runtime capacity)
    Tensor normed_exact = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), hidden_size_});
    {
        bf16* src = buf_.normed.data_as<bf16>();
        bf16* dst = normed_exact.data_as<bf16>();
        ctx.queue().memcpy(dst, src, static_cast<size_t>(seq_len) * hidden_size_ * sizeof(bf16));
    }

    classify_head_.forward(ctx, normed_exact, all_logits_, seq_len);
    return all_logits_;
}
