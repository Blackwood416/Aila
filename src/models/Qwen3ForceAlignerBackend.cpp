#include "Qwen3ForceAlignerBackend.hpp"
#include "profile/Profiling.hpp"

using bf16 = sycl::ext::oneapi::bfloat16;

bool Qwen3ForceAlignerBackend::load(Context& ctx, ModelWeights& weights,
                                     const ModelSpec& spec,
                                     int max_seq_len, std::string* error_message) {
    classify_num_ = spec.classify_num;
    if (classify_num_ <= 0) {
        if (error_message) *error_message = "Qwen3ForceAlignerBackend: classify_num not set";
        return false;
    }

    // Load the full transformer backbone via parent.
    // Parent sets up all 28 layers, KV cache, embeddings, lm_head.
    if (!Qwen3ASRBackend::load(ctx, weights, spec, max_seq_len, error_message)) {
        return false;
    }

    // Replace lm_head with classify_head.
    // The parent's transpose_weight() already transposed "thinker.lm_head.weight"
    // from [classify_num, hidden_size] to [hidden_size, classify_num].
    // We just need to re-initialize the Linear with the correct output dimension.
    {
        Tensor& src = weights.get("thinker.lm_head.weight");  // already [H, classify_num]
        classify_head_.init(ctx, src, cfg_.hidden_size, classify_num_, true);
    }

    // Re-allocate logits buffer: parent allocated [1, vocab_size], we need [1, classify_num].
    buf_.logits = Tensor::allocate(ctx, {1, classify_num_});

    AILA_LOG_INFO("[ForceAligner] Backend loaded: classify_num=%d hidden=%d layers=%d",
                  classify_num_, cfg_.hidden_size, cfg_.num_hidden_layers);

    return true;
}

Tensor& Qwen3ForceAlignerBackend::forward_all(Context& ctx,
                                               const int* token_ids_device,
                                               int seq_len) {
    if (seq_len <= 0)
        throw std::runtime_error("Qwen3ForceAlignerBackend::forward_all: seq_len must be positive");

    // Run the full transformer backbone via parent::forward().
    Qwen3ASRBackend::forward(ctx, token_ids_device, seq_len);

    // Resize all_logits_ buffer if needed
    if (!all_logits_.valid() || all_logits_.shape(0) < static_cast<int64_t>(seq_len)) {
        all_logits_ = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), classify_num_});
    }

    // Use exact-sized input buffer: buf_.normed may have larger runtime capacity
    // which oneDNN can't slice implicitly.
    int H = cfg_.hidden_size;
    Tensor normed_exact = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), H});
    {
        bf16* src = buf_.normed.data_as<bf16>();
        bf16* dst = normed_exact.data_as<bf16>();
        ctx.queue().memcpy(dst, src, static_cast<size_t>(seq_len) * H * sizeof(bf16));
    }

    classify_head_.forward(ctx, normed_exact, all_logits_, seq_len);
    return all_logits_;
}
