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
    // The weight key "thinker.lm_head.weight" has shape [classify_num, hidden_size]
    // instead of [vocab_size, hidden_size].
    {
        Tensor& src = weights.get("thinker.lm_head.weight");
        Tensor transposed = Tensor::allocate(ctx, {src.shape(1), src.shape(0)}, src.dtype());
        ops::transpose(ctx, src, transposed);
        ctx.synchronize();
        classify_head_.init(ctx, transposed, cfg_.hidden_size, classify_num_, true);
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
    // This executes the prefill path (embedding lookup, 28 transformer layers,
    // audio override injection, MRoPE, attention, FFN, final norm).
    // After it returns, buf_.normed contains final-normed hidden states
    // for ALL positions [seq_len, hidden_size].
    // Note: parent also runs lm_head on the last position and returns buf_.logits,
    // but we ignore that — we need classify_head on all positions.
    Qwen3ASRBackend::forward(ctx, token_ids_device, seq_len);

    // Resize all_logits_ buffer if needed
    if (!all_logits_.valid() || all_logits_.shape(0) < static_cast<int64_t>(seq_len)) {
        all_logits_ = Tensor::allocate(ctx, {static_cast<int64_t>(seq_len), classify_num_},
                                       dnnl::memory::data_type::f32);
    }

    // Run classify_head on ALL positions of the final-normed hidden states.
    // buf_.normed is [seq_len, hidden_size] (inherited protected member).
    classify_head_.forward(ctx, buf_.normed, all_logits_, seq_len);

    return all_logits_;
}
