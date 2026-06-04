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
