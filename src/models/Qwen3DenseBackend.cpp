#include "Qwen3DenseBackend.hpp"

bool Qwen3DenseBackend::load(Context& ctx,
                             ModelWeights& weights,
                             const ModelSpec& spec,
                             int max_seq_len,
                             std::string* error_message) {
    if (spec.family != ModelFamily::Qwen3Dense) {
        if (error_message) {
            *error_message = "Qwen3DenseBackend: invalid model family";
        }
        return false;
    }
    cfg_ = spec.qwen3;
    model_.load(ctx, weights, cfg_, max_seq_len);
    return true;
}

