#pragma once

#include "IModelBackend.hpp"
#include "Qwen3.hpp"

class Qwen3DenseBackend : public IModelBackend {
public:
    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              int max_seq_len,
              std::string* error_message) override;

    Tensor& forward(Context& ctx, const int* token_ids_device, int seq_len) override {
        return model_.forward(ctx, token_ids_device, seq_len);
    }

    void reset() override { model_.reset(); }
    void truncate_kv_cache(int new_len) override { model_.truncate_kv_cache(new_len); }
    int max_seq_len() const override { return model_.max_seq_len(); }
    int vocab_size() const override { return cfg_.vocab_size; }
    ModelFamily family() const override { return ModelFamily::Qwen3Dense; }

private:
    Qwen3Config cfg_{};
    Qwen3Model model_{};
};

