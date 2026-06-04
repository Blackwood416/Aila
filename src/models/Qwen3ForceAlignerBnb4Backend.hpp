#pragma once

#include "Qwen3ASRBnb4Backend.hpp"

class Qwen3ForceAlignerBnb4Backend : public Qwen3ASRBnb4Backend {
public:
    Qwen3ForceAlignerBnb4Backend() = default;

    ModelFamily family() const override { return ModelFamily::Qwen3ForceAligner; }

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;

    // All-position classify forward for forced alignment
    Tensor& forward_all(Context& ctx, const int* token_ids_device, int seq_len);

    int classify_num() const { return classify_num_; }

private:
    int classify_num_ = 0;
    Linear classify_head_;
    Tensor all_logits_;
};
