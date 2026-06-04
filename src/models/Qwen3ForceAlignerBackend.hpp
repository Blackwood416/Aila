#pragma once

#include "Qwen3ASRBackend.hpp"
#include "../ops/Ops.hpp"

class Qwen3ForceAlignerBackend : public Qwen3ASRBackend {
public:
    Qwen3ForceAlignerBackend() = default;

    ModelFamily family() const override { return ModelFamily::Qwen3ForceAligner; }

    bool load(Context& ctx, ModelWeights& weights, const ModelSpec& spec,
              int max_seq_len, std::string* error_message) override;

    // Single prefill forward returning all-position classify logits [seq_len, classify_num].
    // Unlike the parent forward() which only returns logits for the last position,
    // this processes ALL positions through the classify head.
    Tensor& forward_all(Context& ctx, const int* token_ids_device, int seq_len);

    int classify_num() const { return classify_num_; }

private:
    int classify_num_ = 0;
    Linear classify_head_;
    Tensor all_logits_;  // buffer for all-position logits [seq_len, classify_num]
};
