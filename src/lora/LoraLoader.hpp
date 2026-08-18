#pragma once

#include <string>
#include <vector>
#include "LoraConfig.hpp"
#include "../core/Context.hpp"
#include "../utils/SafeTensors.hpp"

namespace aila {
namespace lora {

struct LoraPair {
    std::string weight_name;       // "model.layers.N.self_attn.q_proj.weight"
    int r = 0;
    int in_features = 0;
    int out_features = 0;
    std::vector<float> lora_a;    // [r * in_features] F32, row-major
    std::vector<float> lora_b;    // [out_features * r] F32, row-major
};

struct LoraAdapter {
    LoraConfig config;
    std::vector<LoraPair> pairs;
};

class LoraLoader {
public:
    static bool load(const std::string& lora_dir, LoraAdapter& adapter,
                     std::string* error_message = nullptr);

    static int merge_into_weights(Context& ctx, const LoraAdapter& adapter,
                                  ModelWeights& weights, std::string* error_message = nullptr);

    static std::string peft_key_to_base_name(const std::string& lora_key);
};

} // namespace lora
} // namespace aila
