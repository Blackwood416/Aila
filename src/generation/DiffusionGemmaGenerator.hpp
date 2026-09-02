#pragma once

#include "../models/DiffusionGemmaBnb4Backend.hpp"
#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct DiffusionGemmaGeneratorConfig {
    int canvas_length = 256;
    int max_denoising_steps = 25;
    float t_min = 0.4f;
    float t_max = 0.8f;
    float entropy_bound = 0.1f;
    float confidence_threshold = 0.005f;
    int stability_threshold = 1;
    int pad_token_id = 0;
    std::vector<int> eos_token_ids{1, 106, 50};
    uint64_t seed = 42;
};

class DiffusionGemmaGenerator {
public:
    using CanvasCallback =
        std::function<void(const std::vector<int>& canvas_tokens)>;

    // Runs the full block-diffusion loop. `output_ids` receives only the
    // generated tokens (committed canvases, truncated to max_new_tokens).
    bool generate(Context& ctx, DiffusionGemmaBnb4Backend& backend,
                  const std::vector<int>& prompt_ids,
                  const DiffusionGemmaGeneratorConfig& config,
                  int max_new_tokens, std::vector<int>& output_ids,
                  std::string* error_message,
                  CanvasCallback callback = {});
};
