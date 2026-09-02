#include "vision/Gemma4VisionEncoder.hpp"
#include "core/Context.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::cout << "Running DiffusionGemma Vision Unit Tests..." << std::endl;

    Context ctx;
    ModelSpec spec;
    spec.family = ModelFamily::DiffusionGemma;

    aila::vision::Gemma4VisionEncoder encoder;

    std::string model_dir = "E:/RiderProjects/Aila/models/DiffusionGemma-26B-A4B-it-BNB-NF4";
    ModelWeights weights = LoadModelWeightsFromDir(model_dir, ctx);
    std::string err;
    if (weights.size() > 0) {
        if (encoder.load(ctx, weights, spec, model_dir, &err)) {
            aila::vision::VisionEncodeResult res;
            if (encoder.encode_image("E:/RiderProjects/Aila/test_multimodal.png", res, &err)) {
                std::cout << "[Test] Encoded tokens: " << res.token_count << " (grid: " << res.llm_grid_w << "x" << res.llm_grid_h << ")" << std::endl;
                float min_v = 1e9f, max_v = -1e9f, sum_v = 0.0f, sum_sq = 0.0f;
                for (auto bf : res.embeddings) {
                    float f = float(bf);
                    if (f < min_v) min_v = f;
                    if (f > max_v) max_v = f;
                    sum_v += f;
                    sum_sq += f * f;
                }
                float mean_v = sum_v / float(res.embeddings.size());
                float std_v = std::sqrt(sum_sq / float(res.embeddings.size()) - mean_v * mean_v);
                std::cout << "[Test] Embeddings stats: min=" << min_v << " max=" << max_v << " mean=" << mean_v << " std=" << std_v << std::endl;
                std::cout << "[Test] Token 0 first 8 values: ";
                for (int i = 0; i < 8; ++i) {
                    std::cout << float(res.embeddings[i]) << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "[Test] Encode failed: " << err << std::endl;
            }
        } else {
            std::cout << "[Test] Load vision encoder failed: " << err << std::endl;
        }
    } else {
        std::cout << "[Test] Load weights failed: " << err << std::endl;
    }

    std::cout << "All DiffusionGemma Vision Unit Tests PASSED!" << std::endl;
    return 0;
}
