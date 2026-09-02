#pragma once

#include "engine/Types.hpp"
#include "utils/SafeTensors.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>

class Context;

namespace aila {
namespace vision {

using bf16 = sycl::ext::oneapi::bfloat16;

struct VisionEncodeResult {
    int token_count = 0;
    int llm_grid_t = 1;
    int llm_grid_h = 0;
    int llm_grid_w = 0;
    std::vector<bf16> embeddings; // [token_count, out_hidden_size]
};

class IVisionEncoder {
public:
    virtual ~IVisionEncoder() = default;

    virtual bool load(Context& ctx,
                      ModelWeights& weights,
                      const ModelSpec& spec,
                      const std::string& model_dir,
                      std::string* error_message) = 0;

    virtual bool ready() const = 0;
    virtual int out_hidden_size() const = 0;

    virtual bool encode_image(const std::string& uri,
                              VisionEncodeResult& out,
                              std::string* error_message) = 0;

    virtual bool encode_image(const ContentPart& part,
                              VisionEncodeResult& out,
                              std::string* error_message) = 0;

    virtual bool warmup(std::string* error_message) = 0;
};

} // namespace vision
} // namespace aila
