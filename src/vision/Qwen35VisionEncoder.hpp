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

class Qwen35VisionEncoder {
public:
    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              const std::string& model_dir,
              std::string* error_message);

    bool ready() const { return loaded_; }
    int out_hidden_size() const { return out_hidden_size_; }

    bool encode_image(const std::string& uri,
                      VisionEncodeResult& out,
                      std::string* error_message) const;

private:
    struct BlockWeights {
        std::vector<float> ln1_w;
        std::vector<float> ln1_b;
        std::vector<float> qkv_w;
        std::vector<float> qkv_b;
        std::vector<float> proj_w;
        std::vector<float> proj_b;
        std::vector<float> ln2_w;
        std::vector<float> ln2_b;
        std::vector<float> fc1_w;
        std::vector<float> fc1_b;
        std::vector<float> fc2_w;
        std::vector<float> fc2_b;
    };

    bool loaded_ = false;

    int depth_ = 0;
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int out_hidden_size_ = 0;
    int num_heads_ = 0;
    int head_dim_ = 0;
    int patch_size_ = 16;
    int merge_size_ = 2;

    int min_tokens_ = 8;
    int max_tokens_ = 64;
    int min_pixels_ = 256 * 256;
    int max_pixels_ = 1024 * 1024;

    float image_mean_[3] = {0.5f, 0.5f, 0.5f};
    float image_std_[3] = {0.5f, 0.5f, 0.5f};

    std::vector<float> patch_kernel_fused_; // [hidden, 3, patch, patch]
    std::vector<float> patch_bias_;          // [hidden]
    std::vector<float> pos_embed_;           // [pos_len, hidden]

    std::vector<BlockWeights> blocks_;

    std::vector<float> merger_norm_w_;       // optional [hidden]
    std::vector<float> merger_norm_b_;       // optional [hidden]
    std::vector<float> merger_fc1_w_;        // [3072, 3072]
    std::vector<float> merger_fc1_b_;        // [3072]
    std::vector<float> merger_fc2_w_;        // [out_hidden, 3072]
    std::vector<float> merger_fc2_b_;        // [out_hidden]

    static bool read_image_rgb(const std::string& uri,
                               int& width,
                               int& height,
                               std::vector<uint8_t>& rgb,
                               std::string* error_message);

    static void resize_rgb_bicubic(const std::vector<uint8_t>& src,
                                   int src_w,
                                   int src_h,
                                   int dst_w,
                                   int dst_h,
                                   std::vector<uint8_t>& dst);

    static void choose_target_size(int src_w,
                                   int src_h,
                                   int align,
                                   int min_pixels,
                                   int max_pixels,
                                   int& out_w,
                                   int& out_h);

    static void layer_norm_affine(const std::vector<float>& in,
                                  int rows,
                                  int cols,
                                  const std::vector<float>& gamma,
                                  const std::vector<float>& beta,
                                  float eps,
                                  std::vector<float>& out);

    static void linear_rowmajor(const std::vector<float>& in,
                                int rows,
                                int in_dim,
                                const std::vector<float>& weight,
                                const std::vector<float>& bias,
                                int out_dim,
                                std::vector<float>& out);

    static void gelu_tanh_inplace(std::vector<float>& x);

    static void softmax_inplace(std::vector<float>& x);

    static bool read_tensor_as_float(Context& ctx,
                                     ModelWeights& weights,
                                     const std::string& name,
                                     std::vector<float>& out,
                                     std::string* error_message,
                                     bool required = true);

    bool read_preprocessor(const std::string& model_dir, std::string* error_message);
};

} // namespace vision
} // namespace aila
