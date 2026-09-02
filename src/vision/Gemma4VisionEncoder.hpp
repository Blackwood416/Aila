#pragma once

#include "IVisionEncoder.hpp"
#include "engine/Types.hpp"
#include "utils/SafeTensors.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../core/Tensor.hpp"
#include <string>
#include <vector>
#include <deque>
#include <cstdint>
#include <sycl/sycl.hpp>

class Context;

namespace aila {
namespace vision {

class Gemma4VisionEncoder : public IVisionEncoder {
public:
    Gemma4VisionEncoder();
    ~Gemma4VisionEncoder() override;

    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              const std::string& model_dir,
              std::string* error_message) override;

    bool ready() const override { return loaded_; }
    int out_hidden_size() const override { return out_hidden_size_; }

    bool encode_image(const std::string& uri,
                      VisionEncodeResult& out,
                      std::string* error_message) override;
    bool encode_image(const ContentPart& part,
                      VisionEncodeResult& out,
                      std::string* error_message) override;
    bool warmup(std::string* error_message) override;

private:
    struct ViTLayer {
        Tensor* input_ln_weight = nullptr;
        Bnb4BitWeightRef q_ref;
        Bnb4BitWeightRef k_ref;
        Bnb4BitWeightRef v_ref;
        Bnb4BitWeightRef o_ref;
        Bnb4BitLinear q_proj;
        Bnb4BitLinear k_proj;
        Bnb4BitLinear v_proj;
        Bnb4BitLinear o_proj;
        Tensor* q_norm_weight = nullptr;
        Tensor* k_norm_weight = nullptr;
        Tensor* post_attn_ln_weight = nullptr;
        Tensor* pre_ffn_ln_weight = nullptr;
        Bnb4BitWeightRef gate_ref;
        Bnb4BitWeightRef up_ref;
        Bnb4BitWeightRef down_ref;
        Bnb4BitLinear gate_proj;
        Bnb4BitLinear up_proj;
        Bnb4BitLinear down_proj;
        Tensor* post_ffn_ln_weight = nullptr;
    };

    struct RuntimeBuffers {
        Tensor patch_input;       // bf16 [num_patches, patch_dim]
        Tensor hidden_a;          // bf16 [num_patches, hidden_size]
        Tensor hidden_b;          // bf16 [num_patches, hidden_size]
        Tensor normed;            // bf16 [num_patches, hidden_size]
        Tensor q;                 // bf16 [num_patches, hidden_size]
        Tensor k;                 // bf16 [num_patches, hidden_size]
        Tensor v;                 // bf16 [num_patches, hidden_size]
        Tensor attn_out;          // bf16 [num_patches, hidden_size]
        Tensor gate;              // bf16 [num_patches, intermediate_size]
        Tensor up;                // bf16 [num_patches, intermediate_size]
        Tensor ffn_out;           // bf16 [num_patches, hidden_size]
        Tensor scores;            // f32 [num_heads, num_patches, num_patches]
        Tensor pooled;            // bf16 [280, hidden_size]
        Tensor embed_out;         // bf16 [280, out_hidden_size]
        Bnb4BitLinearScratch linear_scratch;
    };

    bool loaded_ = false;
    Context* ctx_ = nullptr;

    int depth_ = 27;
    int hidden_size_ = 1152;
    int intermediate_size_ = 4304;
    int out_hidden_size_ = 2816;
    int num_heads_ = 16;
    int head_dim_ = 72;
    int patch_size_ = 16;
    int pooling_kernel_size_ = 3;
    int default_output_length_ = 280;
    float rms_norm_eps_ = 1e-6f;

    float image_mean_[3] = {0.5f, 0.5f, 0.5f};
    float image_std_[3] = {0.5f, 0.5f, 0.5f};

    Bnb4BitWeightRef patch_input_ref_;
    Bnb4BitLinear patch_input_proj_;
    Tensor* pos_embed_table_ = nullptr;
    Tensor* std_bias_ = nullptr;
    Tensor* std_scale_ = nullptr;
    int pos_embed_len_ = 0;

    std::vector<ViTLayer> layers_;

    Bnb4BitWeightRef embed_proj_ref_;
    Bnb4BitLinear embed_proj_;

    RuntimeBuffers buf_;
    Tensor empty_tensor_;
    std::deque<Tensor> owned_weights_;

    static bool read_image_rgb(const std::string& uri,
                               int& width,
                               int& height,
                               std::vector<uint8_t>& rgb,
                               std::string* error_message);

    static bool read_image_rgb_from_memory(const uint8_t* data,
                                           size_t size,
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
};

} // namespace vision
} // namespace aila
