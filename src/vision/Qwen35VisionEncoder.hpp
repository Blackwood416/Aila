#pragma once

#include "engine/Types.hpp"
#include "utils/SafeTensors.hpp"
#include "../ops/Ops.hpp"
#include <string>
#include <vector>
#include <deque>
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
    ~Qwen35VisionEncoder();

    bool load(Context& ctx,
              ModelWeights& weights,
              const ModelSpec& spec,
              const std::string& model_dir,
              std::string* error_message);

    bool ready() const { return loaded_; }
    int out_hidden_size() const { return out_hidden_size_; }

    bool encode_image(const std::string& uri,
                      VisionEncodeResult& out,
                      std::string* error_message);
    bool warmup(std::string* error_message);

private:
    struct BlockWeights {
        Tensor* ln1_weight = nullptr;
        Tensor* ln1_bias = nullptr;
        Linear qkv;
        Tensor* qkv_bias = nullptr;
        Linear proj;
        Tensor* proj_bias = nullptr;
        Tensor* ln2_weight = nullptr;
        Tensor* ln2_bias = nullptr;
        Linear fc1;
        Tensor* fc1_bias = nullptr;
        Linear fc2;
        Tensor* fc2_bias = nullptr;
    };

    struct RuntimeBuffers {
        Tensor image_rgb;      // u8 [bytes]
        Tensor patch_rows;     // bf16 [patch_cap, patch_dim]
        Tensor hidden_a;       // bf16 [patch_cap, hidden]
        Tensor hidden_b;       // bf16 [patch_cap, hidden]
        Tensor normed;         // bf16 [patch_cap, hidden]
        Tensor qkv;            // bf16 [patch_cap, 3 * hidden]
        Tensor q;              // bf16 [patch_cap, hidden]
        Tensor k;              // bf16 [patch_cap, hidden]
        Tensor v;              // bf16 [patch_cap, hidden]
        Tensor attn_out;       // bf16 [patch_cap, hidden]
        Tensor ffn;            // bf16 [patch_cap, intermediate]
        Tensor merger_normed;  // bf16 [out_cap, merger_in_dim]
        Tensor merger_hidden;  // bf16 [out_cap, merger_fc1_out]
        Tensor merger_out;     // bf16 [out_cap, out_hidden]
        Tensor scores;         // f32 [num_heads, patch_cap, patch_cap]
    };

    bool loaded_ = false;
    Context* ctx_ = nullptr;

    int depth_ = 0;
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
    int out_hidden_size_ = 0;
    int num_heads_ = 0;
    int head_dim_ = 0;
    int patch_size_ = 16;
    int merge_size_ = 2;

    int min_tokens_ = 1;
    int max_tokens_ = 1024;
    int min_pixels_ = 256 * 256;
    int max_pixels_ = 1024 * 1024;

    float image_mean_[3] = {0.5f, 0.5f, 0.5f};
    float image_std_[3] = {0.5f, 0.5f, 0.5f};

    Tensor* patch_weight_ = nullptr;
    Tensor* patch_bias_ = nullptr;
    Linear patch_proj_;
    Tensor* pos_embed_ = nullptr;
    int pos_embed_len_ = 0;

    std::vector<BlockWeights> blocks_;

    Tensor* merger_norm_weight_ = nullptr;
    Tensor* merger_norm_bias_ = nullptr;
    Linear merger_fc1_;
    Tensor* merger_fc1_bias_ = nullptr;
    int merger_fc1_out_ = 0;
    Linear merger_fc2_;
    Tensor* merger_fc2_bias_ = nullptr;

    RuntimeBuffers buf_;
    Tensor empty_tensor_;
    std::deque<Tensor> owned_weights_;

    int image_bytes_capacity_ = 0;
    int patch_capacity_ = 0;
    int out_token_capacity_ = 0;

    std::vector<int> pos_y_host_;
    std::vector<int> pos_x_host_;
    int* pos_y_device_ = nullptr;
    int* pos_x_device_ = nullptr;
    int pos_capacity_ = 0;

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

    Tensor* get_tensor(ModelWeights& weights,
                       const std::string& name,
                       std::string* error_message,
                       bool required = true);
    Tensor* ensure_bf16_weight(Context& ctx,
                               ModelWeights& weights,
                               const std::string& name,
                               std::string* error_message,
                               bool required = true);
    bool prepare_patch_weight(Context& ctx,
                              ModelWeights& weights,
                              std::string* error_message);

    void ensure_runtime_buffers(int image_bytes,
                                int num_patches,
                                int out_tokens,
                                int merger_in_dim);
    void ensure_position_buffers(int num_patches);
    void release_runtime_buffers();
    void fill_merge_block_positions(int grid_w, int grid_h);

    bool encode_rgb_image(const std::string& uri,
                          int src_w,
                          int src_h,
                          const std::vector<uint8_t>& src_rgb,
                          double decode_ms,
                          bool profile,
                          VisionEncodeResult& out,
                          std::string* error_message);
    bool read_preprocessor(const std::string& model_dir, std::string* error_message);
};

} // namespace vision
} // namespace aila
