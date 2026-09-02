#include "Gemma4VisionEncoder.hpp"

#include "core/Context.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "simdjson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <objbase.h>
#include <wincodec.h>
#endif

namespace aila {
namespace vision {

namespace {

void set_error(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

bool load_weight_ref(Context& ctx, ModelWeights& weights,
                     const std::string& name, Bnb4BitWeightRef& out,
                     int in_features, std::string* error_message) {
    if (!weights.has(name)) {
        if (error_message) {
            *error_message = "Gemma4VisionEncoder: missing weight: " + name;
        }
        return false;
    }
    if (!LoadBnb4BitWeightRef(ctx, weights, name, out, error_message, in_features)) {
        return false;
    }
    return true;
}

} // namespace

Gemma4VisionEncoder::Gemma4VisionEncoder() = default;
Gemma4VisionEncoder::~Gemma4VisionEncoder() = default;

bool Gemma4VisionEncoder::load(Context& ctx,
                               ModelWeights& weights,
                               const ModelSpec& spec,
                               const std::string& model_dir,
                               std::string* error_message) {
    ctx_ = &ctx;
    depth_ = 27;
    hidden_size_ = 1152;
    intermediate_size_ = 4304;
    out_hidden_size_ = 2816;
    num_heads_ = 16;
    head_dim_ = 72;
    patch_size_ = 16;
    pooling_kernel_size_ = 3;
    default_output_length_ = 280;
    rms_norm_eps_ = 1e-6f;

    std::string config_path = model_dir + "/config.json";
    std::string config_json = read_file_text(config_path);
    if (!config_json.empty()) {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        if (parser.parse(config_json).get(doc) == simdjson::SUCCESS) {
            simdjson::dom::element vc;
            if (doc["vision_config"].get(vc) == simdjson::SUCCESS) {
                int64_t v_depth = 0;
                if (vc["num_hidden_layers"].get(v_depth) == simdjson::SUCCESS && v_depth > 0) {
                    depth_ = static_cast<int>(v_depth);
                }
                int64_t v_hidden = 0;
                if (vc["hidden_size"].get(v_hidden) == simdjson::SUCCESS && v_hidden > 0) {
                    hidden_size_ = static_cast<int>(v_hidden);
                }
                int64_t v_inter = 0;
                if (vc["intermediate_size"].get(v_inter) == simdjson::SUCCESS && v_inter > 0) {
                    intermediate_size_ = static_cast<int>(v_inter);
                }
                int64_t v_heads = 0;
                if (vc["num_attention_heads"].get(v_heads) == simdjson::SUCCESS && v_heads > 0) {
                    num_heads_ = static_cast<int>(v_heads);
                }
                int64_t v_head_dim = 0;
                if (vc["head_dim"].get(v_head_dim) == simdjson::SUCCESS && v_head_dim > 0) {
                    head_dim_ = static_cast<int>(v_head_dim);
                }
                int64_t v_patch = 0;
                if (vc["patch_size"].get(v_patch) == simdjson::SUCCESS && v_patch > 0) {
                    patch_size_ = static_cast<int>(v_patch);
                }
                int64_t v_pool = 0;
                if (vc["pooling_kernel_size"].get(v_pool) == simdjson::SUCCESS && v_pool > 0) {
                    pooling_kernel_size_ = static_cast<int>(v_pool);
                }
                int64_t v_out_len = 0;
                if (vc["default_output_length"].get(v_out_len) == simdjson::SUCCESS && v_out_len > 0) {
                    default_output_length_ = static_cast<int>(v_out_len);
                }
            }
            simdjson::dom::element tc;
            if (doc["text_config"].get(tc) == simdjson::SUCCESS) {
                int64_t t_hidden = 0;
                if (tc["hidden_size"].get(t_hidden) == simdjson::SUCCESS && t_hidden > 0) {
                    out_hidden_size_ = static_cast<int>(t_hidden);
                }
            }
        }
    }

    const int patch_dim = 3 * patch_size_ * patch_size_; // 768

    // Patch input proj: in_features = 3 * patch_size * patch_size = 768, out_features = hidden_size (1152)
    if (!load_weight_ref(ctx, weights, "model.encoder.vision_tower.patch_embedder.input_proj.weight",
                         patch_input_ref_, patch_dim, error_message) ||
        !patch_input_proj_.init(ctx, patch_input_ref_, error_message)) {
        return false;
    }

    if (weights.has("model.encoder.vision_tower.patch_embedder.position_embedding_table")) {
        pos_embed_table_ = &weights.get("model.encoder.vision_tower.patch_embedder.position_embedding_table");
    }
    if (weights.has("model.encoder.vision_tower.std_bias")) {
        std_bias_ = &weights.get("model.encoder.vision_tower.std_bias");
    }
    if (weights.has("model.encoder.vision_tower.std_scale")) {
        std_scale_ = &weights.get("model.encoder.vision_tower.std_scale");
    }

    layers_.resize(static_cast<size_t>(depth_));
    for (int l = 0; l < depth_; ++l) {
        std::string lp = "model.encoder.vision_tower.encoder.layers." + std::to_string(l);
        ViTLayer& layer = layers_[static_cast<size_t>(l)];

        if (!weights.has(lp + ".input_layernorm.weight")) {
            set_error(error_message, "Missing " + lp + ".input_layernorm.weight");
            return false;
        }
        layer.input_ln_weight = &weights.get(lp + ".input_layernorm.weight");

        if (!load_weight_ref(ctx, weights, lp + ".self_attn.q_proj.linear.weight", layer.q_ref, hidden_size_, error_message) ||
            !layer.q_proj.init(ctx, layer.q_ref, error_message) ||
            !load_weight_ref(ctx, weights, lp + ".self_attn.k_proj.linear.weight", layer.k_ref, hidden_size_, error_message) ||
            !layer.k_proj.init(ctx, layer.k_ref, error_message) ||
            !load_weight_ref(ctx, weights, lp + ".self_attn.v_proj.linear.weight", layer.v_ref, hidden_size_, error_message) ||
            !layer.v_proj.init(ctx, layer.v_ref, error_message) ||
            !load_weight_ref(ctx, weights, lp + ".self_attn.o_proj.linear.weight", layer.o_ref, hidden_size_, error_message) ||
            !layer.o_proj.init(ctx, layer.o_ref, error_message)) {
            return false;
        }

        if (weights.has(lp + ".self_attn.q_norm.weight")) {
            layer.q_norm_weight = &weights.get(lp + ".self_attn.q_norm.weight");
        }
        if (weights.has(lp + ".self_attn.k_norm.weight")) {
            layer.k_norm_weight = &weights.get(lp + ".self_attn.k_norm.weight");
        }

        if (weights.has(lp + ".post_attention_layernorm.weight")) {
            layer.post_attn_ln_weight = &weights.get(lp + ".post_attention_layernorm.weight");
        }
        if (weights.has(lp + ".pre_feedforward_layernorm.weight")) {
            layer.pre_ffn_ln_weight = &weights.get(lp + ".pre_feedforward_layernorm.weight");
        }

        if (!load_weight_ref(ctx, weights, lp + ".mlp.gate_proj.linear.weight", layer.gate_ref, hidden_size_, error_message) ||
            !layer.gate_proj.init(ctx, layer.gate_ref, error_message) ||
            !load_weight_ref(ctx, weights, lp + ".mlp.up_proj.linear.weight", layer.up_ref, hidden_size_, error_message) ||
            !layer.up_proj.init(ctx, layer.up_ref, error_message) ||
            !load_weight_ref(ctx, weights, lp + ".mlp.down_proj.linear.weight", layer.down_ref, intermediate_size_, error_message) ||
            !layer.down_proj.init(ctx, layer.down_ref, error_message)) {
            return false;
        }

        if (weights.has(lp + ".post_feedforward_layernorm.weight")) {
            layer.post_ffn_ln_weight = &weights.get(lp + ".post_feedforward_layernorm.weight");
        }
    }

    // Embed Vision Projection: 1152 -> 2816
    if (!load_weight_ref(ctx, weights, "model.encoder.embed_vision.embedding_projection.weight",
                         embed_proj_ref_, hidden_size_, error_message) ||
        !embed_proj_.init(ctx, embed_proj_ref_, error_message)) {
        return false;
    }

    loaded_ = true;
    AILA_LOG_INFO("[Gemma4 Vision] Loaded vision tower: depth=%d hidden=%d intermediate=%d heads=%d head_dim=%d out_hidden=%d out_tokens=%d",
                  depth_, hidden_size_, intermediate_size_, num_heads_, head_dim_, out_hidden_size_, default_output_length_);
    return true;
}

namespace {

void get_aspect_ratio_preserving_size(int height, int width, int patch_size, int max_patches, int pooling_kernel_size, int& target_h, int& target_w) {
    double total_px = static_cast<double>(height) * static_cast<double>(width);
    double target_px = static_cast<double>(max_patches) * (patch_size * patch_size);
    double factor = std::sqrt(target_px / total_px);
    double ideal_height = factor * height;
    double ideal_width = factor * width;
    int side_mult = pooling_kernel_size * patch_size; // 48

    target_h = static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;
    target_w = static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;

    int max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (target_h == 0) {
        target_h = side_mult;
        target_w = std::min(static_cast<int>(std::floor(static_cast<double>(width) / height)) * side_mult, max_side_length);
    } else if (target_w == 0) {
        target_w = side_mult;
        target_h = std::min(static_cast<int>(std::floor(static_cast<double>(height) / width)) * side_mult, max_side_length);
    }
}

void apply_gemma4_vision_2d_rope(Context& ctx, Tensor& q, Tensor& k,
                                int num_patches, int grid_w,
                                int num_heads, int head_dim, float rope_theta) {
    bf16* q_ptr = static_cast<bf16*>(q.data());
    bf16* k_ptr = static_cast<bf16*>(k.data());
    const int spatial_dim = head_dim / 2; // 36
    const int num_pairs = spatial_dim / 2; // 18

    ctx.queue().parallel_for(
        sycl::range<2>(num_patches, num_heads),
        [=](sycl::item<2> it) {
            int p_idx = static_cast<int>(it.get_id(0));
            int h = static_cast<int>(it.get_id(1));
            int px = p_idx % grid_w;
            int py = p_idx / grid_w;
            size_t head_offset = (static_cast<size_t>(p_idx) * num_heads + h) * head_dim;
            bf16* q_head = q_ptr + head_offset;
            bf16* k_head = k_ptr + head_offset;

            for (int d = 0; d < num_pairs; ++d) {
                float inv_f = 1.0f / sycl::native::powr(rope_theta, (2.0f * d) / static_cast<float>(spatial_dim));
                float theta_x = static_cast<float>(px) * inv_f;
                float cos_x = sycl::cos(theta_x);
                float sin_x = sycl::sin(theta_x);

                float q0 = float(q_head[d]);
                float q1 = float(q_head[d + num_pairs]);
                q_head[d] = bf16(q0 * cos_x - q1 * sin_x);
                q_head[d + num_pairs] = bf16(q1 * cos_x + q0 * sin_x);

                float k0 = float(k_head[d]);
                float k1 = float(k_head[d + num_pairs]);
                k_head[d] = bf16(k0 * cos_x - k1 * sin_x);
                k_head[d + num_pairs] = bf16(k1 * cos_x + k0 * sin_x);

                float theta_y = static_cast<float>(py) * inv_f;
                float cos_y = sycl::cos(theta_y);
                float sin_y = sycl::sin(theta_y);

                float q2 = float(q_head[spatial_dim + d]);
                float q3 = float(q_head[spatial_dim + d + num_pairs]);
                q_head[spatial_dim + d] = bf16(q2 * cos_y - q3 * sin_y);
                q_head[spatial_dim + d + num_pairs] = bf16(q3 * cos_y + q2 * sin_y);

                float k2 = float(k_head[spatial_dim + d]);
                float k3 = float(k_head[spatial_dim + d + num_pairs]);
                k_head[spatial_dim + d] = bf16(k2 * cos_y - k3 * sin_y);
                k_head[spatial_dim + d + num_pairs] = bf16(k3 * cos_y + k2 * sin_y);
            }
        });
}

} // namespace

bool Gemma4VisionEncoder::encode_image(const ContentPart& part,
                                       VisionEncodeResult& out,
                                       std::string* error_message) {
    if (!loaded_) {
        set_error(error_message, "Gemma4VisionEncoder is not loaded");
        return false;
    }

    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
    if (!part.binary_data.empty()) {
        if (!read_image_rgb_from_memory(part.binary_data.data(), part.binary_data.size(), width, height, rgb, error_message)) {
            return false;
        }
    } else {
        if (!read_image_rgb(part.uri, width, height, rgb, error_message)) {
            return false;
        }
    }

    return encode_image(part.uri, out, error_message);
}

bool Gemma4VisionEncoder::encode_image(const std::string& uri,
                                       VisionEncodeResult& out,
                                       std::string* error_message) {
    if (!loaded_ || !ctx_) {
        set_error(error_message, "Gemma4VisionEncoder is not initialized");
        return false;
    }

    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
    if (!read_image_rgb(uri, width, height, rgb, error_message)) {
        return false;
    }

    // Aspect-ratio preserving resize according to Gemma4 patch budget (max_patches = 2520, pooling_k = 3)
    int target_w = 768;
    int target_h = 768;
    get_aspect_ratio_preserving_size(height, width, patch_size_, 2520, pooling_kernel_size_, target_h, target_w);

    std::vector<uint8_t> resized_rgb;
    if (width == target_w && height == target_h) {
        resized_rgb = std::move(rgb);
    } else {
        resize_rgb_bicubic(rgb, width, height, target_w, target_h, resized_rgb);
    }

    const int grid_w = target_w / patch_size_; // e.g. 48
    const int grid_h = target_h / patch_size_; // e.g. 48
    const int num_patches = grid_w * grid_h;   // e.g. 2304
    const int patch_dim = 3 * patch_size_ * patch_size_; // 768

    // Allocate runtime device buffers
    buf_.patch_input = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(patch_dim)}, dnnl::memory::data_type::bf16);
    buf_.hidden_a = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.hidden_b = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.normed = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.q = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.k = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.v = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.attn_out = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.gate = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(intermediate_size_)}, dnnl::memory::data_type::bf16);
    buf_.up = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(intermediate_size_)}, dnnl::memory::data_type::bf16);
    buf_.ffn_out = Tensor::allocate(*ctx_, {static_cast<size_t>(num_patches), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.scores = Tensor::allocate(*ctx_, {static_cast<size_t>(num_heads_), static_cast<size_t>(num_patches), static_cast<size_t>(num_patches)}, dnnl::memory::data_type::f32);

    // Prepare scaled patch rows on host (Gemma4 scales 2 * (x - 0.5))
    std::vector<bf16> patch_rows(static_cast<size_t>(num_patches) * static_cast<size_t>(patch_dim));
    for (int py = 0; py < grid_h; ++py) {
        for (int px = 0; px < grid_w; ++px) {
            int p_idx = py * grid_w + px;
            bf16* dst_p = patch_rows.data() + static_cast<size_t>(p_idx) * static_cast<size_t>(patch_dim);
            int idx = 0;
            for (int dy = 0; dy < patch_size_; ++dy) {
                for (int dx = 0; dx < patch_size_; ++dx) {
                    int iy = py * patch_size_ + dy;
                    int ix = px * patch_size_ + dx;
                    size_t src_idx = (static_cast<size_t>(iy) * target_w + ix) * 3;
                    for (int c = 0; c < 3; ++c) {
                        float v = static_cast<float>(resized_rgb[src_idx + c]) / 255.0f;
                        dst_p[idx++] = static_cast<bf16>(v);
                    }
                }
            }
        }
    }

    ctx_->queue().memcpy(buf_.patch_input.data(), patch_rows.data(), patch_rows.size() * sizeof(bf16)).wait();

    // 1. Patch Projection: [num_patches, patch_dim] -> [num_patches, hidden_size]
    patch_input_proj_.forward(*ctx_, buf_.linear_scratch, buf_.patch_input, buf_.hidden_a, num_patches);

    // Add 2D Positional Embeddings from position_embedding_table (shape [2, 10240, 1152])
    if (pos_embed_table_ && pos_embed_table_->data()) {
        bf16* hidden_ptr = static_cast<bf16*>(buf_.hidden_a.data());
        const bf16* table_x = static_cast<const bf16*>(pos_embed_table_->data());
        const bf16* table_y = table_x + 10240 * hidden_size_;
        const int h_dim = hidden_size_;

        ctx_->queue().parallel_for(
            sycl::range<2>(num_patches, hidden_size_),
            [=](sycl::item<2> it) {
                int p_idx = static_cast<int>(it.get_id(0));
                int d = static_cast<int>(it.get_id(1));
                int px = p_idx % grid_w;
                int py = p_idx / grid_w;
                float px_emb = float(table_x[static_cast<size_t>(px) * h_dim + d]);
                float py_emb = float(table_y[static_cast<size_t>(py) * h_dim + d]);
                size_t offset = static_cast<size_t>(p_idx) * h_dim + d;
                hidden_ptr[offset] = bf16(float(hidden_ptr[offset]) + px_emb + py_emb);
            }).wait();
    }

    // 2. Transformer forward pass (27 ViT layers)
    Tensor* cur = &buf_.hidden_a;

    for (int l = 0; l < depth_; ++l) {
        ViTLayer& layer = layers_[static_cast<size_t>(l)];

        // Pre-norm
        ops::rms_norm(*ctx_, *cur, *layer.input_ln_weight, rms_norm_eps_, buf_.normed, num_patches, hidden_size_);

        // Q, K, V Projections
        layer.q_proj.forward(*ctx_, buf_.linear_scratch, buf_.normed, buf_.q, num_patches);
        layer.k_proj.forward(*ctx_, buf_.linear_scratch, buf_.normed, buf_.k, num_patches);
        layer.v_proj.forward(*ctx_, buf_.linear_scratch, buf_.normed, buf_.v, num_patches);

        // QK head RMSNorm + V head RMSNorm (without scale weight)
        if (layer.q_norm_weight && layer.k_norm_weight) {
            ops::rms_norm(*ctx_, buf_.q, *layer.q_norm_weight, rms_norm_eps_, buf_.q, num_patches * num_heads_, head_dim_);
            ops::rms_norm(*ctx_, buf_.k, *layer.k_norm_weight, rms_norm_eps_, buf_.k, num_patches * num_heads_, head_dim_);
        }
        ops::rms_norm_unweighted(*ctx_, buf_.v, rms_norm_eps_, buf_.v, num_patches * num_heads_, head_dim_);

        // 2D Multi-dimensional RoPE on Q and K
        apply_gemma4_vision_2d_rope(*ctx_, buf_.q, buf_.k, num_patches, grid_w, num_heads_, head_dim_, 100.0f);

        // Attention: QK^T * 1.0 (Gemma4 vision scaling = 1.0) + Softmax + V accumulation
        ops::attention_bidi(*ctx_, buf_.q, buf_.k, buf_.v, buf_.attn_out, buf_.scores, num_patches, num_heads_, head_dim_, 1.0f);

        // O proj: attn_out -> ffn_out
        layer.o_proj.forward(*ctx_, buf_.linear_scratch, buf_.attn_out, buf_.ffn_out, num_patches);

        // Post-attention Layernorm
        if (layer.post_attn_ln_weight) {
            ops::rms_norm(*ctx_, buf_.ffn_out, *layer.post_attn_ln_weight, rms_norm_eps_, buf_.ffn_out, num_patches, hidden_size_);
        }

        // Residual Add: cur = cur + ffn_out
        ops::residual_add(*ctx_, *cur, buf_.ffn_out, num_patches * hidden_size_);

        // Pre-feedforward Layernorm
        if (layer.pre_ffn_ln_weight) {
            ops::rms_norm(*ctx_, *cur, *layer.pre_ffn_ln_weight, rms_norm_eps_, buf_.normed, num_patches, hidden_size_);
        } else {
            ops::rms_norm(*ctx_, *cur, *layer.input_ln_weight, rms_norm_eps_, buf_.normed, num_patches, hidden_size_);
        }

        // MLP: Gate + Up
        layer.gate_proj.forward(*ctx_, buf_.linear_scratch, buf_.normed, buf_.gate, num_patches);
        layer.up_proj.forward(*ctx_, buf_.linear_scratch, buf_.normed, buf_.up, num_patches);

        // GELU tanh activation + elementwise mul
        bf16* gate_p = static_cast<bf16*>(buf_.gate.data());
        const bf16* up_p = static_cast<const bf16*>(buf_.up.data());
        ctx_->queue().parallel_for(
            sycl::range<1>(static_cast<size_t>(num_patches) * intermediate_size_),
            [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float g = float(gate_p[i]);
                constexpr float k0 = 0.7978845608028654f;
                constexpr float k1 = 0.044715f;
                float act = 0.5f * g * (1.0f + sycl::tanh(k0 * (g + k1 * g * g * g)));
                gate_p[i] = bf16(act * float(up_p[i]));
            });

        // Down proj
        layer.down_proj.forward(*ctx_, buf_.linear_scratch, buf_.gate, buf_.ffn_out, num_patches);

        // Post-feedforward Layernorm
        if (layer.post_ffn_ln_weight) {
            ops::rms_norm(*ctx_, buf_.ffn_out, *layer.post_ffn_ln_weight, rms_norm_eps_, buf_.ffn_out, num_patches, hidden_size_);
        }

        // Residual Add: cur = cur + ffn_out
        ops::residual_add(*ctx_, *cur, buf_.ffn_out, num_patches * hidden_size_);
    }

    // 3. 2D Spatial Pooling (3x3 grid average pooling + sqrt(hidden_size) + standardization)
    const int pooled_w = grid_w / pooling_kernel_size_; // e.g. 48 / 3 = 16
    const int pooled_h = grid_h / pooling_kernel_size_; // e.g. 48 / 3 = 16
    const int out_tokens = pooled_w * pooled_h;         // e.g. 256 soft tokens

    buf_.pooled = Tensor::allocate(*ctx_, {static_cast<size_t>(out_tokens), static_cast<size_t>(hidden_size_)}, dnnl::memory::data_type::bf16);
    buf_.embed_out = Tensor::allocate(*ctx_, {static_cast<size_t>(out_tokens), static_cast<size_t>(out_hidden_size_)}, dnnl::memory::data_type::bf16);

    bf16* pooled_p = static_cast<bf16*>(buf_.pooled.data());
    const bf16* src_p = static_cast<const bf16*>(cur->data());
    const int h_dim = hidden_size_;
    const float root_h = std::sqrt(static_cast<float>(hidden_size_)); // sqrt(1152) = 33.9411255f
    const bf16* bias_p = std_bias_ ? static_cast<const bf16*>(std_bias_->data()) : nullptr;
    const bf16* scale_p = std_scale_ ? static_cast<const bf16*>(std_scale_->data()) : nullptr;

    ctx_->queue().parallel_for(
        sycl::range<2>(out_tokens, hidden_size_),
        [=](sycl::item<2> it) {
            int out_idx = static_cast<int>(it.get_id(0));
            int d = static_cast<int>(it.get_id(1));
            int kx = out_idx % pooled_w;
            int ky = out_idx / pooled_w;

            float sum = 0.0f;
            for (int dy = 0; dy < 3; ++dy) {
                for (int dx = 0; dx < 3; ++dx) {
                    int py = ky * 3 + dy;
                    int px = kx * 3 + dx;
                    int p_idx = py * grid_w + px;
                    sum += float(src_p[static_cast<size_t>(p_idx) * h_dim + d]);
                }
            }
            float feat = (sum / 9.0f) * root_h;
            if (bias_p && scale_p) {
                feat = (feat - float(bias_p[d])) * float(scale_p[d]);
            }
            pooled_p[static_cast<size_t>(out_idx) * h_dim + d] = bf16(feat);
        }).wait();

    // Pre-projection RMSNorm (without weight)
    ops::rms_norm_unweighted(*ctx_, buf_.pooled, rms_norm_eps_, buf_.pooled, out_tokens, hidden_size_);

    // 4. Final Projection: [out_tokens, 1152] -> [out_tokens, 2816]
    embed_proj_.forward(*ctx_, buf_.linear_scratch, buf_.pooled, buf_.embed_out, out_tokens);
    ctx_->synchronize();

    // 5. Copy embeddings to host
    out.token_count = out_tokens;
    out.llm_grid_t = 1;
    out.llm_grid_h = pooled_h;
    out.llm_grid_w = pooled_w;
    out.embeddings.resize(static_cast<size_t>(out_tokens) * static_cast<size_t>(out_hidden_size_));

    ctx_->queue().memcpy(out.embeddings.data(), buf_.embed_out.data(), out.embeddings.size() * sizeof(bf16)).wait();
    buf_ = RuntimeBuffers{};
    ctx_->synchronize();
    return true;
}

bool Gemma4VisionEncoder::warmup(std::string* error_message) {
    if (!loaded_) {
        set_error(error_message, "Gemma4VisionEncoder is not loaded");
        return false;
    }
    return true;
}

#ifdef _WIN32
std::string percent_decode(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        char c = in[i];
        if (c == '%' && i + 2 < in.size()) {
            auto hex = [](char x) -> int {
                if (x >= '0' && x <= '9') return x - '0';
                if (x >= 'a' && x <= 'f') return x - 'a' + 10;
                if (x >= 'A' && x <= 'F') return x - 'A' + 10;
                return -1;
            };
            int hi = hex(in[i + 1]);
            int lo = hex(in[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(c);
    }
    return out;
}

std::string uri_to_local_path(const std::string& uri) {
    std::string p = uri;
    const std::string file_prefix1 = "file:///";
    const std::string file_prefix2 = "file://";
    if (p.rfind(file_prefix1, 0) == 0) {
        p = p.substr(file_prefix1.size());
        if (!(p.size() >= 2 && p[1] == ':')) {
            p = "/" + p;
        }
    } else if (p.rfind(file_prefix2, 0) == 0) {
        p = p.substr(file_prefix2.size());
    }
    p = percent_decode(p);
    return p;
}

std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, out.data(), len);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}
#endif

bool Gemma4VisionEncoder::read_image_rgb(const std::string& uri,
                                         int& width,
                                         int& height,
                                         std::vector<uint8_t>& rgb,
                                         std::string* error_message) {
#ifdef _WIN32
    std::string local_path = uri_to_local_path(uri);
    std::filesystem::path fp(local_path);
    if (!std::filesystem::exists(fp)) {
        set_error(error_message, "Image file does not exist: " + fp.string());
        return false;
    }
    std::wstring wpath = utf8_to_wide(fp.string());

    HRESULT hr_ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool need_uninit = SUCCEEDED(hr_ci);
    if (hr_ci == RPC_E_CHANGED_MODE) {
        need_uninit = false;
    } else if (FAILED(hr_ci)) {
        set_error(error_message, "COM init failed for image decode");
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    bool ok = false;
    std::string err = "Failed to decode image";

    do {
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&factory));
        if (FAILED(hr) || !factory) {
            err = "WIC factory create failed";
            break;
        }

        hr = factory->CreateDecoderFromFilename(
            wpath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
        if (FAILED(hr) || !decoder) {
            err = "WIC decoder create failed";
            break;
        }

        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr) || !frame) {
            err = "WIC frame decode failed";
            break;
        }

        hr = factory->CreateFormatConverter(&converter);
        if (FAILED(hr) || !converter) {
            err = "WIC format converter create failed";
            break;
        }

        hr = converter->Initialize(
            frame, GUID_WICPixelFormat24bppBGR,
            WICBitmapDitherTypeNone, nullptr, 0.0,
            WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            err = "WIC format convert init failed";
            break;
        }

        UINT w = 0;
        UINT h = 0;
        hr = converter->GetSize(&w, &h);
        if (FAILED(hr) || w == 0 || h == 0) {
            err = "WIC invalid image size";
            break;
        }

        width = static_cast<int>(w);
        height = static_cast<int>(h);
        UINT stride = w * 3;
        UINT bytes = stride * h;
        std::vector<uint8_t> bgr(bytes);
        hr = converter->CopyPixels(nullptr, stride, bytes, bgr.data());
        if (FAILED(hr)) {
            err = "WIC CopyPixels failed";
            break;
        }

        rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
        for (int y = 0; y < height; ++y) {
            const uint8_t* src_row = bgr.data() + static_cast<size_t>(y) * static_cast<size_t>(stride);
            uint8_t* dst_row = rgb.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 3u;
            for (int x = 0; x < width; ++x) {
                dst_row[3 * x + 0] = src_row[3 * x + 2];
                dst_row[3 * x + 1] = src_row[3 * x + 1];
                dst_row[3 * x + 2] = src_row[3 * x + 0];
            }
        }

        ok = true;
    } while (false);

    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (need_uninit) CoUninitialize();

    if (!ok) {
        set_error(error_message, err + ": " + fp.string());
        return false;
    }
    return true;
#else
    set_error(error_message, "WIC image reading not supported on non-Windows platform");
    return false;
#endif
}

bool Gemma4VisionEncoder::read_image_rgb_from_memory(const uint8_t* data,
                                                     size_t size,
                                                     int& width,
                                                     int& height,
                                                     std::vector<uint8_t>& rgb,
                                                     std::string* error_message) {
#ifdef _WIN32
    if (!data || size == 0) {
        set_error(error_message, "Empty image memory buffer");
        return false;
    }

    HRESULT hr_ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    bool need_uninit = SUCCEEDED(hr_ci);
    if (hr_ci == RPC_E_CHANGED_MODE) {
        need_uninit = false;
    } else if (FAILED(hr_ci)) {
        set_error(error_message, "COM init failed for memory image decode");
        return false;
    }

    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    bool ok = false;
    std::string err = "Failed to decode memory image";

    do {
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                      IID_PPV_ARGS(&factory));
        if (FAILED(hr) || !factory) {
            err = "WIC factory create failed";
            break;
        }

        hr = factory->CreateStream(&stream);
        if (FAILED(hr) || !stream) {
            err = "WIC create stream failed";
            break;
        }

        hr = stream->InitializeFromMemory(const_cast<BYTE*>(data), static_cast<DWORD>(size));
        if (FAILED(hr)) {
            err = "WIC init stream failed";
            break;
        }

        hr = factory->CreateDecoderFromStream(
            stream, nullptr, WICDecodeMetadataCacheOnLoad, &decoder);
        if (FAILED(hr) || !decoder) {
            err = "WIC create decoder failed";
            break;
        }

        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr) || !frame) {
            err = "WIC get frame failed";
            break;
        }

        hr = factory->CreateFormatConverter(&converter);
        if (FAILED(hr) || !converter) {
            err = "WIC create converter failed";
            break;
        }

        hr = converter->Initialize(
            frame, GUID_WICPixelFormat24bppBGR,
            WICBitmapDitherTypeNone, nullptr, 0.0,
            WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            err = "WIC init format converter failed";
            break;
        }

        UINT w = 0;
        UINT h = 0;
        hr = converter->GetSize(&w, &h);
        if (FAILED(hr) || w == 0 || h == 0) {
            err = "WIC invalid dimensions";
            break;
        }

        width = static_cast<int>(w);
        height = static_cast<int>(h);
        UINT stride = w * 3;
        UINT bytes = stride * h;
        std::vector<uint8_t> bgr(bytes);
        hr = converter->CopyPixels(nullptr, stride, bytes, bgr.data());
        if (FAILED(hr)) {
            err = "WIC copy pixels failed";
            break;
        }

        rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3u);
        for (int y = 0; y < height; ++y) {
            const uint8_t* src_row = bgr.data() + static_cast<size_t>(y) * static_cast<size_t>(stride);
            uint8_t* dst_row = rgb.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 3u;
            for (int x = 0; x < width; ++x) {
                dst_row[3 * x + 0] = src_row[3 * x + 2];
                dst_row[3 * x + 1] = src_row[3 * x + 1];
                dst_row[3 * x + 2] = src_row[3 * x + 0];
            }
        }

        ok = true;
    } while (false);

    if (converter) converter->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    if (need_uninit) CoUninitialize();

    if (!ok) {
        set_error(error_message, err);
        return false;
    }
    return true;
#else
    set_error(error_message, "WIC image reading not supported on non-Windows platform");
    return false;
#endif
}

void Gemma4VisionEncoder::resize_rgb_bicubic(const std::vector<uint8_t>& src,
                                             int src_w,
                                             int src_h,
                                             int dst_w,
                                             int dst_h,
                                             std::vector<uint8_t>& dst) {
    dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y) {
        float src_yf = (y + 0.5f) * scale_y - 0.5f;
        int y0 = std::max(0, std::min(src_h - 1, static_cast<int>(std::floor(src_yf))));
        int y1 = std::max(0, std::min(src_h - 1, y0 + 1));
        float wy = src_yf - y0;

        for (int x = 0; x < dst_w; ++x) {
            float src_xf = (x + 0.5f) * scale_x - 0.5f;
            int x0 = std::max(0, std::min(src_w - 1, static_cast<int>(std::floor(src_xf))));
            int x1 = std::max(0, std::min(src_w - 1, x0 + 1));
            float wx = src_xf - x0;

            for (int c = 0; c < 3; ++c) {
                float v00 = src[(static_cast<size_t>(y0) * src_w + x0) * 3 + c];
                float v01 = src[(static_cast<size_t>(y0) * src_w + x1) * 3 + c];
                float v10 = src[(static_cast<size_t>(y1) * src_w + x0) * 3 + c];
                float v11 = src[(static_cast<size_t>(y1) * src_w + x1) * 3 + c];

                float v0 = v00 * (1.0f - wx) + v01 * wx;
                float v1 = v10 * (1.0f - wx) + v11 * wx;
                float v = v0 * (1.0f - wy) + v1 * wy;

                dst[(static_cast<size_t>(y) * dst_w + x) * 3 + c] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, v)));
            }
        }
    }
}

} // namespace vision
} // namespace aila
