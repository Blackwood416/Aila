#include "Qwen35VisionEncoder.hpp"

#include "core/Context.hpp"
#include "utils/EnvUtils.hpp"
#include "simdjson.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
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

float gelu_tanh_scalar(float x) {
    constexpr float kAlpha = 0.7978845608028654f;
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

void reorder_qwen_block_sequence(const std::vector<float>& src,
                                 int grid_w,
                                 int grid_h,
                                 int hidden,
                                 int merge_size,
                                 std::vector<float>& dst,
                                 std::vector<int>* pos_y = nullptr,
                                 std::vector<int>* pos_x = nullptr) {
    const int num_tokens = grid_w * grid_h;
    dst.resize(static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden));
    if (pos_y) pos_y->resize(static_cast<size_t>(num_tokens));
    if (pos_x) pos_x->resize(static_cast<size_t>(num_tokens));

    int out_idx = 0;
    for (int by = 0; by < grid_h; by += merge_size) {
        for (int bx = 0; bx < grid_w; bx += merge_size) {
            for (int dy = 0; dy < merge_size; ++dy) {
                for (int dx = 0; dx < merge_size; ++dx) {
                    const int sy = by + dy;
                    const int sx = bx + dx;
                    const int src_idx = sy * grid_w + sx;
                    std::memcpy(
                        dst.data() + static_cast<size_t>(out_idx) * static_cast<size_t>(hidden),
                        src.data() + static_cast<size_t>(src_idx) * static_cast<size_t>(hidden),
                        static_cast<size_t>(hidden) * sizeof(float));
                    if (pos_y) (*pos_y)[static_cast<size_t>(out_idx)] = sy;
                    if (pos_x) (*pos_x)[static_cast<size_t>(out_idx)] = sx;
                    ++out_idx;
                }
            }
        }
    }
}

void apply_qwen_vision_mrope_inplace(std::vector<float>& x,
                                     int num_tokens,
                                     int num_heads,
                                     int head_dim,
                                     const std::vector<int>& pos_y,
                                     const std::vector<int>& pos_x) {
    if (num_tokens <= 0 || num_heads <= 0 || head_dim <= 0) return;
    if (static_cast<int>(pos_y.size()) != num_tokens || static_cast<int>(pos_x.size()) != num_tokens) return;

    const int n_dims = head_dim / 2;
    if (n_dims <= 0) return;

    // Match ggml GGML_ROPE_TYPE_VISION:
    // - rotate first/second half pairs
    // - cache index advances over pair index
    // - section sizes are expressed in head-dim units, not n_dims units
    const int sections[4] = {
        head_dim / 4,
        head_dim / 4,
        head_dim / 4,
        head_dim - 3 * (head_dim / 4)
    };
    const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const int sec_w = sections[0] + sections[1];
    const int sec_e = sec_w + sections[2];
    const float theta_scale = std::pow(10000.0f, -2.0f / static_cast<float>(n_dims));

    for (int t = 0; t < num_tokens; ++t) {
        for (int h = 0; h < num_heads; ++h) {
            float* v = x.data() +
                       static_cast<size_t>(t) * static_cast<size_t>(num_heads * head_dim) +
                       static_cast<size_t>(h) * static_cast<size_t>(head_dim);

            float theta_t = static_cast<float>(pos_y[static_cast<size_t>(t)]);
            float theta_h = static_cast<float>(pos_x[static_cast<size_t>(t)]);
            float theta_w = static_cast<float>(pos_y[static_cast<size_t>(t)]);
            float theta_e = static_cast<float>(pos_x[static_cast<size_t>(t)]);

            for (int p = 0; p < n_dims; ++p) {
                const int sector = sect_dims > 0 ? (p % sect_dims) : 0;
                if (sector == 0) {
                    theta_t = static_cast<float>(pos_y[static_cast<size_t>(t)]);
                } else if (sector == sections[0]) {
                    theta_h = static_cast<float>(pos_x[static_cast<size_t>(t)]);
                } else if (sector == sec_w) {
                    theta_w = static_cast<float>(pos_y[static_cast<size_t>(t)]);
                } else if (sector == sec_e) {
                    theta_e = static_cast<float>(pos_x[static_cast<size_t>(t)]);
                }

                float angle = theta_t;
                if (sector >= sections[0] && sector < sec_w) {
                    angle = theta_h;
                } else if (sector >= sec_w && sector < sec_e) {
                    angle = theta_w;
                } else if (sector >= sec_e) {
                    angle = theta_e;
                }
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x0 = v[p];
                const float x1 = v[p + n_dims];
                v[p] = x0 * c - x1 * s;
                v[p + n_dims] = x0 * s + x1 * c;

                theta_t *= theta_scale;
                theta_h *= theta_scale;
                theta_w *= theta_scale;
                theta_e *= theta_scale;
            }
        }
    }
}

int64_t read_int64_fallback(simdjson::dom::element root, const char* key, int64_t fallback) {
    simdjson::dom::element v;
    if (root.at_key(key).get(v) != simdjson::SUCCESS) return fallback;
    int64_t out_i = 0;
    if (v.get_int64().get(out_i) == simdjson::SUCCESS) return out_i;
    double out_d = 0.0;
    if (v.get_double().get(out_d) == simdjson::SUCCESS) return static_cast<int64_t>(out_d);
    return fallback;
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

} // namespace

bool Qwen35VisionEncoder::read_tensor_as_float(Context& ctx,
                                               ModelWeights& weights,
                                               const std::string& name,
                                               std::vector<float>& out,
                                               std::string* error_message,
                                               bool required) {
    if (!weights.has(name)) {
        if (required) {
            set_error(error_message, "Missing tensor: " + name);
            return false;
        }
        out.clear();
        return true;
    }

    Tensor& t = weights.get(name);
    size_t n = static_cast<size_t>(t.numel());
    out.resize(n);

    if (t.dtype() == dnnl::memory::data_type::f32) {
        ctx.memcpy_d2h(out.data(), t.data(), n * sizeof(float));
        return true;
    }

    if (t.dtype() == dnnl::memory::data_type::bf16) {
        std::vector<bf16> tmp(n);
        ctx.memcpy_d2h(tmp.data(), t.data(), n * sizeof(bf16));
        for (size_t i = 0; i < n; ++i) {
            out[i] = static_cast<float>(tmp[i]);
        }
        return true;
    }

    set_error(error_message, "Unsupported tensor dtype for " + name);
    return false;
}

bool Qwen35VisionEncoder::read_preprocessor(const std::string& model_dir, std::string* error_message) {
    // Keep vision token count dynamic by default. Qwen's official processor
    // derives placeholder count from the resized image grid directly instead of
    // clamping to a narrow token range.
    min_tokens_ = 1;
    max_tokens_ = std::numeric_limits<int>::max() / 4;

    min_pixels_ = 256 * 256;
    max_pixels_ = 1024 * 1024;
    if (min_pixels_ < 1024) min_pixels_ = 256 * 256;
    if (max_pixels_ < min_pixels_) max_pixels_ = min_pixels_;

    std::string p = model_dir + "/preprocessor_config.json";
    std::string text = read_file_text(p);
    if (text.empty()) return true;

    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(text);

        simdjson::dom::element mean_elem;
        if (root.at_key("image_mean").get(mean_elem) == simdjson::SUCCESS) {
            auto arr = mean_elem.get_array();
            size_t i = 0;
            for (auto v : arr) {
                if (i >= 3) break;
                double x = 0.5;
                if (v.get_double().get(x) == simdjson::SUCCESS) {
                    image_mean_[i] = static_cast<float>(x);
                }
                ++i;
            }
        }

        simdjson::dom::element std_elem;
        if (root.at_key("image_std").get(std_elem) == simdjson::SUCCESS) {
            auto arr = std_elem.get_array();
            size_t i = 0;
            for (auto v : arr) {
                if (i >= 3) break;
                double x = 0.5;
                if (v.get_double().get(x) == simdjson::SUCCESS) {
                    image_std_[i] = static_cast<float>(x);
                }
                ++i;
            }
        }

        simdjson::dom::element size_elem;
        if (root.at_key("size").get(size_elem) == simdjson::SUCCESS) {
            int64_t min_p = read_int64_fallback(size_elem, "shortest_edge", min_pixels_);
            int64_t max_p = read_int64_fallback(size_elem, "longest_edge", max_pixels_);
            if (min_p > 0) min_pixels_ = static_cast<int>(std::min<int64_t>(min_p, std::numeric_limits<int>::max()));
            if (max_p > 0) max_pixels_ = static_cast<int>(std::min<int64_t>(max_p, std::numeric_limits<int>::max()));
            if (max_pixels_ < min_pixels_) max_pixels_ = min_pixels_;

        }

        // fall through to env override stage below
    } catch (const std::exception& e) {
        set_error(error_message, std::string("preprocessor_config parse warning: ") + e.what());
    }

    // Environment overrides are applied last so experiments always take effect.
    min_tokens_ = aila::env::read_int_raw("AILA_Q35_VISION_MIN_TOKENS", min_tokens_);
    max_tokens_ = aila::env::read_int_raw("AILA_Q35_VISION_MAX_TOKENS", max_tokens_);
    min_pixels_ = aila::env::read_int_raw("AILA_Q35_VISION_MIN_PIXELS", min_pixels_);
    max_pixels_ = aila::env::read_int_raw("AILA_Q35_VISION_MAX_PIXELS", max_pixels_);
    patch_size_ = std::max(1, aila::env::read_int_raw("AILA_Q35_VISION_PATCH", patch_size_));
    merge_size_ = std::max(1, aila::env::read_int_raw("AILA_Q35_VISION_MERGE", merge_size_));

    if (min_tokens_ < 1) min_tokens_ = 1;
    if (max_tokens_ < min_tokens_) max_tokens_ = min_tokens_;
    if (min_pixels_ < 1024) min_pixels_ = 1024;
    if (max_pixels_ < min_pixels_) max_pixels_ = min_pixels_;
    return true;
}

bool Qwen35VisionEncoder::load(Context& ctx,
                               ModelWeights& weights,
                               const ModelSpec& spec,
                               const std::string& model_dir,
                               std::string* error_message) {
    loaded_ = false;
    blocks_.clear();
    patch_kernel_fused_.clear();
    patch_bias_.clear();
    pos_embed_.clear();
    merger_norm_w_.clear();
    merger_norm_b_.clear();
    merger_fc1_w_.clear();
    merger_fc1_b_.clear();
    merger_fc2_w_.clear();
    merger_fc2_b_.clear();

    if (spec.family != ModelFamily::Qwen35Hybrid || !spec.vision.enabled) {
        set_error(error_message, "Qwen35VisionEncoder requires qwen3_5 vision config");
        return false;
    }

    depth_ = spec.vision.depth;
    hidden_size_ = spec.vision.hidden_size;
    intermediate_size_ = spec.vision.intermediate_size;
    out_hidden_size_ = spec.vision.out_hidden_size;
    num_heads_ = spec.vision.num_heads;
    patch_size_ = std::max(1, spec.vision.patch_size);
    merge_size_ = std::max(1, spec.vision.spatial_merge_size);

    if (depth_ <= 0 || hidden_size_ <= 0 || intermediate_size_ <= 0 || out_hidden_size_ <= 0 ||
        num_heads_ <= 0 || hidden_size_ % num_heads_ != 0) {
        set_error(error_message, "Invalid vision config dimensions");
        return false;
    }

    head_dim_ = hidden_size_ / num_heads_;

    std::string pp_warn;
    read_preprocessor(model_dir, &pp_warn);

    std::vector<float> patch_w;
    if (!read_tensor_as_float(ctx, weights, "model.visual.patch_embed.proj.weight", patch_w, error_message)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.patch_embed.proj.bias", patch_bias_, error_message)) {
        return false;
    }

    const int p2 = patch_size_ * patch_size_;
    const size_t expect_temporal2 = static_cast<size_t>(hidden_size_) * 3u * 2u * static_cast<size_t>(p2);
    const size_t expect_spatial2d = static_cast<size_t>(hidden_size_) * 3u * static_cast<size_t>(p2);
    if (patch_w.size() != expect_temporal2 && patch_w.size() != expect_spatial2d) {
        std::ostringstream oss;
        oss << "Unexpected patch_embed.proj.weight size: got " << patch_w.size()
            << ", expected " << expect_temporal2 << " or " << expect_spatial2d;
        set_error(error_message, oss.str());
        return false;
    }
    if (patch_bias_.size() != static_cast<size_t>(hidden_size_)) {
        set_error(error_message, "Unexpected patch_embed.proj.bias shape");
        return false;
    }

    patch_kernel_fused_.resize(expect_spatial2d);
    if (patch_w.size() == expect_temporal2) {
        for (int oc = 0; oc < hidden_size_; ++oc) {
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < p2; ++i) {
                    size_t base = static_cast<size_t>(oc) * 3u * 2u * static_cast<size_t>(p2) +
                                  static_cast<size_t>(c) * 2u * static_cast<size_t>(p2) +
                                  static_cast<size_t>(i);
                    float w0 = patch_w[base + 0 * static_cast<size_t>(p2)];
                    float w1 = patch_w[base + 1 * static_cast<size_t>(p2)];
                    patch_kernel_fused_[static_cast<size_t>(oc) * 3u * static_cast<size_t>(p2) +
                                        static_cast<size_t>(c) * static_cast<size_t>(p2) +
                                        static_cast<size_t>(i)] = w0 + w1;
                }
            }
        }
    } else {
        patch_kernel_fused_ = std::move(patch_w);
    }

    if (!read_tensor_as_float(ctx, weights, "model.visual.pos_embed.weight", pos_embed_, error_message)) {
        return false;
    }
    if (pos_embed_.size() % static_cast<size_t>(hidden_size_) != 0) {
        set_error(error_message, "Unexpected model.visual.pos_embed.weight shape");
        return false;
    }

    blocks_.resize(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        BlockWeights& b = blocks_[static_cast<size_t>(i)];
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";

        if (!read_tensor_as_float(ctx, weights, p + "norm1.weight", b.ln1_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "norm1.bias", b.ln1_b, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "attn.qkv.weight", b.qkv_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "attn.qkv.bias", b.qkv_b, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "attn.proj.weight", b.proj_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "attn.proj.bias", b.proj_b, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "norm2.weight", b.ln2_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "norm2.bias", b.ln2_b, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "mlp.linear_fc1.weight", b.fc1_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "mlp.linear_fc1.bias", b.fc1_b, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "mlp.linear_fc2.weight", b.fc2_w, error_message)) return false;
        if (!read_tensor_as_float(ctx, weights, p + "mlp.linear_fc2.bias", b.fc2_b, error_message)) return false;

        if (b.ln1_w.size() != static_cast<size_t>(hidden_size_) ||
            b.ln1_b.size() != static_cast<size_t>(hidden_size_) ||
            b.ln2_w.size() != static_cast<size_t>(hidden_size_) ||
            b.ln2_b.size() != static_cast<size_t>(hidden_size_)) {
            set_error(error_message, "Vision layer norm shape mismatch");
            return false;
        }

        if (b.qkv_w.size() != static_cast<size_t>(3 * hidden_size_ * hidden_size_) ||
            b.qkv_b.size() != static_cast<size_t>(3 * hidden_size_)) {
            set_error(error_message, "Vision attn.qkv shape mismatch");
            return false;
        }
        if (b.proj_w.size() != static_cast<size_t>(hidden_size_ * hidden_size_) ||
            b.proj_b.size() != static_cast<size_t>(hidden_size_)) {
            set_error(error_message, "Vision attn.proj shape mismatch");
            return false;
        }
        if (b.fc1_w.size() != static_cast<size_t>(intermediate_size_ * hidden_size_) ||
            b.fc1_b.size() != static_cast<size_t>(intermediate_size_)) {
            set_error(error_message, "Vision mlp.linear_fc1 shape mismatch");
            return false;
        }
        if (b.fc2_w.size() != static_cast<size_t>(hidden_size_ * intermediate_size_) ||
            b.fc2_b.size() != static_cast<size_t>(hidden_size_)) {
            set_error(error_message, "Vision mlp.linear_fc2 shape mismatch");
            return false;
        }
    }

    const int merger_in_dim = hidden_size_ * merge_size_ * merge_size_;
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.norm.weight", merger_norm_w_, error_message, false)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.norm.bias", merger_norm_b_, error_message, false)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.linear_fc1.weight", merger_fc1_w_, error_message)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.linear_fc1.bias", merger_fc1_b_, error_message)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.linear_fc2.weight", merger_fc2_w_, error_message)) {
        return false;
    }
    if (!read_tensor_as_float(ctx, weights, "model.visual.merger.linear_fc2.bias", merger_fc2_b_, error_message)) {
        return false;
    }

    if (!merger_norm_w_.empty()) {
        bool per_hidden = (merger_norm_w_.size() == static_cast<size_t>(hidden_size_) &&
                           merger_norm_b_.size() == static_cast<size_t>(hidden_size_));
        bool per_merged = (merger_norm_w_.size() == static_cast<size_t>(merger_in_dim) &&
                           merger_norm_b_.size() == static_cast<size_t>(merger_in_dim));
        if (!per_hidden && !per_merged) {
            set_error(error_message, "Vision merger.norm shape mismatch");
            return false;
        }
    }
    if (merger_fc1_w_.size() % static_cast<size_t>(merger_in_dim) != 0) {
        set_error(error_message, "Vision merger.linear_fc1 weight shape mismatch");
        return false;
    }
    int merger_fc1_out = static_cast<int>(merger_fc1_w_.size() / static_cast<size_t>(merger_in_dim));
    if (!merger_fc1_b_.empty() && merger_fc1_b_.size() != static_cast<size_t>(merger_fc1_out)) {
        set_error(error_message, "Vision merger.linear_fc1 bias shape mismatch");
        return false;
    }

    if (merger_fc2_w_.size() % static_cast<size_t>(merger_fc1_out) != 0) {
        set_error(error_message, "Vision merger.linear_fc2 weight shape mismatch");
        return false;
    }
    int merger_fc2_out = static_cast<int>(merger_fc2_w_.size() / static_cast<size_t>(merger_fc1_out));
    if (merger_fc2_out != out_hidden_size_) {
        std::ostringstream oss;
        oss << "Vision merger output mismatch: config out_hidden_size=" << out_hidden_size_
            << ", weight gives " << merger_fc2_out;
        set_error(error_message, oss.str());
        return false;
    }
    if (merger_fc2_b_.size() != static_cast<size_t>(out_hidden_size_)) {
        set_error(error_message, "Vision merger.linear_fc2 bias shape mismatch");
        return false;
    }

    loaded_ = true;
    return true;
}

bool Qwen35VisionEncoder::read_image_rgb(const std::string& uri,
                                         int& width,
                                         int& height,
                                         std::vector<uint8_t>& rgb,
                                         std::string* error_message) {
#ifdef _WIN32
    if (uri.rfind("http://", 0) == 0 || uri.rfind("https://", 0) == 0) {
        set_error(error_message, "HTTP/HTTPS image URL is not supported yet; please use a local file path");
        return false;
    }

    std::string local_path = uri_to_local_path(uri);
    std::filesystem::path fp(local_path);
    if (fp.is_relative()) {
        fp = std::filesystem::absolute(fp);
    }

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
    (void)uri;
    (void)width;
    (void)height;
    (void)rgb;
    set_error(error_message, "Image decode is currently implemented for Windows only");
    return false;
#endif
}

void Qwen35VisionEncoder::resize_rgb_bicubic(const std::vector<uint8_t>& src,
                                             int src_w,
                                             int src_h,
                                             int dst_w,
                                             int dst_h,
                                             std::vector<uint8_t>& dst) {
    if (src_w == dst_w && src_h == dst_h) {
        dst = src;
        return;
    }

    auto cubic_weight = [](float x) -> float {
        constexpr float a = -0.5f;
        x = std::fabs(x);
        if (x <= 1.0f) {
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return (((a * x - 5.0f * a) * x + 8.0f * a) * x) - 4.0f * a;
        }
        return 0.0f;
    };

    dst.resize(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3u);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y_base = static_cast<int>(std::floor(src_y));

        for (int x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x_base = static_cast<int>(std::floor(src_x));
            uint8_t* q = dst.data() + (static_cast<size_t>(y) * static_cast<size_t>(dst_w) + static_cast<size_t>(x)) * 3u;

            for (int c = 0; c < 3; ++c) {
                float value = 0.0f;
                float weight_sum = 0.0f;
                for (int ky = -1; ky <= 2; ++ky) {
                    const int sy = std::clamp(y_base + ky, 0, src_h - 1);
                    const float wy = cubic_weight(src_y - static_cast<float>(y_base + ky));
                    for (int kx = -1; kx <= 2; ++kx) {
                        const int sx = std::clamp(x_base + kx, 0, src_w - 1);
                        const float wx = cubic_weight(src_x - static_cast<float>(x_base + kx));
                        const float w = wy * wx;
                        const uint8_t* p = src.data() +
                            (static_cast<size_t>(sy) * static_cast<size_t>(src_w) + static_cast<size_t>(sx)) * 3u;
                        value += static_cast<float>(p[c]) * w;
                        weight_sum += w;
                    }
                }
                if (weight_sum != 0.0f) {
                    value /= weight_sum;
                }
                q[c] = static_cast<uint8_t>(std::clamp<float>(std::round(value), 0.0f, 255.0f));
            }
        }
    }
}

void Qwen35VisionEncoder::choose_target_size(int src_w,
                                             int src_h,
                                             int align,
                                             int min_pixels,
                                             int max_pixels,
                                             int& out_w,
                                             int& out_h) {
    auto round_by_factor = [align](float x) -> int {
        return static_cast<int>(std::round(x / static_cast<float>(align))) * align;
    };
    auto ceil_by_factor = [align](float x) -> int {
        return static_cast<int>(std::ceil(x / static_cast<float>(align))) * align;
    };
    auto floor_by_factor = [align](float x) -> int {
        return static_cast<int>(std::floor(x / static_cast<float>(align))) * align;
    };

    if (src_w <= 0 || src_h <= 0) {
        out_w = align;
        out_h = align;
        return;
    }

    int w = std::max(align, round_by_factor(static_cast<float>(src_w)));
    int h = std::max(align, round_by_factor(static_cast<float>(src_h)));

    if (max_pixels > 0 && h * w > max_pixels) {
        const float beta = std::sqrt(static_cast<float>(src_h * src_w) /
                                     static_cast<float>(max_pixels));
        h = std::max(align, floor_by_factor(static_cast<float>(src_h) / beta));
        w = std::max(align, floor_by_factor(static_cast<float>(src_w) / beta));
    } else if (min_pixels > 0 && h * w < min_pixels) {
        const float beta = std::sqrt(static_cast<float>(min_pixels) /
                                     static_cast<float>(src_h * src_w));
        h = std::max(align, ceil_by_factor(static_cast<float>(src_h) * beta));
        w = std::max(align, ceil_by_factor(static_cast<float>(src_w) * beta));
    }

    out_w = w;
    out_h = h;
}

void Qwen35VisionEncoder::layer_norm_affine(const std::vector<float>& in,
                                            int rows,
                                            int cols,
                                            const std::vector<float>& gamma,
                                            const std::vector<float>& beta,
                                            float eps,
                                            std::vector<float>& out) {
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) mean += x[c];
        mean /= static_cast<float>(cols);

        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = x[c] - mean;
            var += d * d;
        }
        var /= static_cast<float>(cols);
        float inv = 1.0f / std::sqrt(var + eps);

        float* y = out.data() + static_cast<size_t>(r) * static_cast<size_t>(cols);
        for (int c = 0; c < cols; ++c) {
            float g = (gamma.size() == static_cast<size_t>(cols)) ? gamma[static_cast<size_t>(c)] : 1.0f;
            float b = (beta.size() == static_cast<size_t>(cols)) ? beta[static_cast<size_t>(c)] : 0.0f;
            y[c] = (x[c] - mean) * inv * g + b;
        }
    }
}

void Qwen35VisionEncoder::linear_rowmajor(const std::vector<float>& in,
                                          int rows,
                                          int in_dim,
                                          const std::vector<float>& weight,
                                          const std::vector<float>& bias,
                                          int out_dim,
                                          std::vector<float>& out) {
    out.assign(static_cast<size_t>(rows) * static_cast<size_t>(out_dim), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* x = in.data() + static_cast<size_t>(r) * static_cast<size_t>(in_dim);
        float* y = out.data() + static_cast<size_t>(r) * static_cast<size_t>(out_dim);
        for (int od = 0; od < out_dim; ++od) {
            const float* w = weight.data() + static_cast<size_t>(od) * static_cast<size_t>(in_dim);
            float acc = 0.0f;
            for (int id = 0; id < in_dim; ++id) {
                acc += x[id] * w[id];
            }
            if (!bias.empty()) acc += bias[static_cast<size_t>(od)];
            y[od] = acc;
        }
    }
}

void Qwen35VisionEncoder::gelu_tanh_inplace(std::vector<float>& x) {
    for (float& v : x) {
        v = gelu_tanh_scalar(v);
    }
}

void Qwen35VisionEncoder::softmax_inplace(std::vector<float>& x) {
    if (x.empty()) return;
    float m = -std::numeric_limits<float>::max();
    for (float v : x) m = std::max(m, v);
    float s = 0.0f;
    for (float& v : x) {
        v = std::exp(v - m);
        s += v;
    }
    if (s <= 0.0f) {
        float u = 1.0f / static_cast<float>(x.size());
        for (float& v : x) v = u;
        return;
    }
    float inv = 1.0f / s;
    for (float& v : x) v *= inv;
}

bool Qwen35VisionEncoder::encode_image(const std::string& uri,
                                       VisionEncodeResult& out,
                                       std::string* error_message) const {
    out = VisionEncodeResult{};
    if (!loaded_) {
        set_error(error_message, "Vision encoder is not loaded");
        return false;
    }

    int src_w = 0;
    int src_h = 0;
    std::vector<uint8_t> src_rgb;
    if (!read_image_rgb(uri, src_w, src_h, src_rgb, error_message)) {
        return false;
    }

    const int align = std::max(1, patch_size_ * merge_size_);
    int target_w = align;
    int target_h = align;
    choose_target_size(src_w, src_h, align, min_pixels_, max_pixels_, target_w, target_h);

    auto merged_tokens = [&](int w, int h) -> int {
        int g_w = std::max(1, w / align);
        int g_h = std::max(1, h / align);
        return g_w * g_h;
    };
    auto clamp_dims = [&](int& w, int& h) {
        w = std::max(w, align);
        h = std::max(h, align);
        w = (w / align) * align;
        h = (h / align) * align;
        if (w < align) w = align;
        if (h < align) h = align;
    };

    clamp_dims(target_w, target_h);
    int m_tok = merged_tokens(target_w, target_h);
    if (m_tok > max_tokens_ || m_tok < min_tokens_) {
        int desired = std::clamp(m_tok, min_tokens_, max_tokens_);
        float scale = std::sqrt(static_cast<float>(desired) / static_cast<float>(std::max(1, m_tok)));
        target_w = static_cast<int>(std::round(target_w * scale));
        target_h = static_cast<int>(std::round(target_h * scale));
        clamp_dims(target_w, target_h);
    }
    m_tok = merged_tokens(target_w, target_h);
    while (m_tok > max_tokens_ && target_w > align && target_h > align) {
        if (target_w >= target_h) target_w -= align;
        else target_h -= align;
        clamp_dims(target_w, target_h);
        m_tok = merged_tokens(target_w, target_h);
    }
    while (m_tok < min_tokens_) {
        if (target_w <= target_h) target_w += align;
        else target_h += align;
        clamp_dims(target_w, target_h);
        m_tok = merged_tokens(target_w, target_h);
        if (target_w > 4096 || target_h > 4096) break;
    }

    std::vector<uint8_t> resized_rgb;
    resize_rgb_bicubic(src_rgb, src_w, src_h, target_w, target_h, resized_rgb);

    std::vector<float> chw(static_cast<size_t>(3) * static_cast<size_t>(target_h) * static_cast<size_t>(target_w));
    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            const uint8_t* p = resized_rgb.data() + (static_cast<size_t>(y) * static_cast<size_t>(target_w) + static_cast<size_t>(x)) * 3u;
            float r = static_cast<float>(p[0]) / 255.0f;
            float g = static_cast<float>(p[1]) / 255.0f;
            float b = static_cast<float>(p[2]) / 255.0f;
            chw[(0 * target_h + y) * target_w + x] = (r - image_mean_[0]) / image_std_[0];
            chw[(1 * target_h + y) * target_w + x] = (g - image_mean_[1]) / image_std_[1];
            chw[(2 * target_h + y) * target_w + x] = (b - image_mean_[2]) / image_std_[2];
        }
    }

    const int patch_grid_w = target_w / patch_size_;
    const int patch_grid_h = target_h / patch_size_;
    const int num_patches = patch_grid_w * patch_grid_h;
    const int patch_dim = 3 * patch_size_ * patch_size_;

    if (num_patches <= 0) {
        set_error(error_message, "Invalid patch grid after resize");
        return false;
    }

    std::vector<float> patch_rows(static_cast<size_t>(num_patches) * static_cast<size_t>(patch_dim));
    for (int py = 0; py < patch_grid_h; ++py) {
        for (int px = 0; px < patch_grid_w; ++px) {
            const int row = py * patch_grid_w + px;
            float* dst = patch_rows.data() + static_cast<size_t>(row) * static_cast<size_t>(patch_dim);
            int t = 0;
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < patch_size_; ++ky) {
                    int iy = py * patch_size_ + ky;
                    for (int kx = 0; kx < patch_size_; ++kx) {
                        int ix = px * patch_size_ + kx;
                        dst[t++] = chw[(c * target_h + iy) * target_w + ix];
                    }
                }
            }
        }
    }

    std::vector<float> tokens;
    linear_rowmajor(patch_rows, num_patches, patch_dim,
                    patch_kernel_fused_, patch_bias_, hidden_size_, tokens);

    const int pos_len = static_cast<int>(pos_embed_.size() / static_cast<size_t>(hidden_size_));
    if (pos_len <= 0) {
        set_error(error_message, "Invalid vision position embedding");
        return false;
    }

    if (pos_len == num_patches) {
        for (int i = 0; i < num_patches * hidden_size_; ++i) {
            tokens[static_cast<size_t>(i)] += pos_embed_[static_cast<size_t>(i)];
        }
    } else {
        int base_side = static_cast<int>(std::round(std::sqrt(static_cast<float>(pos_len))));
        if (base_side * base_side != pos_len) {
            int n = std::min(pos_len, num_patches);
            for (int i = 0; i < n; ++i) {
                float* row = tokens.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden_size_);
                const float* pe = pos_embed_.data() + static_cast<size_t>(i) * static_cast<size_t>(hidden_size_);
                for (int d = 0; d < hidden_size_; ++d) row[d] += pe[d];
            }
        } else {
            for (int y = 0; y < patch_grid_h; ++y) {
                float fy = 0.0f;
                if (patch_grid_h > 1) {
                    fy = static_cast<float>(y) * static_cast<float>(base_side - 1) /
                         static_cast<float>(patch_grid_h - 1);
                }
                int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, base_side - 1);
                int y1 = std::clamp(y0 + 1, 0, base_side - 1);
                float wy1 = fy - static_cast<float>(y0);
                float wy0 = 1.0f - wy1;

                for (int x = 0; x < patch_grid_w; ++x) {
                    float fx = 0.0f;
                    if (patch_grid_w > 1) {
                        fx = static_cast<float>(x) * static_cast<float>(base_side - 1) /
                             static_cast<float>(patch_grid_w - 1);
                    }
                    int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, base_side - 1);
                    int x1 = std::clamp(x0 + 1, 0, base_side - 1);
                    float wx1 = fx - static_cast<float>(x0);
                    float wx0 = 1.0f - wx1;

                    int id00 = y0 * base_side + x0;
                    int id01 = y0 * base_side + x1;
                    int id10 = y1 * base_side + x0;
                    int id11 = y1 * base_side + x1;

                    float* row = tokens.data() +
                                 static_cast<size_t>(y * patch_grid_w + x) * static_cast<size_t>(hidden_size_);
                    const float* pe00 = pos_embed_.data() + static_cast<size_t>(id00) * static_cast<size_t>(hidden_size_);
                    const float* pe01 = pos_embed_.data() + static_cast<size_t>(id01) * static_cast<size_t>(hidden_size_);
                    const float* pe10 = pos_embed_.data() + static_cast<size_t>(id10) * static_cast<size_t>(hidden_size_);
                    const float* pe11 = pos_embed_.data() + static_cast<size_t>(id11) * static_cast<size_t>(hidden_size_);

                    for (int d = 0; d < hidden_size_; ++d) {
                        float v = wy0 * (wx0 * pe00[d] + wx1 * pe01[d]) +
                                  wy1 * (wx0 * pe10[d] + wx1 * pe11[d]);
                        row[d] += v;
                    }
                }
            }
        }
    }

    std::vector<float> grouped_tokens;
    std::vector<int> pos_y;
    std::vector<int> pos_x;
    reorder_qwen_block_sequence(tokens, patch_grid_w, patch_grid_h, hidden_size_, merge_size_,
                                grouped_tokens, &pos_y, &pos_x);
    tokens = std::move(grouped_tokens);

    std::vector<float> ln1;
    std::vector<float> qkv;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> attn_out;
    std::vector<float> proj_out;
    std::vector<float> ln2;
    std::vector<float> ffn1;
    std::vector<float> ffn2;
    std::vector<float> probs(static_cast<size_t>(num_patches));

    for (const auto& b : blocks_) {
        layer_norm_affine(tokens, num_patches, hidden_size_,
                          b.ln1_w, b.ln1_b, 1e-6f, ln1);

        linear_rowmajor(ln1, num_patches, hidden_size_,
                        b.qkv_w, b.qkv_b, hidden_size_ * 3, qkv);

        q.resize(static_cast<size_t>(num_patches) * static_cast<size_t>(hidden_size_));
        k.resize(static_cast<size_t>(num_patches) * static_cast<size_t>(hidden_size_));
        v.resize(static_cast<size_t>(num_patches) * static_cast<size_t>(hidden_size_));
        for (int t = 0; t < num_patches; ++t) {
            const float* src = qkv.data() + static_cast<size_t>(t) * static_cast<size_t>(hidden_size_ * 3);
            std::memcpy(q.data() + static_cast<size_t>(t) * static_cast<size_t>(hidden_size_),
                        src, static_cast<size_t>(hidden_size_) * sizeof(float));
            std::memcpy(k.data() + static_cast<size_t>(t) * static_cast<size_t>(hidden_size_),
                        src + hidden_size_, static_cast<size_t>(hidden_size_) * sizeof(float));
            std::memcpy(v.data() + static_cast<size_t>(t) * static_cast<size_t>(hidden_size_),
                        src + hidden_size_ * 2, static_cast<size_t>(hidden_size_) * sizeof(float));
        }

        apply_qwen_vision_mrope_inplace(q, num_patches, num_heads_, head_dim_, pos_y, pos_x);
        apply_qwen_vision_mrope_inplace(k, num_patches, num_heads_, head_dim_, pos_y, pos_x);

        attn_out.assign(static_cast<size_t>(num_patches) * static_cast<size_t>(hidden_size_), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        for (int h = 0; h < num_heads_; ++h) {
            int hd_off = h * head_dim_;
            for (int qi = 0; qi < num_patches; ++qi) {
                float max_s = -std::numeric_limits<float>::max();
                const float* qv = q.data() + static_cast<size_t>(qi) * static_cast<size_t>(hidden_size_) + hd_off;
                for (int kj = 0; kj < num_patches; ++kj) {
                    const float* kv = k.data() + static_cast<size_t>(kj) * static_cast<size_t>(hidden_size_) + hd_off;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) dot += qv[d] * kv[d];
                    float s = dot * scale;
                    probs[static_cast<size_t>(kj)] = s;
                    if (s > max_s) max_s = s;
                }

                float sum_e = 0.0f;
                for (int kj = 0; kj < num_patches; ++kj) {
                    float e = std::exp(probs[static_cast<size_t>(kj)] - max_s);
                    probs[static_cast<size_t>(kj)] = e;
                    sum_e += e;
                }
                float inv = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;

                float* outv = attn_out.data() + static_cast<size_t>(qi) * static_cast<size_t>(hidden_size_) + hd_off;
                for (int d = 0; d < head_dim_; ++d) {
                    float acc = 0.0f;
                    for (int kj = 0; kj < num_patches; ++kj) {
                        const float* vv = v.data() + static_cast<size_t>(kj) * static_cast<size_t>(hidden_size_) + hd_off;
                        acc += probs[static_cast<size_t>(kj)] * inv * vv[d];
                    }
                    outv[d] = acc;
                }
            }
        }

        linear_rowmajor(attn_out, num_patches, hidden_size_,
                        b.proj_w, b.proj_b, hidden_size_, proj_out);
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] += proj_out[i];
        }

        layer_norm_affine(tokens, num_patches, hidden_size_,
                          b.ln2_w, b.ln2_b, 1e-6f, ln2);
        linear_rowmajor(ln2, num_patches, hidden_size_,
                        b.fc1_w, b.fc1_b, intermediate_size_, ffn1);
        gelu_tanh_inplace(ffn1);
        linear_rowmajor(ffn1, num_patches, intermediate_size_,
                        b.fc2_w, b.fc2_b, hidden_size_, ffn2);
        for (size_t i = 0; i < tokens.size(); ++i) {
            tokens[i] += ffn2[i];
        }
    }

    std::vector<float> tokens_for_merge;
    if (!merger_norm_w_.empty() && merger_norm_w_.size() == static_cast<size_t>(hidden_size_)) {
        layer_norm_affine(tokens, num_patches, hidden_size_,
                          merger_norm_w_, merger_norm_b_, 1e-6f, tokens_for_merge);
    } else {
        tokens_for_merge = tokens;
    }

    const int merged_w = patch_grid_w / merge_size_;
    const int merged_h = patch_grid_h / merge_size_;
    const int out_tokens = merged_w * merged_h;
    const int merger_in_dim = hidden_size_ * merge_size_ * merge_size_;

    if (out_tokens <= 0) {
        set_error(error_message, "Invalid merged vision token count");
        return false;
    }

    std::vector<float> merged(static_cast<size_t>(out_tokens) * static_cast<size_t>(merger_in_dim));
    for (int my = 0; my < merged_h; ++my) {
        for (int mx = 0; mx < merged_w; ++mx) {
            const int out_row = my * merged_w + mx;
            float* dst = merged.data() + static_cast<size_t>(out_row) * static_cast<size_t>(merger_in_dim);
            const int block_size = merge_size_ * merge_size_;
            for (int block = 0; block < block_size; ++block) {
                const int src_row = out_row * block_size + block;
                const float* src = tokens_for_merge.data() + static_cast<size_t>(src_row) * static_cast<size_t>(hidden_size_);
                std::memcpy(dst + static_cast<size_t>(block) * static_cast<size_t>(hidden_size_),
                            src, static_cast<size_t>(hidden_size_) * sizeof(float));
            }
        }
    }

    std::vector<float> merger_normed;
    if (!merger_norm_w_.empty() && merger_norm_w_.size() == static_cast<size_t>(merger_in_dim)) {
        layer_norm_affine(merged, out_tokens, merger_in_dim,
                          merger_norm_w_, merger_norm_b_, 1e-6f, merger_normed);
    } else {
        merger_normed = std::move(merged);
    }

    const int merger_fc1_out =
        !merger_fc1_b_.empty() ? static_cast<int>(merger_fc1_b_.size())
                               : static_cast<int>(merger_fc1_w_.size() / static_cast<size_t>(merger_in_dim));
    std::vector<float> merger_h;
    linear_rowmajor(merger_normed, out_tokens, merger_in_dim,
                    merger_fc1_w_, merger_fc1_b_, merger_fc1_out, merger_h);
    gelu_tanh_inplace(merger_h);

    std::vector<float> merger_out;
    linear_rowmajor(merger_h, out_tokens, merger_fc1_out,
                    merger_fc2_w_, merger_fc2_b_, out_hidden_size_, merger_out);

    out.token_count = out_tokens;
    out.llm_grid_t = 1;
    out.llm_grid_h = merged_h;
    out.llm_grid_w = merged_w;
    out.embeddings.resize(static_cast<size_t>(out_tokens) * static_cast<size_t>(out_hidden_size_));
    for (size_t i = 0; i < merger_out.size(); ++i) {
        out.embeddings[i] = bf16(merger_out[i]);
    }
    return true;
}

} // namespace vision
} // namespace aila
