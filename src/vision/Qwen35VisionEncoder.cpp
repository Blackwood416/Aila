#include "Qwen35VisionEncoder.hpp"

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

int vision_cpu_threads() {
    static int threads = []() {
        int n = aila::env::read_int_raw("AILA_Q35_VISION_THREADS", 0);
        if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency());
        return std::max(1, n);
    }();
    return threads;
}

template <typename Fn>
void parallel_for_1d(int begin, int end, int min_items_per_thread, const Fn& fn) {
    const int total = end - begin;
    if (total <= 0) return;

    int threads = vision_cpu_threads();
    if (threads <= 1 || total <= std::max(1, min_items_per_thread)) {
        for (int i = begin; i < end; ++i) fn(i);
        return;
    }

    threads = std::min(threads, std::max(1, total / std::max(1, min_items_per_thread)));
    if (threads <= 1) {
        for (int i = begin; i < end; ++i) fn(i);
        return;
    }

    const int chunk = (total + threads - 1) / threads;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads - 1));

    for (int t = 0; t < threads - 1; ++t) {
        const int s = begin + t * chunk;
        const int e = std::min(end, s + chunk);
        workers.emplace_back([s, e, &fn]() {
            for (int i = s; i < e; ++i) fn(i);
        });
    }

    const int main_begin = begin + (threads - 1) * chunk;
    for (int i = main_begin; i < end; ++i) fn(i);

    for (auto& worker : workers) worker.join();
}

std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
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

Qwen35VisionEncoder::~Qwen35VisionEncoder() {
    release_runtime_buffers();
}

Tensor* Qwen35VisionEncoder::get_tensor(ModelWeights& weights,
                                        const std::string& name,
                                        std::string* error_message,
                                        bool required) {
    if (!weights.has(name)) {
        if (required) {
            set_error(error_message, "Missing tensor: " + name);
        }
        return nullptr;
    }
    return &weights.get(name);
}

Tensor* Qwen35VisionEncoder::ensure_bf16_weight(Context& ctx,
                                                ModelWeights& weights,
                                                const std::string& name,
                                                std::string* error_message,
                                                bool required) {
    Tensor* src = get_tensor(weights, name, error_message, required);
    if (!src) return nullptr;
    if (src->dtype() == dnnl::memory::data_type::bf16) {
        return src;
    }
    if (src->dtype() != dnnl::memory::data_type::f32) {
        set_error(error_message, "Unsupported tensor dtype for " + name + ": expected bf16/f32");
        return nullptr;
    }

    owned_weights_.emplace_back(Tensor::allocate(ctx, src->shape(), dnnl::memory::data_type::bf16));
    Tensor& dst = owned_weights_.back();
    const float* src_ptr = static_cast<const float*>(src->data());
    bf16* dst_ptr = static_cast<bf16*>(dst.data());
    const size_t n = static_cast<size_t>(src->numel());
    ctx.queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        const size_t i = idx[0];
        dst_ptr[i] = bf16(src_ptr[i]);
    });
    return &dst;
}

bool Qwen35VisionEncoder::prepare_patch_weight(Context& ctx,
                                               ModelWeights& weights,
                                               std::string* error_message) {
    constexpr const char* kName = "model.visual.patch_embed.proj.weight";
    Tensor* src = get_tensor(weights, kName, error_message);
    if (!src) return false;

    const int p2 = patch_size_ * patch_size_;
    const int patch_dim = 3 * p2;
    const int64_t expect_temporal2 = static_cast<int64_t>(hidden_size_) * 3 * 2 * p2;
    const int64_t expect_spatial2d = static_cast<int64_t>(hidden_size_) * 3 * p2;
    const int64_t numel = src->numel();

    if (numel != expect_temporal2 && numel != expect_spatial2d) {
        std::ostringstream oss;
        oss << "Unexpected patch_embed.proj.weight size: got " << numel
            << ", expected " << expect_temporal2 << " or " << expect_spatial2d;
        set_error(error_message, oss.str());
        return false;
    }

    if (numel == expect_spatial2d) {
        patch_weight_ = (src->dtype() == dnnl::memory::data_type::bf16)
                            ? src
                            : ensure_bf16_weight(ctx, weights, kName, error_message);
        return patch_weight_ != nullptr;
    }

    if (src->dtype() != dnnl::memory::data_type::bf16 &&
        src->dtype() != dnnl::memory::data_type::f32) {
        set_error(error_message, "Unsupported patch_embed.proj.weight dtype: expected bf16/f32");
        return false;
    }

    owned_weights_.emplace_back(Tensor::allocate(ctx,
                                                 {hidden_size_, patch_dim},
                                                 dnnl::memory::data_type::bf16));
    Tensor& fused = owned_weights_.back();
    bf16* dst_ptr = static_cast<bf16*>(fused.data());
    const int total = hidden_size_ * patch_dim;

    if (src->dtype() == dnnl::memory::data_type::f32) {
        const float* src_ptr = static_cast<const float*>(src->data());
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> idx) {
            const int flat = static_cast<int>(idx[0]);
            const int oc = flat / patch_dim;
            const int rem = flat % patch_dim;
            const int c = rem / p2;
            const int inner = rem % p2;
            const int base = oc * 3 * 2 * p2 + c * 2 * p2 + inner;
            dst_ptr[flat] = bf16(src_ptr[base] + src_ptr[base + p2]);
        });
    } else {
        const bf16* src_ptr = static_cast<const bf16*>(src->data());
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> idx) {
            const int flat = static_cast<int>(idx[0]);
            const int oc = flat / patch_dim;
            const int rem = flat % patch_dim;
            const int c = rem / p2;
            const int inner = rem % p2;
            const int base = oc * 3 * 2 * p2 + c * 2 * p2 + inner;
            dst_ptr[flat] = bf16(static_cast<float>(src_ptr[base]) +
                                 static_cast<float>(src_ptr[base + p2]));
        });
    }

    patch_weight_ = &fused;
    return true;
}

bool Qwen35VisionEncoder::read_preprocessor(const std::string& model_dir, std::string* error_message) {
    std::string p = model_dir + "/preprocessor_config.json";
    std::string text = read_file_text(p);
    if (!text.empty()) {
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
        } catch (const std::exception& e) {
            set_error(error_message, std::string("preprocessor_config parse warning: ") + e.what());
        }
    }

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
    release_runtime_buffers();

    owned_weights_.clear();
    blocks_.clear();
    patch_weight_ = nullptr;
    patch_bias_ = nullptr;
    pos_embed_ = nullptr;
    pos_embed_len_ = 0;
    merger_norm_weight_ = nullptr;
    merger_norm_bias_ = nullptr;
    merger_fc1_bias_ = nullptr;
    merger_fc1_out_ = 0;
    merger_fc2_bias_ = nullptr;
    patch_proj_ = Linear();
    merger_fc1_ = Linear();
    merger_fc2_ = Linear();
    ctx_ = &ctx;

    if (spec.family != ModelFamily::Qwen35Hybrid || !spec.vision.enabled) {
        set_error(error_message, "Qwen35VisionEncoder requires qwen3_5 vision config");
        return false;
    }

    depth_ = spec.vision.depth;
    hidden_size_ = spec.vision.hidden_size;
    intermediate_size_ = spec.vision.intermediate_size;
    out_hidden_size_ = spec.vision.out_hidden_size;
    num_heads_ = spec.vision.num_heads;
    head_dim_ = 0;

    patch_size_ = std::max(1, spec.vision.patch_size);
    merge_size_ = std::max(1, spec.vision.spatial_merge_size);
    min_tokens_ = 1;
    max_tokens_ = 1024;
    min_pixels_ = 256 * 256;
    max_pixels_ = 1024 * 1024;
    image_mean_[0] = image_mean_[1] = image_mean_[2] = 0.5f;
    image_std_[0] = image_std_[1] = image_std_[2] = 0.5f;

    if (depth_ <= 0 || hidden_size_ <= 0 || intermediate_size_ <= 0 || out_hidden_size_ <= 0 ||
        num_heads_ <= 0 || hidden_size_ % num_heads_ != 0) {
        set_error(error_message, "Invalid vision config dimensions");
        return false;
    }
    head_dim_ = hidden_size_ / num_heads_;

    std::string pp_warn;
    read_preprocessor(model_dir, &pp_warn);
    if (!pp_warn.empty()) {
        AILA_LOG_WARN("[Vision] %s", pp_warn.c_str());
    }

    auto require_floatish = [&](Tensor* t, const std::string& name) -> bool {
        if (!t) return false;
        if (t->dtype() != dnnl::memory::data_type::bf16 &&
            t->dtype() != dnnl::memory::data_type::f32) {
            set_error(error_message, "Unsupported tensor dtype for " + name + ": expected bf16/f32");
            return false;
        }
        return true;
    };

    patch_bias_ = get_tensor(weights, "model.visual.patch_embed.proj.bias", error_message);
    if (!patch_bias_) return false;
    if (!require_floatish(patch_bias_, "model.visual.patch_embed.proj.bias")) return false;
    if (patch_bias_->numel() != hidden_size_) {
        set_error(error_message, "Unexpected patch_embed.proj.bias shape");
        return false;
    }
    if (!prepare_patch_weight(ctx, weights, error_message)) return false;
    patch_proj_.init(ctx, *patch_weight_, 3 * patch_size_ * patch_size_, hidden_size_);

    pos_embed_ = get_tensor(weights, "model.visual.pos_embed.weight", error_message);
    if (!pos_embed_) return false;
    if (!require_floatish(pos_embed_, "model.visual.pos_embed.weight")) return false;
    if (pos_embed_->numel() % hidden_size_ != 0) {
        set_error(error_message, "Unexpected model.visual.pos_embed.weight shape");
        return false;
    }
    pos_embed_len_ = static_cast<int>(pos_embed_->numel() / hidden_size_);
    if (pos_embed_len_ <= 0) {
        set_error(error_message, "Invalid model.visual.pos_embed.weight shape");
        return false;
    }

    blocks_.resize(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        BlockWeights& b = blocks_[static_cast<size_t>(i)];
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";

        b.ln1_weight = get_tensor(weights, p + "norm1.weight", error_message);
        b.ln1_bias = get_tensor(weights, p + "norm1.bias", error_message);
        Tensor* qkv_weight = ensure_bf16_weight(ctx, weights, p + "attn.qkv.weight", error_message);
        b.qkv_bias = get_tensor(weights, p + "attn.qkv.bias", error_message);
        Tensor* proj_weight = ensure_bf16_weight(ctx, weights, p + "attn.proj.weight", error_message);
        b.proj_bias = get_tensor(weights, p + "attn.proj.bias", error_message);
        b.ln2_weight = get_tensor(weights, p + "norm2.weight", error_message);
        b.ln2_bias = get_tensor(weights, p + "norm2.bias", error_message);
        Tensor* fc1_weight = ensure_bf16_weight(ctx, weights, p + "mlp.linear_fc1.weight", error_message);
        b.fc1_bias = get_tensor(weights, p + "mlp.linear_fc1.bias", error_message);
        Tensor* fc2_weight = ensure_bf16_weight(ctx, weights, p + "mlp.linear_fc2.weight", error_message);
        b.fc2_bias = get_tensor(weights, p + "mlp.linear_fc2.bias", error_message);

        if (!b.ln1_weight || !b.ln1_bias || !qkv_weight || !b.qkv_bias || !proj_weight || !b.proj_bias ||
            !b.ln2_weight || !b.ln2_bias || !fc1_weight || !b.fc1_bias || !fc2_weight || !b.fc2_bias) {
            return false;
        }

        if (!require_floatish(b.ln1_weight, p + "norm1.weight") ||
            !require_floatish(b.ln1_bias, p + "norm1.bias") ||
            !require_floatish(b.qkv_bias, p + "attn.qkv.bias") ||
            !require_floatish(b.proj_bias, p + "attn.proj.bias") ||
            !require_floatish(b.ln2_weight, p + "norm2.weight") ||
            !require_floatish(b.ln2_bias, p + "norm2.bias") ||
            !require_floatish(b.fc1_bias, p + "mlp.linear_fc1.bias") ||
            !require_floatish(b.fc2_bias, p + "mlp.linear_fc2.bias")) {
            return false;
        }

        if (b.ln1_weight->numel() != hidden_size_ ||
            b.ln1_bias->numel() != hidden_size_ ||
            b.ln2_weight->numel() != hidden_size_ ||
            b.ln2_bias->numel() != hidden_size_) {
            set_error(error_message, "Vision layer norm shape mismatch");
            return false;
        }

        if (qkv_weight->numel() != static_cast<int64_t>(3 * hidden_size_) * hidden_size_ ||
            b.qkv_bias->numel() != static_cast<int64_t>(3 * hidden_size_)) {
            set_error(error_message, "Vision attn.qkv shape mismatch");
            return false;
        }
        if (proj_weight->numel() != static_cast<int64_t>(hidden_size_) * hidden_size_ ||
            b.proj_bias->numel() != hidden_size_) {
            set_error(error_message, "Vision attn.proj shape mismatch");
            return false;
        }
        if (fc1_weight->numel() != static_cast<int64_t>(intermediate_size_) * hidden_size_ ||
            b.fc1_bias->numel() != intermediate_size_) {
            set_error(error_message, "Vision mlp.linear_fc1 shape mismatch");
            return false;
        }
        if (fc2_weight->numel() != static_cast<int64_t>(hidden_size_) * intermediate_size_ ||
            b.fc2_bias->numel() != hidden_size_) {
            set_error(error_message, "Vision mlp.linear_fc2 shape mismatch");
            return false;
        }

        b.qkv = Linear();
        b.qkv.init(ctx, *qkv_weight, hidden_size_, 3 * hidden_size_);
        b.proj = Linear();
        b.proj.init(ctx, *proj_weight, hidden_size_, hidden_size_);
        b.fc1 = Linear();
        b.fc1.init(ctx, *fc1_weight, hidden_size_, intermediate_size_);
        b.fc2 = Linear();
        b.fc2.init(ctx, *fc2_weight, intermediate_size_, hidden_size_);
    }

    const int merger_in_dim = hidden_size_ * merge_size_ * merge_size_;
    merger_norm_weight_ = get_tensor(weights, "model.visual.merger.norm.weight", nullptr, false);
    merger_norm_bias_ = get_tensor(weights, "model.visual.merger.norm.bias", nullptr, false);
    if ((merger_norm_weight_ == nullptr) != (merger_norm_bias_ == nullptr)) {
        set_error(error_message, "Vision merger.norm weight/bias presence mismatch");
        return false;
    }
    if (merger_norm_weight_) {
        if (!require_floatish(merger_norm_weight_, "model.visual.merger.norm.weight") ||
            !require_floatish(merger_norm_bias_, "model.visual.merger.norm.bias")) {
            return false;
        }
        const bool per_hidden = merger_norm_weight_->numel() == hidden_size_ &&
                                merger_norm_bias_->numel() == hidden_size_;
        const bool per_merged = merger_norm_weight_->numel() == merger_in_dim &&
                                merger_norm_bias_->numel() == merger_in_dim;
        if (!per_hidden && !per_merged) {
            set_error(error_message, "Vision merger.norm shape mismatch");
            return false;
        }
    }

    Tensor* merger_fc1_weight = ensure_bf16_weight(ctx, weights, "model.visual.merger.linear_fc1.weight", error_message);
    merger_fc1_bias_ = get_tensor(weights, "model.visual.merger.linear_fc1.bias", error_message);
    Tensor* merger_fc2_weight = ensure_bf16_weight(ctx, weights, "model.visual.merger.linear_fc2.weight", error_message);
    merger_fc2_bias_ = get_tensor(weights, "model.visual.merger.linear_fc2.bias", error_message);
    if (!merger_fc1_weight || !merger_fc1_bias_ || !merger_fc2_weight || !merger_fc2_bias_) {
        return false;
    }
    if (!require_floatish(merger_fc1_bias_, "model.visual.merger.linear_fc1.bias") ||
        !require_floatish(merger_fc2_bias_, "model.visual.merger.linear_fc2.bias")) {
        return false;
    }

    if (merger_fc1_weight->numel() % merger_in_dim != 0) {
        set_error(error_message, "Vision merger.linear_fc1 weight shape mismatch");
        return false;
    }
    merger_fc1_out_ = static_cast<int>(merger_fc1_weight->numel() / merger_in_dim);
    if (merger_fc1_out_ <= 0 || merger_fc1_bias_->numel() != merger_fc1_out_) {
        set_error(error_message, "Vision merger.linear_fc1 bias shape mismatch");
        return false;
    }

    if (merger_fc2_weight->numel() % merger_fc1_out_ != 0) {
        set_error(error_message, "Vision merger.linear_fc2 weight shape mismatch");
        return false;
    }
    const int merger_fc2_out = static_cast<int>(merger_fc2_weight->numel() / merger_fc1_out_);
    if (merger_fc2_out != out_hidden_size_) {
        std::ostringstream oss;
        oss << "Vision merger output mismatch: config out_hidden_size=" << out_hidden_size_
            << ", weight gives " << merger_fc2_out;
        set_error(error_message, oss.str());
        return false;
    }
    if (merger_fc2_bias_->numel() != out_hidden_size_) {
        set_error(error_message, "Vision merger.linear_fc2 bias shape mismatch");
        return false;
    }

    merger_fc1_.init(ctx, *merger_fc1_weight, merger_in_dim, merger_fc1_out_);
    merger_fc2_.init(ctx, *merger_fc2_weight, merger_fc1_out_, out_hidden_size_);

    ctx.synchronize();
    loaded_ = true;
    return true;
}

void Qwen35VisionEncoder::ensure_runtime_buffers(int image_bytes,
                                                 int num_patches,
                                                 int out_tokens,
                                                 int merger_in_dim) {
    if (!ctx_) return;

    if (image_bytes > image_bytes_capacity_) {
        buf_.image_rgb = Tensor::allocate(*ctx_, {image_bytes}, dnnl::memory::data_type::u8);
        image_bytes_capacity_ = image_bytes;
    }

    if (num_patches > patch_capacity_) {
        const int patch_dim = 3 * patch_size_ * patch_size_;
        buf_.patch_rows = Tensor::allocate(*ctx_, {num_patches, patch_dim}, dnnl::memory::data_type::bf16);
        buf_.hidden_a = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.hidden_b = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.normed = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.qkv = Tensor::allocate(*ctx_, {num_patches, 3 * hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.q = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.k = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.v = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.attn_out = Tensor::allocate(*ctx_, {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
        buf_.ffn = Tensor::allocate(*ctx_, {num_patches, intermediate_size_}, dnnl::memory::data_type::bf16);
        buf_.scores = Tensor::allocate(*ctx_, {num_heads_, num_patches, num_patches}, dnnl::memory::data_type::f32);
        patch_capacity_ = num_patches;
    }

    if (out_tokens > out_token_capacity_) {
        buf_.merger_normed = Tensor::allocate(*ctx_, {out_tokens, merger_in_dim}, dnnl::memory::data_type::bf16);
        buf_.merger_hidden = Tensor::allocate(*ctx_, {out_tokens, merger_fc1_out_}, dnnl::memory::data_type::bf16);
        buf_.merger_out = Tensor::allocate(*ctx_, {out_tokens, out_hidden_size_}, dnnl::memory::data_type::bf16);
        out_token_capacity_ = out_tokens;
    }
}

void Qwen35VisionEncoder::ensure_position_buffers(int num_patches) {
    if (!ctx_) return;
    if (num_patches > pos_capacity_) {
        if (pos_y_device_) ctx_->free_device(pos_y_device_);
        if (pos_x_device_) ctx_->free_device(pos_x_device_);
        const size_t bytes = static_cast<size_t>(num_patches) * sizeof(int);
        pos_y_device_ = static_cast<int*>(ctx_->alloc_device(bytes));
        pos_x_device_ = static_cast<int*>(ctx_->alloc_device(bytes));
        pos_capacity_ = num_patches;
    }
    pos_y_host_.resize(static_cast<size_t>(num_patches));
    pos_x_host_.resize(static_cast<size_t>(num_patches));
}

void Qwen35VisionEncoder::release_runtime_buffers() {
    buf_.image_rgb = Tensor();
    buf_.patch_rows = Tensor();
    buf_.hidden_a = Tensor();
    buf_.hidden_b = Tensor();
    buf_.normed = Tensor();
    buf_.qkv = Tensor();
    buf_.q = Tensor();
    buf_.k = Tensor();
    buf_.v = Tensor();
    buf_.attn_out = Tensor();
    buf_.ffn = Tensor();
    buf_.merger_normed = Tensor();
    buf_.merger_hidden = Tensor();
    buf_.merger_out = Tensor();
    buf_.scores = Tensor();

    if (ctx_) {
        if (pos_y_device_) ctx_->free_device(pos_y_device_);
        if (pos_x_device_) ctx_->free_device(pos_x_device_);
    }
    pos_y_device_ = nullptr;
    pos_x_device_ = nullptr;
    pos_y_host_.clear();
    pos_x_host_.clear();

    image_bytes_capacity_ = 0;
    patch_capacity_ = 0;
    out_token_capacity_ = 0;
    pos_capacity_ = 0;
}

void Qwen35VisionEncoder::fill_merge_block_positions(int grid_w, int grid_h) {
    const int num_tokens = grid_w * grid_h;
    pos_y_host_.resize(static_cast<size_t>(num_tokens));
    pos_x_host_.resize(static_cast<size_t>(num_tokens));

    int out_idx = 0;
    for (int by = 0; by < grid_h; by += merge_size_) {
        for (int bx = 0; bx < grid_w; bx += merge_size_) {
            for (int dy = 0; dy < merge_size_; ++dy) {
                for (int dx = 0; dx < merge_size_; ++dx) {
                    pos_y_host_[static_cast<size_t>(out_idx)] = by + dy;
                    pos_x_host_[static_cast<size_t>(out_idx)] = bx + dx;
                    ++out_idx;
                }
            }
        }
    }
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

    parallel_for_1d(0, dst_h, 8, [&](int y) {
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
    });
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

bool Qwen35VisionEncoder::encode_rgb_image(const std::string& uri,
                                           int src_w,
                                           int src_h,
                                           const std::vector<uint8_t>& src_rgb,
                                           double decode_ms,
                                           bool profile,
                                           VisionEncodeResult& out,
                                           std::string* error_message) {
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();
    auto stage_start = t_total_start;
    const bool profile_blocks = profile && aila::env::read_flag("AILA_Q35_VISION_PROFILE_BLOCKS", false);
    const bool profile_prep = profile && aila::env::read_flag("AILA_Q35_VISION_PROFILE_PREP", false);
    auto stage_ms = [&](clock::time_point from) -> double {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };
    auto gpu_stage_ms = [&](clock::time_point from) -> double {
        if (profile && ctx_) ctx_->synchronize();
        return stage_ms(from);
    };

    out = VisionEncodeResult{};

    if (!loaded_ || !ctx_) {
        set_error(error_message, "Vision encoder is not loaded");
        return false;
    }

    if (src_w <= 0 || src_h <= 0 || src_rgb.empty()) {
        set_error(error_message, "Vision image is empty");
        return false;
    }

    const int align = std::max(1, patch_size_ * merge_size_);
    int target_w = align;
    int target_h = align;
    choose_target_size(src_w, src_h, align, min_pixels_, max_pixels_, target_w, target_h);

    auto merged_tokens = [&](int w, int h) -> int {
        const int g_w = std::max(1, w / align);
        const int g_h = std::max(1, h / align);
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
        const int desired = std::clamp(m_tok, min_tokens_, max_tokens_);
        const float scale = std::sqrt(static_cast<float>(desired) / static_cast<float>(std::max(1, m_tok)));
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

    stage_start = clock::now();
    std::vector<uint8_t> resized_rgb;
    resize_rgb_bicubic(src_rgb, src_w, src_h, target_w, target_h, resized_rgb);
    const double resize_ms = stage_ms(stage_start);

    const int patch_grid_w = target_w / patch_size_;
    const int patch_grid_h = target_h / patch_size_;
    const int num_patches = patch_grid_w * patch_grid_h;
    const int patch_dim = 3 * patch_size_ * patch_size_;
    const int merged_w = patch_grid_w / merge_size_;
    const int merged_h = patch_grid_h / merge_size_;
    const int out_tokens = merged_w * merged_h;
    const int merger_in_dim = hidden_size_ * merge_size_ * merge_size_;
    const int image_bytes = target_w * target_h * 3;

    if (num_patches <= 0) {
        set_error(error_message, "Invalid patch grid after resize");
        return false;
    }
    if (out_tokens <= 0) {
        set_error(error_message, "Invalid merged vision token count");
        return false;
    }

    ensure_runtime_buffers(image_bytes, num_patches, out_tokens, merger_in_dim);
    ensure_position_buffers(num_patches);

    Tensor image_rgb = Tensor::view(*ctx_, buf_.image_rgb.data(), {image_bytes}, dnnl::memory::data_type::u8);
    Tensor patch_rows = Tensor::view(*ctx_, buf_.patch_rows.data(), {num_patches, patch_dim}, dnnl::memory::data_type::bf16);
    Tensor tokens_a = Tensor::view(*ctx_, buf_.hidden_a.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor tokens_b = Tensor::view(*ctx_, buf_.hidden_b.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor normed = Tensor::view(*ctx_, buf_.normed.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor qkv = Tensor::view(*ctx_, buf_.qkv.data(), {num_patches, 3 * hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor q = Tensor::view(*ctx_, buf_.q.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor k = Tensor::view(*ctx_, buf_.k.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor v = Tensor::view(*ctx_, buf_.v.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor attn_out = Tensor::view(*ctx_, buf_.attn_out.data(), {num_patches, hidden_size_}, dnnl::memory::data_type::bf16);
    Tensor ffn = Tensor::view(*ctx_, buf_.ffn.data(), {num_patches, intermediate_size_}, dnnl::memory::data_type::bf16);
    Tensor scores = Tensor::view(*ctx_, buf_.scores.data(), {num_heads_, num_patches, num_patches}, dnnl::memory::data_type::f32);
    Tensor merger_normed = Tensor::view(*ctx_, buf_.merger_normed.data(), {out_tokens, merger_in_dim}, dnnl::memory::data_type::bf16);
    Tensor merger_hidden = Tensor::view(*ctx_, buf_.merger_hidden.data(), {out_tokens, merger_fc1_out_}, dnnl::memory::data_type::bf16);
    Tensor merger_out = Tensor::view(*ctx_, buf_.merger_out.data(), {out_tokens, out_hidden_size_}, dnnl::memory::data_type::bf16);

    stage_start = clock::now();
    ctx_->memcpy_h2d(image_rgb.data(), resized_rgb.data(), static_cast<size_t>(image_bytes));
    double prep_h2d_ms = 0.0;
    if (profile_prep) prep_h2d_ms = gpu_stage_ms(stage_start);

    if (profile_prep) stage_start = clock::now();
    ops::vision_patchify_rgb_u8(*ctx_, static_cast<const uint8_t*>(image_rgb.data()), patch_rows,
                                target_w, target_h, patch_size_,
                                image_mean_[0], image_mean_[1], image_mean_[2],
                                image_std_[0], image_std_[1], image_std_[2]);
    double prep_patchify_ms = 0.0;
    if (profile_prep) prep_patchify_ms = gpu_stage_ms(stage_start);
    const double prep_ms = profile_prep ? (prep_h2d_ms + prep_patchify_ms) : gpu_stage_ms(stage_start);

    stage_start = clock::now();
    patch_proj_.forward(*ctx_, patch_rows, tokens_a, num_patches);
    if (patch_bias_) {
        ops::bias_add_inplace(*ctx_, tokens_a, *patch_bias_, num_patches, hidden_size_);
    }
    const double patch_proj_ms = gpu_stage_ms(stage_start);

    stage_start = clock::now();
    ops::vision_add_position_embedding(*ctx_, tokens_a, *pos_embed_, patch_grid_w, patch_grid_h, hidden_size_);
    ops::vision_reorder_merge_blocks(*ctx_, tokens_a, tokens_b, patch_grid_w, patch_grid_h, hidden_size_, merge_size_);
    fill_merge_block_positions(patch_grid_w, patch_grid_h);
    const size_t pos_bytes = static_cast<size_t>(num_patches) * sizeof(int);
    ctx_->memcpy_h2d(pos_y_device_, pos_y_host_.data(), pos_bytes);
    ctx_->memcpy_h2d(pos_x_device_, pos_x_host_.data(), pos_bytes);
    const double pos_embed_ms = gpu_stage_ms(stage_start);

    const auto blocks_start = clock::now();
    Tensor* current = &tokens_b;
    Tensor* scratch = &tokens_a;
    double block_ln1_ms = 0.0;
    double block_qkv_ms = 0.0;
    double block_split_ms = 0.0;
    double block_rope_q_ms = 0.0;
    double block_rope_k_ms = 0.0;
    double block_attn_ms = 0.0;
    double block_proj_ms = 0.0;
    double block_resid1_ms = 0.0;
    double block_ln2_ms = 0.0;
    double block_fc1_ms = 0.0;
    double block_gelu_ms = 0.0;
    double block_fc2_ms = 0.0;
    double block_resid2_ms = 0.0;

    for (auto& b : blocks_) {
        if (profile_blocks) stage_start = clock::now();
        ops::layer_norm(*ctx_, *current, *b.ln1_weight, *b.ln1_bias, 1e-6f, normed, num_patches, hidden_size_);
        if (profile_blocks) block_ln1_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        b.qkv.forward(*ctx_, normed, qkv, num_patches);
        if (b.qkv_bias) {
            ops::bias_add_inplace(*ctx_, qkv, *b.qkv_bias, num_patches, 3 * hidden_size_);
        }
        if (profile_blocks) block_qkv_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::split_qkv(*ctx_, qkv, q, k, v, num_patches, hidden_size_, hidden_size_);
        if (profile_blocks) block_split_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::vision_mrope_inplace(*ctx_, q, num_patches, num_heads_, head_dim_, pos_y_device_, pos_x_device_);
        if (profile_blocks) block_rope_q_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::vision_mrope_inplace(*ctx_, k, num_patches, num_heads_, head_dim_, pos_y_device_, pos_x_device_);
        if (profile_blocks) block_rope_k_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::attention_bidi(*ctx_, q, k, v, attn_out, scores, num_patches, num_heads_, head_dim_);
        if (profile_blocks) block_attn_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        b.proj.forward(*ctx_, attn_out, *scratch, num_patches);
        if (b.proj_bias) {
            ops::bias_add_inplace(*ctx_, *scratch, *b.proj_bias, num_patches, hidden_size_);
        }
        if (profile_blocks) block_proj_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::residual_add(*ctx_, *current, *scratch, num_patches * hidden_size_);
        if (profile_blocks) block_resid1_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::layer_norm(*ctx_, *current, *b.ln2_weight, *b.ln2_bias, 1e-6f, normed, num_patches, hidden_size_);
        if (profile_blocks) block_ln2_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        b.fc1.forward(*ctx_, normed, ffn, num_patches);
        if (b.fc1_bias) {
            ops::bias_add_inplace(*ctx_, ffn, *b.fc1_bias, num_patches, intermediate_size_);
        }
        if (profile_blocks) block_fc1_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::gelu_tanh_inplace(*ctx_, ffn, num_patches * intermediate_size_);
        if (profile_blocks) block_gelu_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        b.fc2.forward(*ctx_, ffn, *scratch, num_patches);
        if (b.fc2_bias) {
            ops::bias_add_inplace(*ctx_, *scratch, *b.fc2_bias, num_patches, hidden_size_);
        }
        if (profile_blocks) block_fc2_ms += gpu_stage_ms(stage_start);

        if (profile_blocks) stage_start = clock::now();
        ops::residual_add(*ctx_, *current, *scratch, num_patches * hidden_size_);
        if (profile_blocks) block_resid2_ms += gpu_stage_ms(stage_start);
    }
    const double blocks_ms = gpu_stage_ms(blocks_start);

    stage_start = clock::now();
    Tensor* merged_source_tokens = current;
    if (merger_norm_weight_ && merger_norm_weight_->numel() == hidden_size_) {
        ops::layer_norm(*ctx_, *current, *merger_norm_weight_, *merger_norm_bias_,
                        1e-6f, *scratch, num_patches, hidden_size_);
        merged_source_tokens = scratch;
    }

    Tensor merged_input = Tensor::view(*ctx_, merged_source_tokens->data(),
                                       {out_tokens, merger_in_dim},
                                       dnnl::memory::data_type::bf16);
    Tensor* merger_src = &merged_input;
    if (merger_norm_weight_ && merger_norm_weight_->numel() == merger_in_dim) {
        ops::layer_norm(*ctx_, merged_input, *merger_norm_weight_, *merger_norm_bias_,
                        1e-6f, merger_normed, out_tokens, merger_in_dim);
        merger_src = &merger_normed;
    }

    merger_fc1_.forward(*ctx_, *merger_src, merger_hidden, out_tokens);
    if (merger_fc1_bias_) {
        ops::bias_add_inplace(*ctx_, merger_hidden, *merger_fc1_bias_, out_tokens, merger_fc1_out_);
    }
    ops::gelu_tanh_inplace(*ctx_, merger_hidden, out_tokens * merger_fc1_out_);

    merger_fc2_.forward(*ctx_, merger_hidden, merger_out, out_tokens);
    if (merger_fc2_bias_) {
        ops::bias_add_inplace(*ctx_, merger_out, *merger_fc2_bias_, out_tokens, out_hidden_size_);
    }

    out.token_count = out_tokens;
    out.llm_grid_t = 1;
    out.llm_grid_h = merged_h;
    out.llm_grid_w = merged_w;
    out.embeddings.resize(static_cast<size_t>(out_tokens) * static_cast<size_t>(out_hidden_size_));
    ctx_->memcpy_d2h(out.embeddings.data(), merger_out.data(), out.embeddings.size() * sizeof(bf16));
    const double merger_ms = stage_ms(stage_start);

    if (profile) {
        const double total_ms = stage_ms(t_total_start);
        AILA_LOG_INFO(
            "[VisionProfile] file=%s src=%dx%d target=%dx%d patch=%dx%d merged=%dx%d llm=%dx%dx%d tokens=%d "
            "decode=%.2fms resize=%.2fms prep=%.2fms patch=%.2fms pos=%.2fms blocks=%.2fms merger=%.2fms total=%.2fms",
            uri.c_str(), src_w, src_h, target_w, target_h,
            patch_grid_w, patch_grid_h, merged_w, merged_h,
            out.llm_grid_t, out.llm_grid_h, out.llm_grid_w, out.token_count,
            decode_ms, resize_ms, prep_ms, patch_proj_ms, pos_embed_ms, blocks_ms, merger_ms, total_ms);

        if (profile_prep) {
            AILA_LOG_INFO(
                "[VisionPrepProfile] file=%s bytes=%d target=%dx%d patches=%d h2d=%.2fms patchify=%.2fms sum=%.2fms",
                uri.c_str(), image_bytes, target_w, target_h, num_patches,
                prep_h2d_ms, prep_patchify_ms, prep_ms);
        }

        if (profile_blocks) {
            const double attn_stack_ms = block_ln1_ms + block_qkv_ms + block_split_ms +
                                         block_rope_q_ms + block_rope_k_ms + block_attn_ms +
                                         block_proj_ms + block_resid1_ms;
            const double mlp_stack_ms = block_ln2_ms + block_fc1_ms + block_gelu_ms +
                                        block_fc2_ms + block_resid2_ms;
            const double block_sum_ms = attn_stack_ms + mlp_stack_ms;
            const double avg_block_ms = depth_ > 0 ? (block_sum_ms / static_cast<double>(depth_)) : 0.0;
            AILA_LOG_INFO(
                "[VisionBlocksProfile] file=%s layers=%d tokens=%d heads=%d head_dim=%d "
                "ln1=%.2fms qkv=%.2fms split=%.2fms rope_q=%.2fms rope_k=%.2fms attn=%.2fms "
                "proj=%.2fms resid1=%.2fms ln2=%.2fms fc1=%.2fms gelu=%.2fms fc2=%.2fms resid2=%.2fms "
                "attn_stack=%.2fms mlp_stack=%.2fms sum=%.2fms avg_block=%.2fms",
                uri.c_str(), depth_, num_patches, num_heads_, head_dim_,
                block_ln1_ms, block_qkv_ms, block_split_ms, block_rope_q_ms, block_rope_k_ms, block_attn_ms,
                block_proj_ms, block_resid1_ms, block_ln2_ms, block_fc1_ms, block_gelu_ms, block_fc2_ms, block_resid2_ms,
                attn_stack_ms, mlp_stack_ms, block_sum_ms, avg_block_ms);
        }
    }
    return true;
}

bool Qwen35VisionEncoder::encode_image(const std::string& uri,
                                       VisionEncodeResult& out,
                                       std::string* error_message) {
    using clock = std::chrono::high_resolution_clock;
    auto decode_start = clock::now();

    int src_w = 0;
    int src_h = 0;
    std::vector<uint8_t> src_rgb;
    if (!read_image_rgb(uri, src_w, src_h, src_rgb, error_message)) {
        return false;
    }

    const double decode_ms = std::chrono::duration<double, std::milli>(clock::now() - decode_start).count();
    const bool profile = aila::env::read_flag("AILA_Q35_VISION_PROFILE", false);
    return encode_rgb_image(uri, src_w, src_h, src_rgb, decode_ms, profile, out, error_message);
}

bool Qwen35VisionEncoder::warmup(std::string* error_message) {
    if (!loaded_ || !ctx_) {
        set_error(error_message, "Vision encoder is not loaded");
        return false;
    }

    constexpr int warmup_size = 256;
    std::vector<uint8_t> warmup_rgb(static_cast<size_t>(warmup_size) * warmup_size * 3);
    parallel_for_1d(0, warmup_size * warmup_size, 2048, [&](int i) {
        const int x = i % warmup_size;
        const int y = i / warmup_size;
        const size_t base = static_cast<size_t>(i) * 3;
        warmup_rgb[base + 0] = static_cast<uint8_t>(x);
        warmup_rgb[base + 1] = static_cast<uint8_t>(y);
        warmup_rgb[base + 2] = static_cast<uint8_t>((x + y) & 255);
    });

    VisionEncodeResult warmup_out;
    return encode_rgb_image("<vision-warmup>", warmup_size, warmup_size, warmup_rgb,
                            0.0, false, warmup_out, error_message);
}

} // namespace vision
} // namespace aila
