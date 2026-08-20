#include "Yolo26Detector.hpp"

#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#include <wincodec.h>
#endif

namespace aila::vision {
namespace {

using f16 = sycl::half;

void dump_irtensor(Context& ctx, const std::string& name, const Tensor& tensor) {
    static const std::string directory = aila::env::read_string("AILA_YOLO_DUMP_DIR", "");
    if (directory.empty()) return;
    uint16_t dtype = 0;
    if (tensor.dtype() == dnnl::memory::data_type::f16) dtype = 10;
    else if (tensor.dtype() == dnnl::memory::data_type::f32) dtype = 12;
    else return;
    ctx.synchronize();
    std::vector<uint8_t> payload(tensor.size_bytes());
    ctx.memcpy_d2h(payload.data(), tensor.data(), payload.size());
    const uint16_t rank = static_cast<uint16_t>(tensor.ndim());
    const uint16_t header_size = static_cast<uint16_t>(48 + 16 * rank);
    const uint16_t version = 1, reserved = 0;
    const uint32_t flags = 1;
    const uint64_t numel = tensor.numel(), payload_bytes = payload.size();
    std::vector<uint8_t> header(header_size, 0);
    auto put = [&](size_t offset, const void* source, size_t count) {
        std::memcpy(header.data() + offset, source, count);
    };
    put(0, "IRTN", 4); put(4, &version, 2); put(6, &header_size, 2);
    put(8, &dtype, 2); put(10, &flags, 4); put(14, &rank, 2); put(16, &reserved, 2);
    put(18, &numel, 8); put(26, &payload_bytes, 8);
    int64_t stride = 1;
    std::vector<int64_t> strides(rank);
    for (int index = rank - 1; index >= 0; --index) {
        strides[static_cast<size_t>(index)] = stride;
        stride *= tensor.shape(static_cast<size_t>(index));
    }
    for (uint16_t index = 0; index < rank; ++index) {
        const int64_t dimension = tensor.shape(index);
        put(34 + 8 * index, &dimension, 8);
        put(34 + 8 * rank + 8 * index, &strides[index], 8);
    }
    const int64_t storage_offset = 0;
    put(34 + 16 * rank, &storage_offset, 8);
    const std::filesystem::path target = std::filesystem::path(directory) / (name + ".irtensor");
    std::error_code ignored;
    std::filesystem::create_directories(target.parent_path(), ignored);
    std::ofstream output(target, std::ios::binary);
    output.write(reinterpret_cast<const char*>(header.data()), header.size());
    output.write(reinterpret_cast<const char*>(payload.data()), payload.size());
}

void set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

int checked_int64(int64_t value, const char* what) {
    if (value < 0 || value > (std::numeric_limits<int>::max)()) {
        throw std::runtime_error(std::string("YOLO26 invalid ") + what);
    }
    return static_cast<int>(value);
}

Tensor& weight(ModelWeights& weights, const std::string& name) {
    if (!weights.has(name)) throw std::runtime_error("YOLO26 weight not found: " + name);
    Tensor& value = weights.get(name);
    if (value.dtype() != dnnl::memory::data_type::f16) {
        throw std::runtime_error("YOLO26 weight must be FP16: " + name);
    }
    return value;
}

void copy_tensor(Context& ctx, const Tensor& source, Tensor& destination) {
    if (source.numel() != destination.numel() || source.dtype() != destination.dtype()) {
        throw std::runtime_error("YOLO26 copy tensor mismatch");
    }
    ctx.queue().memcpy(destination.data(), source.data(), source.size_bytes());
}

void add_tensor(Context& ctx, const Tensor& left, const Tensor& right, Tensor& output) {
    if (left.numel() != right.numel() || left.numel() != output.numel()) {
        throw std::runtime_error("YOLO26 add tensor mismatch");
    }
    const f16* a = static_cast<const f16*>(left.data());
    const f16* b = static_cast<const f16*>(right.data());
    f16* out = output.data_as<f16>();
    const size_t count = static_cast<size_t>(output.numel());
    ctx.queue().parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
        const size_t i = index[0];
        out[i] = f16(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    });
}

void concat_channels(Context& ctx, const std::vector<const Tensor*>& inputs, Tensor& output) {
    if (inputs.empty() || output.ndim() != 4 || output.shape(0) != 1) {
        throw std::runtime_error("YOLO26 concat expects non-empty batch-1 NCHW tensors");
    }
    size_t offset = 0;
    for (const Tensor* input : inputs) {
        if (!input || input->ndim() != 4 || input->shape(0) != 1 ||
            input->shape(2) != output.shape(2) || input->shape(3) != output.shape(3)) {
            throw std::runtime_error("YOLO26 concat shape mismatch");
        }
        ctx.queue().memcpy(static_cast<uint8_t*>(output.data()) + offset,
                           input->data(), input->size_bytes());
        offset += input->size_bytes();
    }
    if (offset != output.size_bytes()) throw std::runtime_error("YOLO26 concat channel mismatch");
}

void upsample_nearest(Context& ctx, const Tensor& input, Tensor& output) {
    const int channels = checked_int64(input.shape(1), "upsample channels");
    const int in_h = checked_int64(input.shape(2), "upsample height");
    const int in_w = checked_int64(input.shape(3), "upsample width");
    const int out_h = checked_int64(output.shape(2), "upsample output height");
    const int out_w = checked_int64(output.shape(3), "upsample output width");
    if (out_h != in_h * 2 || out_w != in_w * 2 || output.shape(1) != input.shape(1)) {
        throw std::runtime_error("YOLO26 nearest upsample shape mismatch");
    }
    const f16* src = static_cast<const f16*>(input.data());
    f16* dst = output.data_as<f16>();
    const size_t total = static_cast<size_t>(channels) * out_h * out_w;
    ctx.queue().parallel_for(sycl::range<1>(total), [=](sycl::id<1> index) {
        const size_t i = index[0];
        const int x = static_cast<int>(i % out_w);
        const int y = static_cast<int>((i / out_w) % out_h);
        const int c = static_cast<int>(i / (static_cast<size_t>(out_w) * out_h));
        dst[i] = src[(static_cast<size_t>(c) * in_h + y / 2) * in_w + x / 2];
    });
}

class ConvLayer {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              Tensor& input, int out_channels, int kernel, int stride,
              int groups, bool silu, bool direct = false) {
        input_ = &input;
        const int in_channels = checked_int64(input.shape(1), "convolution input channels");
        const int in_h = checked_int64(input.shape(2), "convolution input height");
        const int in_w = checked_int64(input.shape(3), "convolution input width");
        const int pad = kernel / 2;
        const int out_h = (in_h + 2 * pad - kernel) / stride + 1;
        const int out_w = (in_w + 2 * pad - kernel) / stride + 1;
        output_ = Tensor::allocate(ctx, {1, out_channels, out_h, out_w}, dnnl::memory::data_type::f16);

        const std::string stem = direct ? prefix : prefix + ".conv";
        weight_ = &weight(weights, stem + ".weight");
        bias_ = &weight(weights, stem + ".bias");
        if (weight_->ndim() != 4 || weight_->shape(0) != out_channels ||
            weight_->shape(1) != in_channels / groups || weight_->shape(2) != kernel ||
            weight_->shape(3) != kernel || bias_->numel() != out_channels) {
            throw std::runtime_error("YOLO26 convolution shape mismatch: " + prefix);
        }

        src_md_ = dnnl::memory::desc({1, in_channels, in_h, in_w},
                                     dnnl::memory::data_type::f16,
                                     dnnl::memory::format_tag::nchw);
        dst_md_ = dnnl::memory::desc({1, out_channels, out_h, out_w},
                                     dnnl::memory::data_type::f16,
                                     dnnl::memory::format_tag::nchw);
        if (groups == 1) {
            weights_md_ = dnnl::memory::desc({out_channels, in_channels, kernel, kernel},
                                             dnnl::memory::data_type::f16,
                                             dnnl::memory::format_tag::oihw);
        } else {
            weights_md_ = dnnl::memory::desc({groups, out_channels / groups,
                                              in_channels / groups, kernel, kernel},
                                             dnnl::memory::data_type::f16,
                                             dnnl::memory::format_tag::goihw);
        }
        bias_md_ = dnnl::memory::desc({out_channels}, dnnl::memory::data_type::f16,
                                      dnnl::memory::format_tag::a);

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        if (silu) {
            dnnl::post_ops post_ops;
            post_ops.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
            attr.set_post_ops(post_ops);
        }
        dnnl::memory::dims strides{stride, stride};
        dnnl::memory::dims dilates{0, 0};
        dnnl::memory::dims padding{pad, pad};
        auto pd = dnnl::convolution_forward::primitive_desc(
            ctx.engine(), dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_direct, src_md_, weights_md_, bias_md_, dst_md_,
            strides, dilates, padding, padding, attr);
        AILA_LOG_DEBUG("[YOLO26][oneDNN] %s: %s", prefix.c_str(), pd.impl_info_str());
        primitive_ = dnnl::convolution_forward(pd);
        const size_t scratch_bytes = pd.scratchpad_desc().get_size();
        if (scratch_bytes) {
            scratch_ = Tensor::allocate(ctx, {static_cast<int64_t>(scratch_bytes)},
                                        dnnl::memory::data_type::u8);
            scratch_mem_ = scratch_.make_dnnl_memory(pd.scratchpad_desc());
        }
        src_mem_ = input.make_dnnl_memory(src_md_);
        weights_mem_ = weight_->make_dnnl_memory(weights_md_);
        bias_mem_ = bias_->make_dnnl_memory(bias_md_);
        dst_mem_ = output_.make_dnnl_memory(dst_md_);
        args_ = {{DNNL_ARG_SRC, src_mem_}, {DNNL_ARG_WEIGHTS, weights_mem_},
                 {DNNL_ARG_BIAS, bias_mem_}, {DNNL_ARG_DST, dst_mem_}};
        if (scratch_.valid()) args_.emplace(DNNL_ARG_SCRATCHPAD, scratch_mem_);
    }

    void forward(Context& ctx) { primitive_.execute(ctx.stream(), args_); }
    Tensor& output() { return output_; }

private:
    Tensor* input_ = nullptr;
    Tensor* weight_ = nullptr;
    Tensor* bias_ = nullptr;
    Tensor output_;
    Tensor scratch_;
    dnnl::memory::desc src_md_, weights_md_, bias_md_, dst_md_;
    dnnl::memory src_mem_, weights_mem_, bias_mem_, dst_mem_, scratch_mem_;
    dnnl::convolution_forward primitive_;
    std::unordered_map<int, dnnl::memory> args_;
};

class MaxPoolLayer {
public:
    void init(Context& ctx, Tensor& input, int kernel) {
        input_ = &input;
        output_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        auto md = dnnl::memory::desc(input.shape(), dnnl::memory::data_type::f16,
                                    dnnl::memory::format_tag::nchw);
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const int pad = kernel / 2;
        auto pd = dnnl::pooling_forward::primitive_desc(
            ctx.engine(), dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_max, md, md, {1, 1}, {kernel, kernel},
            {0, 0}, {pad, pad}, {pad, pad}, attr);
        primitive_ = dnnl::pooling_forward(pd);
        const size_t scratch_bytes = pd.scratchpad_desc().get_size();
        if (scratch_bytes) {
            scratch_ = Tensor::allocate(ctx, {static_cast<int64_t>(scratch_bytes)},
                                        dnnl::memory::data_type::u8);
            scratch_mem_ = scratch_.make_dnnl_memory(pd.scratchpad_desc());
        }
        src_mem_ = input.make_dnnl_memory(md);
        dst_mem_ = output_.make_dnnl_memory(md);
        args_ = {{DNNL_ARG_SRC, src_mem_}, {DNNL_ARG_DST, dst_mem_}};
        if (scratch_.valid()) args_.emplace(DNNL_ARG_SCRATCHPAD, scratch_mem_);
    }
    void forward(Context& ctx) { primitive_.execute(ctx.stream(), args_); }
    Tensor& output() { return output_; }
private:
    Tensor* input_ = nullptr;
    Tensor output_, scratch_;
    dnnl::memory src_mem_, dst_mem_, scratch_mem_;
    dnnl::pooling_forward primitive_;
    std::unordered_map<int, dnnl::memory> args_;
};

class Bottleneck {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              Tensor& input, bool shortcut) {
        input_ = &input;
        const int channels = checked_int64(input.shape(1), "bottleneck channels");
        const int first_channels = checked_int64(
            weight(weights, prefix + ".cv1.conv.weight").shape(0), "bottleneck hidden channels");
        const int output_channels = checked_int64(
            weight(weights, prefix + ".cv2.conv.weight").shape(0), "bottleneck output channels");
        cv1_.init(ctx, weights, prefix + ".cv1", input, first_channels, 3, 1, 1, true);
        cv2_.init(ctx, weights, prefix + ".cv2", cv1_.output(), output_channels, 3, 1, 1, true);
        add_ = shortcut && channels == output_channels;
        if (add_) output_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
    }
    void forward(Context& ctx) {
        cv1_.forward(ctx); cv2_.forward(ctx);
        if (add_) add_tensor(ctx, *input_, cv2_.output(), output_);
    }
    Tensor& output() { return add_ ? output_ : cv2_.output(); }
private:
    Tensor* input_ = nullptr;
    ConvLayer cv1_, cv2_;
    Tensor output_;
    bool add_ = false;
};

class C3k {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              Tensor& input, bool shortcut) {
        const int channels = checked_int64(input.shape(1), "C3k channels");
        const int hidden = channels / 2;
        cv1_.init(ctx, weights, prefix + ".cv1", input, hidden, 1, 1, 1, true);
        cv2_.init(ctx, weights, prefix + ".cv2", input, hidden, 1, 1, 1, true);
        blocks_.resize(2);
        blocks_[0].init(ctx, weights, prefix + ".m.0", cv1_.output(), shortcut);
        blocks_[1].init(ctx, weights, prefix + ".m.1", blocks_[0].output(), shortcut);
        concat_ = Tensor::allocate(ctx, {1, 2 * hidden, input.shape(2), input.shape(3)},
                                   dnnl::memory::data_type::f16);
        cv3_.init(ctx, weights, prefix + ".cv3", concat_, channels, 1, 1, 1, true);
    }
    void forward(Context& ctx) {
        cv1_.forward(ctx); cv2_.forward(ctx);
        for (auto& block : blocks_) block.forward(ctx);
        concat_channels(ctx, {&blocks_.back().output(), &cv2_.output()}, concat_);
        cv3_.forward(ctx);
    }
    Tensor& output() { return cv3_.output(); }
private:
    ConvLayer cv1_, cv2_, cv3_;
    std::vector<Bottleneck> blocks_;
    Tensor concat_;
};

class Attention {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix, Tensor& input) {
        input_ = &input;
        channels_ = checked_int64(input.shape(1), "attention channels");
        height_ = checked_int64(input.shape(2), "attention height");
        width_ = checked_int64(input.shape(3), "attention width");
        tokens_ = height_ * width_;
        heads_ = (std::max)(channels_ / 64, 1);
        head_dim_ = channels_ / heads_;
        key_dim_ = head_dim_ / 2;
        const int qkv_channels = channels_ + 2 * heads_ * key_dim_;
        qkv_.init(ctx, weights, prefix + ".qkv", input, qkv_channels, 1, 1, 1, false);
        q_packed_ = Tensor::allocate(ctx, {heads_, tokens_, key_dim_}, dnnl::memory::data_type::f16);
        k_packed_ = Tensor::allocate(ctx, {heads_, key_dim_, tokens_}, dnnl::memory::data_type::f16);
        scores_ = Tensor::allocate(ctx, {heads_, tokens_, tokens_}, dnnl::memory::data_type::f32);
        context_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        v_packed_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        pe_.init(ctx, weights, prefix + ".pe", v_packed_, channels_, 3, 1,
                 channels_, false);
        sum_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        proj_.init(ctx, weights, prefix + ".proj", sum_, channels_, 1, 1, 1, false);

        const auto q_md = dnnl::memory::desc(
            {heads_, tokens_, key_dim_}, dnnl::memory::data_type::f16,
            dnnl::memory::format_tag::abc);
        const auto k_md = dnnl::memory::desc(
            {heads_, key_dim_, tokens_}, dnnl::memory::data_type::f16,
            dnnl::memory::format_tag::abc);
        const auto scores_md = dnnl::memory::desc(
            {heads_, tokens_, tokens_}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::abc);
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        const auto pd = dnnl::matmul::primitive_desc(ctx.engine(), q_md, k_md, scores_md, attr);
        AILA_LOG_DEBUG("[YOLO26][oneDNN] %s.qk: %s", prefix.c_str(), pd.impl_info_str());
        qk_primitive_ = dnnl::matmul(pd);
        q_mem_ = q_packed_.make_dnnl_memory(q_md);
        k_mem_ = k_packed_.make_dnnl_memory(k_md);
        scores_mem_ = scores_.make_dnnl_memory(scores_md);
        const size_t scratch_bytes = pd.scratchpad_desc().get_size();
        if (scratch_bytes) {
            qk_scratch_ = Tensor::allocate(ctx, {static_cast<int64_t>(scratch_bytes)},
                                           dnnl::memory::data_type::u8);
            qk_scratch_mem_ = qk_scratch_.make_dnnl_memory(pd.scratchpad_desc());
        }
        qk_args_ = {{DNNL_ARG_SRC, q_mem_}, {DNNL_ARG_WEIGHTS, k_mem_},
                    {DNNL_ARG_DST, scores_mem_}};
        if (qk_scratch_.valid()) qk_args_.emplace(DNNL_ARG_SCRATCHPAD, qk_scratch_mem_);
    }
    void forward(Context& ctx) {
        qkv_.forward(ctx);
        const f16* qkv = qkv_.output().data_as<f16>();
        f16* packed_q = q_packed_.data_as<f16>();
        f16* packed_k = k_packed_.data_as<f16>();
        f16* packed_v = v_packed_.data_as<f16>();
        float* scores = scores_.data_as<float>();
        const int heads = heads_, kd = key_dim_, hd = head_dim_, n = tokens_;
        const int span = kd * 2 + hd;
        const float scale = 1.0f / std::sqrt(static_cast<float>(kd));
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(heads) * n * (2 * kd + hd)),
            [=](sycl::id<1> index) {
                const size_t linear = index[0];
                const int d = static_cast<int>(linear % (2 * kd + hd));
                const int token = static_cast<int>((linear / (2 * kd + hd)) % n);
                const int head = static_cast<int>(linear /
                    (static_cast<size_t>(n) * (2 * kd + hd)));
                if (d < kd) {
                    packed_q[(static_cast<size_t>(head) * n + token) * kd + d] =
                        qkv[(head * span + d) * n + token];
                } else if (d < 2 * kd) {
                    const int kdim = d - kd;
                    packed_k[(static_cast<size_t>(head) * kd + kdim) * n + token] =
                        qkv[(head * span + kd + kdim) * n + token];
                } else {
                    const int vdim = d - 2 * kd;
                    packed_v[(static_cast<size_t>(head) * hd + vdim) * n + token] =
                        qkv[(head * span + 2 * kd + vdim) * n + token];
                }
            });
        qk_primitive_.execute(ctx.stream(), qk_args_);
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(heads) * n),
            [=](sycl::id<1> index) {
                const size_t row = index[0];
                float* values = scores + row * n;
                float maximum = -3.402823466e+38F;
                for (int i = 0; i < n; ++i) {
                    values[i] *= scale;
                    maximum = sycl::fmax(maximum, values[i]);
                }
                float total = 0.0f;
                for (int i = 0; i < n; ++i) { values[i] = sycl::exp(values[i] - maximum); total += values[i]; }
                const float inverse = 1.0f / total;
                for (int i = 0; i < n; ++i) values[i] *= inverse;
            });
        f16* context = context_.data_as<f16>();
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(heads) * hd * n),
            [=](sycl::id<1> index) {
                const size_t linear = index[0];
                const int query = static_cast<int>(linear % n);
                const int d = static_cast<int>((linear / n) % hd);
                const int head = static_cast<int>(linear / (static_cast<size_t>(n) * hd));
                const int qkv_channel = head * span + 2 * kd + d;
                const int out_channel = head * hd + d;
                float value = 0.0f;
                for (int key = 0; key < n; ++key) {
                    value += static_cast<float>(qkv[qkv_channel * n + key]) *
                             scores[(static_cast<size_t>(head) * n + query) * n + key];
                }
                context[out_channel * n + query] = f16(value);
            });
        pe_.forward(ctx);
        add_tensor(ctx, context_, pe_.output(), sum_);
        proj_.forward(ctx);
    }
    Tensor& output() { return proj_.output(); }
private:
    Tensor* input_ = nullptr;
    int channels_ = 0, height_ = 0, width_ = 0, tokens_ = 0;
    int heads_ = 0, head_dim_ = 0, key_dim_ = 0;
    ConvLayer qkv_, pe_, proj_;
    Tensor q_packed_, k_packed_, scores_, context_, sum_, v_packed_, qk_scratch_;
    dnnl::matmul qk_primitive_;
    dnnl::memory q_mem_, k_mem_, scores_mem_, qk_scratch_mem_;
    std::unordered_map<int, dnnl::memory> qk_args_;
};

class PSABlock {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix, Tensor& input) {
        input_ = &input;
        attention_.init(ctx, weights, prefix + ".attn", input);
        attn_sum_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        const int channels = checked_int64(input.shape(1), "PSA channels");
        ffn1_.init(ctx, weights, prefix + ".ffn.0", attn_sum_, 2 * channels, 1, 1, 1, true);
        ffn2_.init(ctx, weights, prefix + ".ffn.1", ffn1_.output(), channels, 1, 1, 1, false);
        output_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
    }
    void forward(Context& ctx) {
        attention_.forward(ctx);
        add_tensor(ctx, *input_, attention_.output(), attn_sum_);
        ffn1_.forward(ctx); ffn2_.forward(ctx);
        add_tensor(ctx, attn_sum_, ffn2_.output(), output_);
    }
    Tensor& output() { return output_; }
private:
    Tensor* input_ = nullptr;
    Attention attention_;
    ConvLayer ffn1_, ffn2_;
    Tensor attn_sum_, output_;
};

class C2PSA {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              Tensor& input, int repeats) {
        const int channels = checked_int64(input.shape(1), "C2PSA channels");
        hidden_ = channels / 2;
        cv1_.init(ctx, weights, prefix + ".cv1", input, channels, 1, 1, 1, true);
        a_ = Tensor::view(ctx, cv1_.output().data(), {1, hidden_, input.shape(2), input.shape(3)},
                          dnnl::memory::data_type::f16);
        b_ = Tensor::view(ctx, static_cast<f16*>(cv1_.output().data()) +
                         static_cast<size_t>(hidden_) * input.shape(2) * input.shape(3),
                         {1, hidden_, input.shape(2), input.shape(3)}, dnnl::memory::data_type::f16);
        blocks_.resize(static_cast<size_t>(repeats));
        Tensor* current = &b_;
        for (int i = 0; i < repeats; ++i) {
            blocks_[static_cast<size_t>(i)].init(ctx, weights, prefix + ".m." + std::to_string(i), *current);
            current = &blocks_[static_cast<size_t>(i)].output();
        }
        concat_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
        cv2_.init(ctx, weights, prefix + ".cv2", concat_, channels, 1, 1, 1, true);
    }
    void forward(Context& ctx) {
        cv1_.forward(ctx);
        for (auto& block : blocks_) block.forward(ctx);
        Tensor& tail = blocks_.empty() ? b_ : blocks_.back().output();
        concat_channels(ctx, {&a_, &tail}, concat_);
        cv2_.forward(ctx);
    }
    Tensor& output() { return cv2_.output(); }
private:
    int hidden_ = 0;
    ConvLayer cv1_, cv2_;
    Tensor a_, b_, concat_;
    std::vector<PSABlock> blocks_;
};

class C3k2 {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              Tensor& input, int out_channels, int repeats, bool c3k,
              float expansion, bool attention = false) {
        hidden_ = static_cast<int>(out_channels * expansion);
        cv1_.init(ctx, weights, prefix + ".cv1", input, 2 * hidden_, 1, 1, 1, true);
        const int64_t plane = static_cast<int64_t>(hidden_) * input.shape(2) * input.shape(3);
        first_ = Tensor::view(ctx, cv1_.output().data(), {1, hidden_, input.shape(2), input.shape(3)},
                              dnnl::memory::data_type::f16);
        second_ = Tensor::view(ctx, static_cast<f16*>(cv1_.output().data()) + plane,
                               {1, hidden_, input.shape(2), input.shape(3)}, dnnl::memory::data_type::f16);
        kinds_c3k_ = c3k;
        attention_ = attention;
        Tensor* current = &second_;
        if (attention) {
            sequence_bottlenecks_.resize(static_cast<size_t>(repeats));
            sequence_psa_.resize(static_cast<size_t>(repeats));
            for (int i = 0; i < repeats; ++i) {
                const std::string base = prefix + ".m." + std::to_string(i);
                sequence_bottlenecks_[i].init(ctx, weights, base + ".0", *current, true);
                sequence_psa_[i].init(ctx, weights, base + ".1", sequence_bottlenecks_[i].output());
                current = &sequence_psa_[i].output();
            }
        } else if (c3k) {
            c3k_blocks_.resize(static_cast<size_t>(repeats));
            for (int i = 0; i < repeats; ++i) {
                c3k_blocks_[i].init(ctx, weights, prefix + ".m." + std::to_string(i), *current, true);
                current = &c3k_blocks_[i].output();
            }
        } else {
            bottlenecks_.resize(static_cast<size_t>(repeats));
            for (int i = 0; i < repeats; ++i) {
                bottlenecks_[i].init(ctx, weights, prefix + ".m." + std::to_string(i), *current, true);
                current = &bottlenecks_[i].output();
            }
        }
        concat_ = Tensor::allocate(ctx, {1, static_cast<int64_t>((2 + repeats) * hidden_),
                                         input.shape(2), input.shape(3)}, dnnl::memory::data_type::f16);
        cv2_.init(ctx, weights, prefix + ".cv2", concat_, out_channels, 1, 1, 1, true);
    }
    void forward(Context& ctx) {
        cv1_.forward(ctx);
        std::vector<const Tensor*> values{&first_, &second_};
        if (attention_) {
            for (size_t i = 0; i < sequence_bottlenecks_.size(); ++i) {
                sequence_bottlenecks_[i].forward(ctx); sequence_psa_[i].forward(ctx);
                values.push_back(&sequence_psa_[i].output());
            }
        } else if (kinds_c3k_) {
            for (auto& block : c3k_blocks_) { block.forward(ctx); values.push_back(&block.output()); }
        } else {
            for (auto& block : bottlenecks_) { block.forward(ctx); values.push_back(&block.output()); }
        }
        concat_channels(ctx, values, concat_);
        cv2_.forward(ctx);
    }
    Tensor& output() { return cv2_.output(); }
private:
    int hidden_ = 0;
    bool kinds_c3k_ = false, attention_ = false;
    ConvLayer cv1_, cv2_;
    Tensor first_, second_, concat_;
    std::vector<Bottleneck> bottlenecks_, sequence_bottlenecks_;
    std::vector<C3k> c3k_blocks_;
    std::vector<PSABlock> sequence_psa_;
};

class SPPF {
public:
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix, Tensor& input) {
        input_ = &input;
        const int channels = checked_int64(input.shape(1), "SPPF channels");
        cv1_.init(ctx, weights, prefix + ".cv1", input, channels / 2, 1, 1, 1, false);
        pools_.resize(3);
        pools_[0].init(ctx, cv1_.output(), 5);
        pools_[1].init(ctx, pools_[0].output(), 5);
        pools_[2].init(ctx, pools_[1].output(), 5);
        concat_ = Tensor::allocate(ctx, {1, 2 * channels, input.shape(2), input.shape(3)},
                                   dnnl::memory::data_type::f16);
        cv2_.init(ctx, weights, prefix + ".cv2", concat_, channels, 1, 1, 1, true);
        output_ = Tensor::allocate(ctx, input.shape(), dnnl::memory::data_type::f16);
    }
    void forward(Context& ctx) {
        cv1_.forward(ctx);
        dump_irtensor(ctx, "layer9_cv1", cv1_.output());
        for (size_t index = 0; index < pools_.size(); ++index) {
            pools_[index].forward(ctx);
            dump_irtensor(ctx, "layer9_pool" + std::to_string(index), pools_[index].output());
        }
        concat_channels(ctx, {&cv1_.output(), &pools_[0].output(), &pools_[1].output(), &pools_[2].output()}, concat_);
        dump_irtensor(ctx, "layer9_concat", concat_);
        cv2_.forward(ctx);
        add_tensor(ctx, *input_, cv2_.output(), output_);
    }
    Tensor& output() { return output_; }
private:
    Tensor* input_ = nullptr;
    ConvLayer cv1_, cv2_;
    std::vector<MaxPoolLayer> pools_;
    Tensor concat_, output_;
};

struct DetectBranch {
    ConvLayer box0, box1, box2;
    ConvLayer cls_dw0, cls_pw0, cls_dw1, cls_pw1, cls_out;
    void init(Context& ctx, ModelWeights& weights, const std::string& prefix,
              int index, Tensor& input, int box_channels, int cls_channels, int nc) {
        const std::string i = std::to_string(index);
        box0.init(ctx, weights, prefix + ".box." + i + ".0", input, box_channels, 3, 1, 1, true);
        box1.init(ctx, weights, prefix + ".box." + i + ".1", box0.output(), box_channels, 3, 1, 1, true);
        box2.init(ctx, weights, prefix + ".box." + i + ".2", box1.output(), 4, 1, 1, 1, false, true);
        const int in_channels = checked_int64(input.shape(1), "Detect input channels");
        cls_dw0.init(ctx, weights, prefix + ".cls." + i + ".0.0", input, in_channels, 3, 1,
                     in_channels, true);
        cls_pw0.init(ctx, weights, prefix + ".cls." + i + ".0.1", cls_dw0.output(), cls_channels,
                     1, 1, 1, true);
        cls_dw1.init(ctx, weights, prefix + ".cls." + i + ".1.0", cls_pw0.output(), cls_channels,
                     3, 1, cls_channels, true);
        cls_pw1.init(ctx, weights, prefix + ".cls." + i + ".1.1", cls_dw1.output(), cls_channels,
                     1, 1, 1, true);
        cls_out.init(ctx, weights, prefix + ".cls." + i + ".2", cls_pw1.output(), nc,
                     1, 1, 1, false, true);
    }
    void forward(Context& ctx) {
        box0.forward(ctx); box1.forward(ctx); box2.forward(ctx);
        cls_dw0.forward(ctx); cls_pw0.forward(ctx); cls_dw1.forward(ctx); cls_pw1.forward(ctx); cls_out.forward(ctx);
    }
};

struct ScaleSpec { float depth; float width; int max_channels; };

ScaleSpec scale_spec(const std::string& scale) {
    if (scale == "n") return {0.5f, 0.25f, 1024};
    if (scale == "s") return {0.5f, 0.5f, 1024};
    if (scale == "m") return {0.5f, 1.0f, 512};
    if (scale == "l") return {1.0f, 1.0f, 512};
    if (scale == "x") return {1.0f, 1.5f, 512};
    throw std::runtime_error("unsupported YOLO26 scale: " + scale);
}

int scaled_channel(int base, const ScaleSpec& scale) {
    const float value = static_cast<float>((std::min)(base, scale.max_channels)) * scale.width;
    return static_cast<int>(std::floor((value + 4.0f) / 8.0f)) * 8;
}

int scaled_repeat(int base, const ScaleSpec& scale) {
    return base > 1 ? (std::max)(static_cast<int>(std::round(base * scale.depth)), 1) : base;
}

bool validate_detection_config(const DetectionConfig& cfg, std::string* error) {
    if (!std::isfinite(cfg.confidence_threshold) || cfg.confidence_threshold < 0.0f ||
        cfg.confidence_threshold > 1.0f || cfg.max_detections < 1 || cfg.max_detections > 300) {
        set_error(error, "detection config requires confidence_threshold in [0,1] and max_detections in [1,300]");
        return false;
    }
    return true;
}

#ifdef _WIN32
std::wstring utf8_to_wide(const std::string& value) {
    if (value.empty()) return {};
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                           static_cast<int>(value.size()), nullptr, 0);
    if (count <= 0) return {};
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                        static_cast<int>(value.size()), result.data(), count);
    return result;
}

bool decode_wic(IStream* stream, const wchar_t* path, std::vector<uint8_t>& rgb,
                int& width, int& height, std::string* error) {
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapFlipRotator* rotator = nullptr;
    IWICFormatConverter* converter = nullptr;
    const char* stage = "COM initialization";
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninit = SUCCEEDED(hr);
    stage = "factory creation";
    hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                          IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) {
        stage = "decoder creation";
        hr = stream ? factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand, &decoder)
                    : factory->CreateDecoderFromFilename(path, nullptr, GENERIC_READ,
                                                         WICDecodeMetadataCacheOnDemand, &decoder);
    }
    if (SUCCEEDED(hr)) { stage = "frame decode"; hr = decoder->GetFrame(0, &frame); }
    IWICBitmapSource* source = frame;
    if (SUCCEEDED(hr)) {
        IWICMetadataQueryReader* metadata = nullptr;
        PROPVARIANT value;
        PropVariantInit(&value);
        if (SUCCEEDED(frame->GetMetadataQueryReader(&metadata)) &&
            SUCCEEDED(metadata->GetMetadataByName(L"/app1/ifd/{ushort=274}", &value)) &&
            value.vt == VT_UI2 && value.uiVal >= 2 && value.uiVal <= 8) {
            static constexpr WICBitmapTransformOptions kExifTransforms[] = {
                WICBitmapTransformRotate0,
                WICBitmapTransformFlipHorizontal,
                WICBitmapTransformRotate180,
                WICBitmapTransformFlipVertical,
                static_cast<WICBitmapTransformOptions>(WICBitmapTransformRotate90 |
                                                        WICBitmapTransformFlipHorizontal),
                WICBitmapTransformRotate90,
                static_cast<WICBitmapTransformOptions>(WICBitmapTransformRotate270 |
                                                        WICBitmapTransformFlipHorizontal),
                WICBitmapTransformRotate270,
            };
            if (SUCCEEDED(factory->CreateBitmapFlipRotator(&rotator)) &&
                SUCCEEDED(rotator->Initialize(frame, kExifTransforms[value.uiVal - 1]))) {
                source = rotator;
            }
        }
        PropVariantClear(&value);
        if (metadata) metadata->Release();
    }
    if (SUCCEEDED(hr)) { stage = "format converter creation"; hr = factory->CreateFormatConverter(&converter); }
    // Requesting 32bpp RGBA avoids decoder-specific restrictions on odd
    // 24bpp row strides (present in valid COCO JPEGs such as width=634).
    if (SUCCEEDED(hr)) {
        stage = "pixel format conversion";
        hr = converter->Initialize(source, GUID_WICPixelFormat32bppRGBA,
                                   WICBitmapDitherTypeNone, nullptr, 0.0,
                                   WICBitmapPaletteTypeCustom);
    }
    UINT w = 0, h = 0;
    if (SUCCEEDED(hr)) { stage = "image dimensions"; hr = converter->GetSize(&w, &h); }
    if (SUCCEEDED(hr) && w && h && w <= 65535 && h <= 65535) {
        const uint64_t bytes = static_cast<uint64_t>(w) * h * 4;
        if (bytes <= (std::numeric_limits<UINT>::max)()) {
            std::vector<uint8_t> rgba(static_cast<size_t>(bytes));
            stage = "pixel copy";
            hr = converter->CopyPixels(nullptr, w * 4, static_cast<UINT>(bytes), rgba.data());
            if (SUCCEEDED(hr)) {
                rgb.resize(static_cast<size_t>(w) * h * 3);
                for (size_t pixel = 0; pixel < static_cast<size_t>(w) * h; ++pixel) {
                    rgb[pixel * 3] = rgba[pixel * 4];
                    rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1];
                    rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2];
                }
                width = static_cast<int>(w); height = static_cast<int>(h);
            }
        } else hr = E_OUTOFMEMORY;
    } else if (SUCCEEDED(hr)) { stage = "image dimension validation"; hr = E_INVALIDARG; }
    if (converter) converter->Release();
    if (rotator) rotator->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (uninit) CoUninitialize();
    if (FAILED(hr)) {
        char message[96]{};
        std::snprintf(message, sizeof(message),
                      "WIC failed during %s (HRESULT=0x%08lX, size=%ux%u)", stage,
                      static_cast<unsigned long>(hr), w, h);
        set_error(error, message); rgb.clear(); return false;
    }
    return true;
}
#endif

bool pixels_to_rgb(const ImageView& image, std::vector<uint8_t>& rgb, std::string* error) {
    int channels = 0;
    switch (image.format) {
        case PixelFormat::RGB8: case PixelFormat::BGR8: channels = 3; break;
        case PixelFormat::RGBA8: case PixelFormat::BGRA8: channels = 4; break;
        default: break;
    }
    const uint64_t packed_row = image.width > 0
        ? static_cast<uint64_t>(image.width) * static_cast<uint64_t>(channels) : 0;
    if (!image.data || channels == 0 || image.width <= 0 || image.height <= 0 ||
        image.row_stride < 0 || static_cast<uint64_t>(image.row_stride) < packed_row) {
        set_error(error, "invalid raw image view"); return false;
    }
    const uint64_t needed = static_cast<uint64_t>(image.row_stride) *
                                static_cast<uint64_t>(image.height - 1) + packed_row;
    if (needed > image.size_bytes) {
        set_error(error, "raw image buffer does not contain the final row"); return false;
    }
    rgb.resize(static_cast<size_t>(image.width) * image.height * 3);
    for (int y = 0; y < image.height; ++y) {
        const uint8_t* src = image.data + static_cast<size_t>(y) * image.row_stride;
        uint8_t* dst = rgb.data() + static_cast<size_t>(y) * image.width * 3;
        for (int x = 0; x < image.width; ++x) {
            const bool bgr = image.format == PixelFormat::BGR8 || image.format == PixelFormat::BGRA8;
            dst[3 * x] = src[channels * x + (bgr ? 2 : 0)];
            dst[3 * x + 1] = src[channels * x + 1];
            dst[3 * x + 2] = src[channels * x + (bgr ? 0 : 2)];
        }
    }
    return true;
}

void letterbox_fp16(const std::vector<uint8_t>& rgb, int width, int height,
                    std::vector<f16>& output, float& gain, int& pad_x, int& pad_y) {
    constexpr int target = 640;
    gain = (std::min)(static_cast<float>(target) / height, static_cast<float>(target) / width);
    const int resized_w = static_cast<int>(std::round(width * gain));
    const int resized_h = static_cast<int>(std::round(height * gain));
    pad_x = static_cast<int>(std::round((target - resized_w) / 2.0f - 0.1f));
    pad_y = static_cast<int>(std::round((target - resized_h) / 2.0f - 0.1f));
    output.assign(static_cast<size_t>(3) * target * target, f16(114.0f / 255.0f));
    for (int y = 0; y < resized_h; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) / gain - 0.5f;
        const int y0 = (std::max)(static_cast<int>(std::floor(source_y)), 0);
        const int y1 = (std::min)(y0 + 1, height - 1);
        const float fy = (std::max)(source_y - std::floor(source_y), 0.0f);
        for (int x = 0; x < resized_w; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) / gain - 0.5f;
            const int x0 = (std::max)(static_cast<int>(std::floor(source_x)), 0);
            const int x1 = (std::min)(x0 + 1, width - 1);
            const float fx = (std::max)(source_x - std::floor(source_x), 0.0f);
            for (int c = 0; c < 3; ++c) {
                const float p00 = rgb[(static_cast<size_t>(y0) * width + x0) * 3 + c];
                const float p01 = rgb[(static_cast<size_t>(y0) * width + x1) * 3 + c];
                const float p10 = rgb[(static_cast<size_t>(y1) * width + x0) * 3 + c];
                const float p11 = rgb[(static_cast<size_t>(y1) * width + x1) * 3 + c];
                const float top = p00 + (p01 - p00) * fx;
                const float bottom = p10 + (p11 - p10) * fx;
                // Ultralytics applies cv::INTER_LINEAR to an 8-bit image before
                // normalization, so retain the intermediate uint8 rounding.
                const float resized = std::clamp(
                    std::round(top + (bottom - top) * fy), 0.0f, 255.0f);
                output[(static_cast<size_t>(c) * target + pad_y + y) * target + pad_x + x] =
                    f16(resized / 255.0f);
            }
        }
    }
}

} // namespace

struct Yolo26Detector::Impl {
    Context* ctx = nullptr;
    ModelWeights* weights = nullptr;
    Yolo26Config cfg;
    Tensor input;
    ConvLayer layer0, layer1, layer3, layer5, layer7, layer17, layer20;
    C3k2 layer2, layer4, layer6, layer8, layer13, layer16, layer19, layer22;
    SPPF layer9;
    C2PSA layer10;
    Tensor up11, cat12, up14, cat15, cat18, cat21;
    std::array<DetectBranch, 3> detect;
    Tensor decoded_boxes, class_scores;
    int last_width = 0, last_height = 0;
    float last_gain = 1.0f;
    int last_pad_x = 0, last_pad_y = 0;
    double preprocess_ms = 0.0, device_wall_ms = 0.0, postprocess_ms = 0.0;

    void initialize() {
        const ScaleSpec scale = scale_spec(cfg.scale);
        const int c1 = scaled_channel(64, scale);
        const int c2 = scaled_channel(128, scale);
        const int c3 = scaled_channel(256, scale);
        const int c4 = scaled_channel(512, scale);
        const int c5 = scaled_channel(1024, scale);
        const int repeats = scaled_repeat(2, scale);
        // Ultralytics switches the two shallow C3k2 blocks from Bottleneck to
        // C3k for the m/l/x variants while retaining Bottleneck for n/s.
        const bool wide_c3k = cfg.scale == "m" || cfg.scale == "l" || cfg.scale == "x";
        input = Tensor::allocate(*ctx, {1, 3, 640, 640}, dnnl::memory::data_type::f16);
        layer0.init(*ctx, *weights, "aila.layers.0", input, c1, 3, 2, 1, true);
        layer1.init(*ctx, *weights, "aila.layers.1", layer0.output(), c2, 3, 2, 1, true);
        layer2.init(*ctx, *weights, "aila.layers.2", layer1.output(), c3, repeats, wide_c3k, 0.25f);
        layer3.init(*ctx, *weights, "aila.layers.3", layer2.output(), c3, 3, 2, 1, true);
        layer4.init(*ctx, *weights, "aila.layers.4", layer3.output(), c4, repeats, wide_c3k, 0.25f);
        layer5.init(*ctx, *weights, "aila.layers.5", layer4.output(), c4, 3, 2, 1, true);
        layer6.init(*ctx, *weights, "aila.layers.6", layer5.output(), c4, repeats, true, 0.5f);
        layer7.init(*ctx, *weights, "aila.layers.7", layer6.output(), c5, 3, 2, 1, true);
        layer8.init(*ctx, *weights, "aila.layers.8", layer7.output(), c5, repeats, true, 0.5f);
        layer9.init(*ctx, *weights, "aila.layers.9", layer8.output());
        layer10.init(*ctx, *weights, "aila.layers.10", layer9.output(), repeats);
        up11 = Tensor::allocate(*ctx, {1, c5, 40, 40}, dnnl::memory::data_type::f16);
        cat12 = Tensor::allocate(*ctx, {1, c5 + c4, 40, 40}, dnnl::memory::data_type::f16);
        layer13.init(*ctx, *weights, "aila.layers.13", cat12, c4, repeats, true, 0.5f);
        up14 = Tensor::allocate(*ctx, {1, c4, 80, 80}, dnnl::memory::data_type::f16);
        cat15 = Tensor::allocate(*ctx, {1, 2 * c4, 80, 80}, dnnl::memory::data_type::f16);
        layer16.init(*ctx, *weights, "aila.layers.16", cat15, c3, repeats, true, 0.5f);
        layer17.init(*ctx, *weights, "aila.layers.17", layer16.output(), c3, 3, 2, 1, true);
        cat18 = Tensor::allocate(*ctx, {1, c3 + c4, 40, 40}, dnnl::memory::data_type::f16);
        layer19.init(*ctx, *weights, "aila.layers.19", cat18, c4, repeats, true, 0.5f);
        layer20.init(*ctx, *weights, "aila.layers.20", layer19.output(), c4, 3, 2, 1, true);
        cat21 = Tensor::allocate(*ctx, {1, c4 + c5, 20, 20}, dnnl::memory::data_type::f16);
        layer22.init(*ctx, *weights, "aila.layers.22", cat21, c5, 1, true, 0.5f, true);
        const int box_channels = (std::max)(16, c3 / 4);
        const int cls_channels = (std::max)(c3, (std::min)(cfg.num_classes, 100));
        detect[0].init(*ctx, *weights, "aila.layers.23", 0, layer16.output(), box_channels, cls_channels, cfg.num_classes);
        detect[1].init(*ctx, *weights, "aila.layers.23", 1, layer19.output(), box_channels, cls_channels, cfg.num_classes);
        detect[2].init(*ctx, *weights, "aila.layers.23", 2, layer22.output(), box_channels, cls_channels, cfg.num_classes);
        decoded_boxes = Tensor::allocate(*ctx, {4, 8400}, dnnl::memory::data_type::f32);
        class_scores = Tensor::allocate(*ctx, {cfg.num_classes, 8400}, dnnl::memory::data_type::f32);
    }

    void forward() {
        layer0.forward(*ctx); dump_irtensor(*ctx, "layer0", layer0.output());
        layer1.forward(*ctx); dump_irtensor(*ctx, "layer1", layer1.output());
        layer2.forward(*ctx); dump_irtensor(*ctx, "layer2", layer2.output());
        layer3.forward(*ctx); dump_irtensor(*ctx, "layer3", layer3.output());
        layer4.forward(*ctx); dump_irtensor(*ctx, "layer4", layer4.output());
        layer5.forward(*ctx); dump_irtensor(*ctx, "layer5", layer5.output());
        layer6.forward(*ctx); dump_irtensor(*ctx, "layer6", layer6.output());
        layer7.forward(*ctx); dump_irtensor(*ctx, "layer7", layer7.output());
        layer8.forward(*ctx); dump_irtensor(*ctx, "layer8", layer8.output());
        layer9.forward(*ctx); dump_irtensor(*ctx, "layer9", layer9.output());
        layer10.forward(*ctx); dump_irtensor(*ctx, "layer10", layer10.output());
        upsample_nearest(*ctx, layer10.output(), up11);
        concat_channels(*ctx, {&up11, &layer6.output()}, cat12); layer13.forward(*ctx);
        dump_irtensor(*ctx, "layer13", layer13.output());
        upsample_nearest(*ctx, layer13.output(), up14);
        concat_channels(*ctx, {&up14, &layer4.output()}, cat15); layer16.forward(*ctx);
        dump_irtensor(*ctx, "layer16", layer16.output());
        layer17.forward(*ctx); concat_channels(*ctx, {&layer17.output(), &layer13.output()}, cat18);
        layer19.forward(*ctx); dump_irtensor(*ctx, "layer19", layer19.output());
        layer20.forward(*ctx);
        concat_channels(*ctx, {&layer20.output(), &layer10.output()}, cat21); layer22.forward(*ctx);
        dump_irtensor(*ctx, "layer22", layer22.output());
        for (size_t index = 0; index < detect.size(); ++index) {
            detect[index].forward(*ctx);
            dump_irtensor(*ctx, "detect_box" + std::to_string(index), detect[index].box2.output());
            dump_irtensor(*ctx, "detect_cls" + std::to_string(index), detect[index].cls_out.output());
        }
    }

    void decode_on_device() {
        float* decoded = decoded_boxes.data_as<float>();
        float* scores = class_scores.data_as<float>();
        constexpr int total_anchors = 8400;
        const int sides[3] = {80, 40, 20};
        int anchor_base = 0;
        for (int level = 0; level < 3; ++level) {
            const int side = sides[level];
            const int count = side * side;
            const int base = anchor_base;
            const float stride = static_cast<float>(cfg.strides[level]);
            const int classes = cfg.num_classes;
            const f16* boxes = detect[level].box2.output().data_as<f16>();
            const f16* logits = detect[level].cls_out.output().data_as<f16>();
            ctx->queue().parallel_for(
                sycl::range<1>(static_cast<size_t>(4 + classes) * count),
                [=](sycl::id<1> index) {
                    const size_t linear = index[0];
                    const int local = static_cast<int>(linear % count);
                    const int channel = static_cast<int>(linear / count);
                    const int global = base + local;
                    if (channel < 4) {
                        const int x = local % side;
                        const int y = local / side;
                        const float anchor = channel % 2 == 0
                            ? static_cast<float>(x) + 0.5f
                            : static_cast<float>(y) + 0.5f;
                        const float distance = static_cast<float>(boxes[channel * count + local]);
                        const float sign = channel < 2 ? -1.0f : 1.0f;
                        decoded[channel * total_anchors + global] =
                            (anchor + sign * distance) * stride;
                    } else {
                        const int cls = channel - 4;
                        const float logit = static_cast<float>(logits[cls * count + local]);
                        scores[cls * total_anchors + global] =
                            1.0f / (1.0f + sycl::exp(-logit));
                    }
                });
            anchor_base += count;
        }
    }

    bool run_rgb(const std::vector<uint8_t>& rgb, int width, int height,
                 const DetectionConfig& detection_config,
                 std::vector<Detection>& output, std::string* error) {
        if (!validate_detection_config(detection_config, error) || width <= 0 || height <= 0 ||
            rgb.size() != static_cast<size_t>(width) * height * 3) return false;
        std::vector<f16> preprocessed;
        const auto preprocess_start = std::chrono::steady_clock::now();
        letterbox_fp16(rgb, width, height, preprocessed, last_gain, last_pad_x, last_pad_y);
        const auto preprocess_end = std::chrono::steady_clock::now();
        last_width = width; last_height = height;
        ctx->memcpy_h2d(input.data(), preprocessed.data(), input.size_bytes());
        dump_irtensor(*ctx, "input", input);
        const auto device_start = std::chrono::steady_clock::now();
        forward();
        decode_on_device();
        ctx->synchronize();
        const auto device_end = std::chrono::steady_clock::now();

        struct Candidate { float score; int anchor; int cls; float box[4]; };
        struct AnchorCandidate { float maximum; int anchor; int level; int local; };
        std::vector<AnchorCandidate> anchors;
        anchors.reserve(8400);
        std::vector<float> box_host(static_cast<size_t>(4) * 8400);
        std::vector<float> cls_host(static_cast<size_t>(cfg.num_classes) * 8400);
        ctx->memcpy_d2h(box_host.data(), decoded_boxes.data(), decoded_boxes.size_bytes());
        ctx->memcpy_d2h(cls_host.data(), class_scores.data(), class_scores.size_bytes());
        const int sizes[3] = {80, 40, 20};
        int anchor_base = 0;
        for (int level = 0; level < 3; ++level) {
            const int count = sizes[level] * sizes[level];
            for (int local = 0; local < count; ++local) {
                float maximum = 0.0f;
                for (int cls = 0; cls < cfg.num_classes; ++cls) {
                    maximum = (std::max)(maximum,
                        cls_host[static_cast<size_t>(cls) * 8400 + anchor_base + local]);
                }
                anchors.push_back({maximum, anchor_base + local, level, local});
            }
            anchor_base += count;
        }
        const int first_k = (std::min)(300, static_cast<int>(anchors.size()));
        std::partial_sort(anchors.begin(), anchors.begin() + first_k, anchors.end(),
            [](const AnchorCandidate& a, const AnchorCandidate& b) {
                return a.maximum != b.maximum ? a.maximum > b.maximum : a.anchor < b.anchor;
            });
        anchors.resize(static_cast<size_t>(first_k));
        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(first_k) * cfg.num_classes);
        for (const auto& anchor : anchors) {
            float decoded[4] = {
                box_host[anchor.anchor], box_host[8400 + anchor.anchor],
                box_host[2 * 8400 + anchor.anchor], box_host[3 * 8400 + anchor.anchor],
            };
            for (int cls = 0; cls < cfg.num_classes; ++cls) {
                const float score = cls_host[static_cast<size_t>(cls) * 8400 + anchor.anchor];
                candidates.push_back({score, anchor.anchor, cls,
                                      {decoded[0], decoded[1], decoded[2], decoded[3]}});
            }
        }
        const int second_k = (std::min)(detection_config.max_detections, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(), candidates.begin() + second_k, candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.score != b.score) return a.score > b.score;
                if (a.anchor != b.anchor) return a.anchor < b.anchor;
                return a.cls < b.cls;
            });
        output.clear(); output.reserve(static_cast<size_t>(second_k));
        for (int i = 0; i < second_k && candidates[i].score >= detection_config.confidence_threshold; ++i) {
            const Candidate& candidate = candidates[i];
            Detection result;
            result.x1 = std::clamp((candidate.box[0] - last_pad_x) / last_gain, 0.0f, static_cast<float>(width));
            result.y1 = std::clamp((candidate.box[1] - last_pad_y) / last_gain, 0.0f, static_cast<float>(height));
            result.x2 = std::clamp((candidate.box[2] - last_pad_x) / last_gain, 0.0f, static_cast<float>(width));
            result.y2 = std::clamp((candidate.box[3] - last_pad_y) / last_gain, 0.0f, static_cast<float>(height));
            result.confidence = candidate.score; result.class_id = candidate.cls;
            result.class_name = cfg.class_names[static_cast<size_t>(candidate.cls)];
            output.push_back(std::move(result));
        }
        const auto postprocess_end = std::chrono::steady_clock::now();
        preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
        device_wall_ms = std::chrono::duration<double, std::milli>(device_end - device_start).count();
        postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - device_end).count();
        return true;
    }
};

Yolo26Detector::Yolo26Detector() : impl_(std::make_unique<Impl>()) {}
Yolo26Detector::~Yolo26Detector() = default;

bool Yolo26Detector::init(Context& ctx, ModelWeights& weights, const Yolo26Config& config,
                          std::string* error_message) {
    try {
        impl_->ctx = &ctx; impl_->weights = &weights; impl_->cfg = config;
        impl_->initialize();
        AILA_LOG_INFO("[YOLO26] Loaded scale=%s classes=%d input=640x640 dtype=fp16",
                      config.scale.c_str(), config.num_classes);
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, e.what()); return false;
    }
}

bool Yolo26Detector::detect_file(const std::string& path, const DetectionConfig& config,
                                 std::vector<Detection>& detections, std::string* error_message) {
#ifdef _WIN32
    const std::wstring wide = utf8_to_wide(path);
    if (wide.empty()) { set_error(error_message, "invalid UTF-8 image path"); return false; }
    std::ifstream input(std::filesystem::path(wide), std::ios::binary | std::ios::ate);
    if (!input) { set_error(error_message, "failed to open image file"); return false; }
    const std::streamsize length = input.tellg();
    if (length <= 0 || static_cast<uint64_t>(length) >
            static_cast<uint64_t>((std::numeric_limits<DWORD>::max)())) {
        set_error(error_message, "invalid encoded image file size"); return false;
    }
    std::vector<uint8_t> encoded(static_cast<size_t>(length));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(encoded.data()), length)) {
        set_error(error_message, "failed to read image file"); return false;
    }
    return detect_encoded(encoded.data(), encoded.size(), config, detections, error_message);
#else
    (void)path; (void)config; (void)detections;
    set_error(error_message, "YOLO26 encoded image decoding currently requires Windows WIC"); return false;
#endif
}

bool Yolo26Detector::detect_encoded(const uint8_t* data, size_t size, const DetectionConfig& config,
                                    std::vector<Detection>& detections, std::string* error_message) {
#ifdef _WIN32
    if (!data || !size || size > (std::numeric_limits<DWORD>::max)()) {
        set_error(error_message, "invalid encoded image buffer"); return false;
    }
    HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, size);
    if (!memory) { set_error(error_message, "failed to allocate WIC image stream"); return false; }
    void* target = GlobalLock(memory);
    if (!target) {
        GlobalFree(memory); set_error(error_message, "failed to lock WIC image stream"); return false;
    }
    std::memcpy(target, data, size); GlobalUnlock(memory);
    IStream* stream = nullptr;
    HRESULT hr = CreateStreamOnHGlobal(memory, TRUE, &stream);
    if (FAILED(hr)) { GlobalFree(memory); set_error(error_message, "failed to create WIC image stream"); return false; }
    std::vector<uint8_t> rgb; int width = 0, height = 0;
    const bool decoded = decode_wic(stream, nullptr, rgb, width, height, error_message);
    stream->Release();
    return decoded && impl_->run_rgb(rgb, width, height, config, detections, error_message);
#else
    (void)data; (void)size; (void)config; (void)detections;
    set_error(error_message, "YOLO26 encoded image decoding currently requires Windows WIC"); return false;
#endif
}

bool Yolo26Detector::detect_pixels(const ImageView& image, const DetectionConfig& config,
                                   std::vector<Detection>& detections, std::string* error_message) {
    std::vector<uint8_t> rgb;
    if (!pixels_to_rgb(image, rgb, error_message)) return false;
    return impl_->run_rgb(rgb, image.width, image.height, config, detections, error_message);
}

int Yolo26Detector::last_image_width() const { return impl_->last_width; }
int Yolo26Detector::last_image_height() const { return impl_->last_height; }
double Yolo26Detector::last_preprocess_ms() const { return impl_->preprocess_ms; }
double Yolo26Detector::last_device_wall_ms() const { return impl_->device_wall_ms; }
double Yolo26Detector::last_postprocess_ms() const { return impl_->postprocess_ms; }
const Yolo26Config& Yolo26Detector::config() const { return impl_->cfg; }

} // namespace aila::vision
