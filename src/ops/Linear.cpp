#include "Ops.hpp"
#include "../utils/EnvUtils.hpp"
#include "../profile/Profiling.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

dnnl::memory::desc linear_weight_md(bool preprocessed, int in_features,
                                    int out_features) {
    if (preprocessed) {
        return dnnl::memory::desc({in_features, out_features},
                                  dnnl::memory::data_type::bf16,
                                  dnnl::memory::format_tag::ab);
    }
    return dnnl::memory::desc({in_features, out_features},
                              dnnl::memory::data_type::bf16,
                              {1, static_cast<int64_t>(in_features)});
}

uint64_t fused_linear_key(int seq_len, dnnl::memory::data_type bias_dtype) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(seq_len)) << 32) |
           static_cast<uint32_t>(static_cast<int>(bias_dtype));
}

struct SharedLinearPrim {
    dnnl::engine engine;
    dnnl::matmul prim;
    dnnl::memory::desc src_md;
    dnnl::memory::desc weight_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc scratchpad_md;
};

struct SharedLinearKey {
    uintptr_t engine = 0;
    int seq_len = 0;
    int in_features = 0;
    int out_features = 0;
    int variant = 0;
    bool operator<(const SharedLinearKey& other) const {
        return std::tie(engine, seq_len, in_features, out_features, variant) <
               std::tie(other.engine, other.seq_len, other.in_features,
                        other.out_features, other.variant);
    }
};

std::mutex& shared_linear_prim_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<SharedLinearKey, SharedLinearPrim>& shared_linear_prims() {
    static auto* prims = new std::map<SharedLinearKey, SharedLinearPrim>();
    return *prims;
}

constexpr size_t kMaxSharedLinearPrims = 256;

SharedLinearKey make_shared_linear_key(Context& ctx, int seq_len, int in_features,
                                       int out_features, int variant) {
    SharedLinearKey key;
    key.engine = reinterpret_cast<uintptr_t>(ctx.engine().get(true));
    key.seq_len = seq_len;
    key.in_features = in_features;
    key.out_features = out_features;
    key.variant = variant;
    return key;
}

bool lookup_shared_linear_prim(const SharedLinearKey& key, SharedLinearPrim& out) {
    std::lock_guard<std::mutex> lock(shared_linear_prim_mutex());
    const auto& prims = shared_linear_prims();
    const auto it = prims.find(key);
    if (it == prims.end()) return false;
    out = it->second;
    return true;
}

void store_shared_linear_prim(const SharedLinearKey& key, const SharedLinearPrim& prim) {
    std::lock_guard<std::mutex> lock(shared_linear_prim_mutex());
    auto& prims = shared_linear_prims();
    if (prims.size() < kMaxSharedLinearPrims) prims.emplace(key, prim);
}

} // namespace

// ============================================================
// Linear 层实现 (oneDNN MatMul)
// ============================================================

void Linear::init(Context& ctx, Tensor& weight, int in_features,
                  int out_features, bool preprocessed, bool prepare_gemv) {
    weight_ = &weight;
    in_features_ = in_features;
    out_features_ = out_features;
    preprocessed_ = preprocessed;

    // Pre-transpose weight to [M, K] for native GEMV kernel (contiguous access).
    // The standard "preprocessed" weight is [K, M] row-major for oneDNN.
    // We store a [M, K] copy so the GEMV kernel can read weight rows contiguously.
    if (preprocessed && prepare_gemv && in_features > 0 && out_features > 0) {
        gemv_weight_ = Tensor::allocate(ctx, {out_features, in_features});
        ops::transpose(ctx, weight, gemv_weight_);
        ctx.synchronize();
    }

    // 预创建 decode (seq_len=1) primitive
    decode_src_md_ = dnnl::memory::desc({1, in_features}, dnnl::memory::data_type::bf16,
                                         dnnl::memory::format_tag::ab);
    
    decode_weight_md_ = linear_weight_md(preprocessed_, in_features, out_features);

    decode_dst_md_ = dnnl::memory::desc({1, out_features}, dnnl::memory::data_type::bf16,
                                         dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), decode_src_md_,
                                            decode_weight_md_, decode_dst_md_, attr);
    decode_prim_ = dnnl::matmul(pd);
    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    if (scratchpad_size > 0) {
        if (scratchpad_size > 8 * 1024 * 1024) {
            AILA_LOG_INFO("[Linear] decode scratchpad in=%d out=%d size=%.1f MB",
                          in_features, out_features,
                          scratchpad_size / (1024.0 * 1024.0));
        }
        decode_scratchpad_ = Tensor::allocate(ctx,
                                              {static_cast<int64_t>(scratchpad_size)},
                                              dnnl::memory::data_type::u8);
        decode_scratchpad_mem_ = decode_scratchpad_.make_dnnl_memory(pd.scratchpad_desc());
    }
    decode_inited_ = true;
}

void Linear::ensure_primitive(Context& ctx, int seq_len) {
    if (seq_len == 1 && decode_inited_) return;
    if (prim_cache_.count(seq_len)) return;
    const SharedLinearKey shared_key = make_shared_linear_key(
        ctx, seq_len, in_features_, out_features_, preprocessed_ ? 1 : 0);
    SharedLinearPrim shared_prim{};
    if (lookup_shared_linear_prim(shared_key, shared_prim)) {
        ctx.make_primitive_cache_room(prim_cache_);
        CachedPrimitive cp;
        cp.prim = shared_prim.prim;
        cp.src_md = shared_prim.src_md;
        cp.weight_md = shared_prim.weight_md;
        cp.dst_md = shared_prim.dst_md;
        cp.scratchpad_md = shared_prim.scratchpad_md;
        prim_cache_.emplace(seq_len, std::move(cp));
        return;
    }
    ctx.make_primitive_cache_room(prim_cache_);

    auto src_md = dnnl::memory::desc({seq_len, in_features_}, dnnl::memory::data_type::bf16,
                                      dnnl::memory::format_tag::ab);
    
    dnnl::memory::desc weight_md = linear_weight_md(preprocessed_, in_features_, out_features_);
    
    auto dst_md = dnnl::memory::desc({seq_len, out_features_}, dnnl::memory::data_type::bf16,
                                      dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md, dst_md, attr);

    CachedPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.dst_md = dst_md;
    cp.scratchpad_md = pd.scratchpad_desc();
    store_shared_linear_prim(shared_key,
        SharedLinearPrim{ctx.engine(), cp.prim, cp.src_md, cp.weight_md,
                         cp.dst_md, cp.scratchpad_md});
    prim_cache_.emplace(seq_len, std::move(cp));
}

void Linear::ensure_primitive_f32(Context& ctx, int seq_len) {
    if (prim_cache_f32_.count(seq_len)) {
        return;
    }
    ctx.make_primitive_cache_room(prim_cache_f32_);
    auto src_md = dnnl::memory::desc({seq_len, in_features_},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);
    dnnl::memory::desc weight_md =
        linear_weight_md(preprocessed_, in_features_, out_features_);
    auto dst_md = dnnl::memory::desc({seq_len, out_features_},
                                     dnnl::memory::data_type::f32,
                                     dnnl::memory::format_tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md,
                                           dst_md, attr);
    CachedPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.dst_md = dst_md;
    cp.scratchpad_md = pd.scratchpad_desc();
    prim_cache_f32_.emplace(seq_len, std::move(cp));
}

void Linear::forward_f32(Context& ctx, Tensor& input, Tensor& output,
                         int seq_len) {
    ensure_primitive_f32(ctx, seq_len);
    auto& cp = prim_cache_f32_[seq_len];
    if (!cp.mem_inited) {
        cp.src_mem = input.make_dnnl_memory(cp.src_md);
        cp.weight_mem = weight_->make_dnnl_memory(cp.weight_md);
        cp.dst_mem = output.make_dnnl_memory(cp.dst_md);
        cp.mem_inited = true;
        cp.src_ptr = input.data();
        cp.weight_ptr = weight_->data();
        cp.dst_ptr = output.data();
        cp.args = {{DNNL_ARG_SRC, cp.src_mem},
                   {DNNL_ARG_WEIGHTS, cp.weight_mem},
                   {DNNL_ARG_DST, cp.dst_mem}};
    }
    cp.src_mem.set_data_handle(input.data());
    cp.weight_mem.set_data_handle(weight_->data());
    cp.dst_mem.set_data_handle(output.data());
    if (cp.scratchpad_md.get_size()) {
        cp.args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(cp.scratchpad_md);
    }
    cp.prim.execute(ctx.stream(), cp.args);
}

void Linear::ensure_bias_primitive(Context& ctx, int seq_len,
                                   dnnl::memory::data_type bias_dtype) {
    const uint64_t key = fused_linear_key(seq_len, bias_dtype);
    if (bias_cache_.count(key)) return;
    ctx.make_primitive_cache_room(bias_cache_);

    auto src_md = dnnl::memory::desc({seq_len, in_features_},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);
    auto weight_md = linear_weight_md(preprocessed_, in_features_, out_features_);
    auto bias_md = dnnl::memory::desc({1, out_features_}, bias_dtype,
                                      dnnl::memory::format_tag::ab);
    auto dst_md = dnnl::memory::desc({seq_len, out_features_},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md,
                                           bias_md, dst_md, attr);

    CachedBiasPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.bias_md = bias_md;
    cp.dst_md = dst_md;
    cp.scratchpad_md = pd.scratchpad_desc();
    bias_cache_.emplace(key, std::move(cp));
}

void Linear::ensure_bias_gelu_tanh_primitive(Context& ctx, int seq_len,
                                              dnnl::memory::data_type bias_dtype) {
    const uint64_t key = fused_linear_key(seq_len, bias_dtype);
    if (bias_gelu_tanh_cache_.count(key)) return;
    ctx.make_primitive_cache_room(bias_gelu_tanh_cache_);

    auto src_md = dnnl::memory::desc({seq_len, in_features_},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);
    auto weight_md = linear_weight_md(preprocessed_, in_features_, out_features_);
    auto bias_md = dnnl::memory::desc({1, out_features_}, bias_dtype,
                                      dnnl::memory::format_tag::ab);
    auto dst_md = dnnl::memory::desc({seq_len, out_features_},
                                     dnnl::memory::data_type::bf16,
                                     dnnl::memory::format_tag::ab);

    dnnl::post_ops post_ops;
    post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0.0f, 0.0f);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_post_ops(post_ops);

    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), src_md, weight_md,
                                           bias_md, dst_md, attr);

    CachedBiasGeluTanhPrimitive cp;
    cp.prim = dnnl::matmul(pd);
    cp.src_md = src_md;
    cp.weight_md = weight_md;
    cp.bias_md = bias_md;
    cp.dst_md = dst_md;
    cp.scratchpad_md = pd.scratchpad_desc();
    bias_gelu_tanh_cache_.emplace(key, std::move(cp));
}

void Linear::forward(Context& ctx, Tensor& input, Tensor& output, int seq_len) {
    // Fast path: native bf16 GEMV kernel for single-token decode.
    // Uses gemv_weight_ (pre-transposed [M,K]) for contiguous memory access.
    if (seq_len == 1 && aila::env::read_flag("AILA_BF16_GEMV", true) && gemv_weight_.valid()) {
        ops::bf16_gemv_bf16(ctx, gemv_weight_,
            input.data_as<bf16>(), output.data_as<bf16>(),
            out_features_, in_features_);
        return;
    }

    ensure_primitive(ctx, seq_len);

    if (seq_len == 1 && decode_inited_) {
        if (!decode_mem_inited_) {
            decode_src_mem_ = dnnl::sycl_interop::make_memory(decode_src_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, input.data());
            decode_weight_mem_ = dnnl::sycl_interop::make_memory(decode_weight_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, weight_->data());
            decode_dst_mem_ = dnnl::sycl_interop::make_memory(decode_dst_md_, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, output.data());
            decode_src_ptr_ = input.data();
            decode_weight_ptr_ = weight_->data();
            decode_dst_ptr_ = output.data();
            // Pre-build args map once
            decode_args_ = {
                {DNNL_ARG_SRC, decode_src_mem_},
                {DNNL_ARG_WEIGHTS, decode_weight_mem_},
                {DNNL_ARG_DST, decode_dst_mem_}
            };
            if (decode_scratchpad_.valid()) {
                decode_args_.emplace(DNNL_ARG_SCRATCHPAD, decode_scratchpad_mem_);
            }
            decode_mem_inited_ = true;
        } else {
            if (decode_src_ptr_ != input.data()) {
                decode_src_mem_.set_data_handle(input.data());
                decode_src_ptr_ = input.data();
                decode_args_[DNNL_ARG_SRC] = decode_src_mem_;
            }
            if (decode_weight_ptr_ != weight_->data()) {
                decode_weight_mem_.set_data_handle(weight_->data());
                decode_weight_ptr_ = weight_->data();
                decode_args_[DNNL_ARG_WEIGHTS] = decode_weight_mem_;
            }
            if (decode_dst_ptr_ != output.data()) {
                decode_dst_mem_.set_data_handle(output.data());
                decode_dst_ptr_ = output.data();
                decode_args_[DNNL_ARG_DST] = decode_dst_mem_;
            }
        }

        decode_prim_.execute(ctx.stream(), decode_args_);
    } else {
        auto& cp = prim_cache_[seq_len];
        if (!cp.mem_inited) {
            cp.src_mem = dnnl::sycl_interop::make_memory(cp.src_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, input.data());
            cp.weight_mem = dnnl::sycl_interop::make_memory(cp.weight_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, weight_->data());
            cp.dst_mem = dnnl::sycl_interop::make_memory(cp.dst_md, ctx.engine(),
                        dnnl::sycl_interop::memory_kind::usm, output.data());
            cp.src_ptr = input.data();
            cp.weight_ptr = weight_->data();
            cp.dst_ptr = output.data();
            cp.args = {
                {DNNL_ARG_SRC, cp.src_mem},
                {DNNL_ARG_WEIGHTS, cp.weight_mem},
                {DNNL_ARG_DST, cp.dst_mem}
            };
            cp.mem_inited = true;
        } else {
            if (cp.src_ptr != input.data()) {
                cp.src_mem.set_data_handle(input.data());
                cp.src_ptr = input.data();
                cp.args[DNNL_ARG_SRC] = cp.src_mem;
            }
            if (cp.weight_ptr != weight_->data()) {
                cp.weight_mem.set_data_handle(weight_->data());
                cp.weight_ptr = weight_->data();
                cp.args[DNNL_ARG_WEIGHTS] = cp.weight_mem;
            }
            if (cp.dst_ptr != output.data()) {
                cp.dst_mem.set_data_handle(output.data());
                cp.dst_ptr = output.data();
                cp.args[DNNL_ARG_DST] = cp.dst_mem;
            }
        }

        if (cp.scratchpad_md.get_size()) {
            cp.args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(cp.scratchpad_md);
        }
        cp.prim.execute(ctx.stream(), cp.args);
    }
}

void Linear::forward_bias(Context& ctx, Tensor& input, Tensor& bias,
                          Tensor& output, int seq_len) {
    // Fast path: native bf16 GEMV + bias for single-token decode
    if (seq_len == 1 && aila::env::read_flag("AILA_BF16_GEMV", true) && gemv_weight_.valid()) {
        ops::bf16_gemv_bf16(ctx, gemv_weight_,
            input.data_as<bf16>(), output.data_as<bf16>(),
            out_features_, in_features_);
        // Add bias in-place
        const bf16* bias_ptr = bias.data_as<bf16>();
        bf16* out = output.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(static_cast<size_t>(out_features_)),
            [=](sycl::id<1> i) {
                out[i] = bf16(static_cast<float>(out[i]) + static_cast<float>(bias_ptr[i]));
            });
        return;
    }

    const uint64_t key = fused_linear_key(seq_len, bias.dtype());
    ensure_bias_primitive(ctx, seq_len, bias.dtype());

    auto& cp = bias_cache_[key];
    if (!cp.mem_inited) {
        cp.src_mem = dnnl::sycl_interop::make_memory(cp.src_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, input.data());
        cp.weight_mem = dnnl::sycl_interop::make_memory(cp.weight_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, weight_->data());
        cp.bias_mem = dnnl::sycl_interop::make_memory(cp.bias_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, bias.data());
        cp.dst_mem = dnnl::sycl_interop::make_memory(cp.dst_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, output.data());
        cp.src_ptr = input.data();
        cp.weight_ptr = weight_->data();
        cp.bias_ptr = bias.data();
        cp.dst_ptr = output.data();
        cp.args = {
            {DNNL_ARG_SRC, cp.src_mem},
            {DNNL_ARG_WEIGHTS, cp.weight_mem},
            {DNNL_ARG_BIAS, cp.bias_mem},
            {DNNL_ARG_DST, cp.dst_mem}
        };
        cp.mem_inited = true;
    } else {
        if (cp.src_ptr != input.data()) {
            cp.src_mem.set_data_handle(input.data());
            cp.src_ptr = input.data();
            cp.args[DNNL_ARG_SRC] = cp.src_mem;
        }
        if (cp.weight_ptr != weight_->data()) {
            cp.weight_mem.set_data_handle(weight_->data());
            cp.weight_ptr = weight_->data();
            cp.args[DNNL_ARG_WEIGHTS] = cp.weight_mem;
        }
        if (cp.bias_ptr != bias.data()) {
            cp.bias_mem.set_data_handle(bias.data());
            cp.bias_ptr = bias.data();
            cp.args[DNNL_ARG_BIAS] = cp.bias_mem;
        }
        if (cp.dst_ptr != output.data()) {
            cp.dst_mem.set_data_handle(output.data());
            cp.dst_ptr = output.data();
            cp.args[DNNL_ARG_DST] = cp.dst_mem;
        }
    }

    if (cp.scratchpad_md.get_size()) {
        cp.args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(cp.scratchpad_md);
    }
    cp.prim.execute(ctx.stream(), cp.args);
}

void Linear::forward_bias_gelu_tanh(Context& ctx, Tensor& input, Tensor& bias,
                                    Tensor& output, int seq_len) {
    const uint64_t key = fused_linear_key(seq_len, bias.dtype());
    ensure_bias_gelu_tanh_primitive(ctx, seq_len, bias.dtype());

    auto& cp = bias_gelu_tanh_cache_[key];
    if (!cp.mem_inited) {
        cp.src_mem = dnnl::sycl_interop::make_memory(cp.src_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, input.data());
        cp.weight_mem = dnnl::sycl_interop::make_memory(cp.weight_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, weight_->data());
        cp.bias_mem = dnnl::sycl_interop::make_memory(cp.bias_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, bias.data());
        cp.dst_mem = dnnl::sycl_interop::make_memory(cp.dst_md, ctx.engine(),
                    dnnl::sycl_interop::memory_kind::usm, output.data());
        cp.src_ptr = input.data();
        cp.weight_ptr = weight_->data();
        cp.bias_ptr = bias.data();
        cp.dst_ptr = output.data();
        cp.args = {
            {DNNL_ARG_SRC, cp.src_mem},
            {DNNL_ARG_WEIGHTS, cp.weight_mem},
            {DNNL_ARG_BIAS, cp.bias_mem},
            {DNNL_ARG_DST, cp.dst_mem}
        };
        cp.mem_inited = true;
    } else {
        if (cp.src_ptr != input.data()) {
            cp.src_mem.set_data_handle(input.data());
            cp.src_ptr = input.data();
            cp.args[DNNL_ARG_SRC] = cp.src_mem;
        }
        if (cp.weight_ptr != weight_->data()) {
            cp.weight_mem.set_data_handle(weight_->data());
            cp.weight_ptr = weight_->data();
            cp.args[DNNL_ARG_WEIGHTS] = cp.weight_mem;
        }
        if (cp.bias_ptr != bias.data()) {
            cp.bias_mem.set_data_handle(bias.data());
            cp.bias_ptr = bias.data();
            cp.args[DNNL_ARG_BIAS] = cp.bias_mem;
        }
        if (cp.dst_ptr != output.data()) {
            cp.dst_mem.set_data_handle(output.data());
            cp.dst_ptr = output.data();
            cp.args[DNNL_ARG_DST] = cp.dst_mem;
        }
    }

    if (cp.scratchpad_md.get_size()) {
        cp.args[DNNL_ARG_SCRATCHPAD] = ctx.scratchpad_memory(cp.scratchpad_md);
    }
    cp.prim.execute(ctx.stream(), cp.args);
}

