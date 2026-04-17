#include "Ops.hpp"
#include <sycl/sycl.hpp>

using bf16 = sycl::ext::oneapi::bfloat16;

// ============================================================
// Linear 层实现 (oneDNN MatMul)
// ============================================================

void Linear::init(Context& ctx, Tensor& weight, int in_features, int out_features, bool preprocessed) {
    weight_ = &weight;
    in_features_ = in_features;
    out_features_ = out_features;
    preprocessed_ = preprocessed;

    // 预创建 decode (seq_len=1) primitive
    decode_src_md_ = dnnl::memory::desc({1, in_features}, dnnl::memory::data_type::bf16,
                                         dnnl::memory::format_tag::ab);
    
    if (preprocessed_) {
        // [in, out] 物理布局
        decode_weight_md_ = dnnl::memory::desc({in_features, out_features},
                                                dnnl::memory::data_type::bf16,
                                                dnnl::memory::format_tag::ab);
    } else {
        // [out, in] 物理布局 (逻辑上是 [in, out] 转置)
        decode_weight_md_ = dnnl::memory::desc({in_features, out_features},
                                                dnnl::memory::data_type::bf16,
                                                {1, static_cast<int64_t>(in_features)});
    }

    decode_dst_md_ = dnnl::memory::desc({1, out_features}, dnnl::memory::data_type::bf16,
                                         dnnl::memory::format_tag::ab);

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::matmul::primitive_desc(ctx.engine(), decode_src_md_,
                                            decode_weight_md_, decode_dst_md_, attr);
    decode_prim_ = dnnl::matmul(pd);
    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    if (scratchpad_size > 0) {
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

    auto src_md = dnnl::memory::desc({seq_len, in_features_}, dnnl::memory::data_type::bf16,
                                      dnnl::memory::format_tag::ab);
    
    dnnl::memory::desc weight_md;
    if (preprocessed_) {
        weight_md = dnnl::memory::desc({in_features_, out_features_},
                                        dnnl::memory::data_type::bf16,
                                        dnnl::memory::format_tag::ab);
    } else {
        weight_md = dnnl::memory::desc({in_features_, out_features_},
                                        dnnl::memory::data_type::bf16,
                                        {1, static_cast<int64_t>(in_features_)});
    }
    
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
    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    if (scratchpad_size > 0) {
        cp.scratchpad = Tensor::allocate(ctx,
                                         {static_cast<int64_t>(scratchpad_size)},
                                         dnnl::memory::data_type::u8);
        cp.scratchpad_mem = cp.scratchpad.make_dnnl_memory(pd.scratchpad_desc());
    }
    prim_cache_.emplace(seq_len, std::move(cp));
}

void Linear::forward(Context& ctx, Tensor& input, Tensor& output, int seq_len) {
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
            if (cp.scratchpad.valid()) {
                cp.args.emplace(DNNL_ARG_SCRATCHPAD, cp.scratchpad_mem);
            }
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

        cp.prim.execute(ctx.stream(), cp.args);
    }
}

