#pragma once

#include "Context.hpp"
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cassert>

// ============================================================
// 轻量级 GPU Tensor
// ============================================================
class Tensor {
public:
    Tensor() = default;

    // 在 GPU 上分配新 tensor
    static Tensor allocate(Context& ctx,
                           const std::vector<int64_t>& shape,
                           dnnl::memory::data_type dtype = dnnl::memory::data_type::bf16) {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.ctx_ = &ctx;
        t.owns_memory_ = true;
        size_t bytes = t.size_bytes();
        t.data_ptr_ = ctx.alloc_device(bytes);
        return t;
    }

    // 从已有 USM 指针创建 view（不拥有内存）
    static Tensor view(Context& ctx, void* ptr,
                       const std::vector<int64_t>& shape,
                       dnnl::memory::data_type dtype = dnnl::memory::data_type::bf16) {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.ctx_ = &ctx;
        t.owns_memory_ = false;
        t.data_ptr_ = ptr;
        return t;
    }

    // 从 dnnl::memory + 元数据创建（用于从 SafeTensors 加载的权重）
    static Tensor from_dnnl(Context& ctx, dnnl::memory mem,
                            const std::vector<int64_t>& shape,
                            dnnl::memory::data_type dtype) {
        Tensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.ctx_ = &ctx;
        t.owns_memory_ = false; // dnnl::memory 管理生命周期
        t.data_ptr_ = mem.get_data_handle();
        t.dnnl_mem_ = mem;
        t.has_dnnl_mem_ = true;
        return t;
    }

    ~Tensor() {
        if (owns_memory_ && data_ptr_ && ctx_) {
            ctx_->free_device(data_ptr_);
            data_ptr_ = nullptr;
        }
    }

    // Move semantics
    Tensor(Tensor&& other) noexcept {
        *this = std::move(other);
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owns_memory_ && data_ptr_ && ctx_) {
                ctx_->free_device(data_ptr_);
            }
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            data_ptr_ = other.data_ptr_;
            ctx_ = other.ctx_;
            owns_memory_ = other.owns_memory_;
            dnnl_mem_ = other.dnnl_mem_;
            has_dnnl_mem_ = other.has_dnnl_mem_;

            other.data_ptr_ = nullptr;
            other.owns_memory_ = false;
            other.has_dnnl_mem_ = false;
        }
        return *this;
    }

    // No copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // ------- Accessors -------

    void* data() { return data_ptr_; }
    const void* data() const { return data_ptr_; }

    template<typename T>
    T* data_as() { return static_cast<T*>(data_ptr_); }

    const std::vector<int64_t>& shape() const { return shape_; }
    int64_t shape(int dim) const { return shape_[dim]; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    dnnl::memory::data_type dtype() const { return dtype_; }
    Context* context() const { return ctx_; }
    bool valid() const { return data_ptr_ != nullptr; }

    int64_t numel() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), (int64_t)1, std::multiplies<int64_t>());
    }

    size_t element_size() const {
        switch (dtype_) {
            case dnnl::memory::data_type::f32: return 4;
            case dnnl::memory::data_type::bf16:
            case dnnl::memory::data_type::f16: return 2;
            case dnnl::memory::data_type::s8:
            case dnnl::memory::data_type::u8:
            case dnnl::memory::data_type::f8_e4m3:
            case dnnl::memory::data_type::f8_e5m2: return 1;
            case dnnl::memory::data_type::s32: return 4;
            case dnnl::memory::data_type::f64: return 8;
            default: return 0;
        }
    }

    size_t size_bytes() const {
        return numel() * element_size();
    }

    // 创建 row-major (plain) 的 dnnl::memory::desc
    dnnl::memory::desc desc() const {
        dnnl::memory::dims dims(shape_.begin(), shape_.end());
        auto tag = get_format_tag(static_cast<int>(shape_.size()));
        return dnnl::memory::desc(dims, dtype_, tag);
    }

    // 创建带自定义 strides 的 desc（用于转置等）
    dnnl::memory::desc desc_with_strides(const std::vector<int64_t>& strides) const {
        dnnl::memory::dims dims(shape_.begin(), shape_.end());
        dnnl::memory::dims str(strides.begin(), strides.end());
        return dnnl::memory::desc(dims, dtype_, str);
    }

    // 获取或创建 dnnl::memory 对象
    dnnl::memory& dnnl_memory() {
        if (!has_dnnl_mem_) {
            assert(ctx_ && data_ptr_);
            auto md = desc();
            dnnl_mem_ = dnnl::sycl_interop::make_memory(
                md, ctx_->engine(),
                dnnl::sycl_interop::memory_kind::usm,
                data_ptr_
            );
            has_dnnl_mem_ = true;
        }
        return dnnl_mem_;
    }

    // 用指定的 desc 创建 dnnl::memory (覆盖默认 desc)
    dnnl::memory make_dnnl_memory(const dnnl::memory::desc& md) {
        assert(ctx_ && data_ptr_);
        return dnnl::sycl_interop::make_memory(
            md, ctx_->engine(),
            dnnl::sycl_interop::memory_kind::usm,
            data_ptr_
        );
    }

    // 零拷贝 reshape（返回新 Tensor view）
    Tensor reshape_view(const std::vector<int64_t>& new_shape) {
        // 验证元素数量一致
        int64_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(),
                                            (int64_t)1, std::multiplies<int64_t>());
        assert(new_numel == numel());
        return Tensor::view(*ctx_, data_ptr_, new_shape, dtype_);
    }

private:
    static dnnl::memory::format_tag get_format_tag(int ndims) {
        switch (ndims) {
            case 1: return dnnl::memory::format_tag::a;
            case 2: return dnnl::memory::format_tag::ab;
            case 3: return dnnl::memory::format_tag::abc;
            case 4: return dnnl::memory::format_tag::abcd;
            case 5: return dnnl::memory::format_tag::abcde;
            default: return dnnl::memory::format_tag::a;
        }
    }

    std::vector<int64_t> shape_;
    dnnl::memory::data_type dtype_ = dnnl::memory::data_type::bf16;
    void* data_ptr_ = nullptr;
    Context* ctx_ = nullptr;
    bool owns_memory_ = false;

    dnnl::memory dnnl_mem_;
    bool has_dnnl_mem_ = false;
};
