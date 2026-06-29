#pragma once

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

// ============================================================
// SYCL + oneDNN 运行时上下文
// ============================================================
class Context {
public:
    Context() {
        q_ = sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()};
        eng_ = dnnl::sycl_interop::make_engine(q_.get_device(), q_.get_context());
        stream_ = dnnl::sycl_interop::make_stream(eng_, q_);

    }

    sycl::queue& queue() { return q_; }
    dnnl::engine& engine() { return eng_; }
    dnnl::stream& stream() { return stream_; }

    // USM Device memory allocation
    void* alloc_device(size_t bytes) {
        void* ptr = sycl::malloc_device(bytes, q_);
        if (!ptr) {
            throw std::runtime_error("GPU memory allocation failed: " + std::to_string(bytes) + " bytes");
        }
        alloc_bytes_[ptr] = bytes;
        current_allocated_bytes_ += bytes;
        if (current_allocated_bytes_ > peak_allocated_bytes_) {
            peak_allocated_bytes_ = current_allocated_bytes_;
        }
        return ptr;
    }

    void free_device(void* ptr) {
        if (!ptr) return;
        auto it = alloc_bytes_.find(ptr);
        if (it != alloc_bytes_.end()) {
            size_t bytes = it->second;
            if (current_allocated_bytes_ >= bytes) {
                current_allocated_bytes_ -= bytes;
            } else {
                current_allocated_bytes_ = 0;
            }
            alloc_bytes_.erase(it);
        }
        sycl::free(ptr, q_);
    }

    // Host -> Device copy (blocking, for synchronous operations)
    void memcpy_h2d(void* dst, const void* src, size_t bytes) {
        q_.memcpy(dst, src, bytes).wait();
    }

    // Device -> Host copy (blocking, for synchronous operations)
    void memcpy_d2h(void* dst, const void* src, size_t bytes) {
        q_.memcpy(dst, src, bytes).wait();
    }

    // Async Host -> Device copy (non-blocking, for pipeline)
    sycl::event memcpy_h2d_async(void* dst, const void* src, size_t bytes) {
        return q_.memcpy(dst, src, bytes);  // No wait!
    }

    // Async Device -> Host copy (non-blocking, for pipeline)
    sycl::event memcpy_d2h_async(void* dst, const void* src, size_t bytes) {
        return q_.memcpy(dst, src, bytes);  // No wait!
    }

    void synchronize() {
        q_.wait_and_throw();
    }

    size_t current_allocated_bytes() const { return current_allocated_bytes_; }
    size_t peak_allocated_bytes() const { return peak_allocated_bytes_; }

private:
    sycl::queue q_;
    dnnl::engine eng_;
    dnnl::stream stream_;
    std::unordered_map<void*, size_t> alloc_bytes_;
    size_t current_allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
};
