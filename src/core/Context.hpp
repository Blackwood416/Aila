#pragma once

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// ============================================================
// SYCL + oneDNN 运行时上下文
// ============================================================
class Context {
public:
    struct ExecutionStats {
        long long lock_count = 0;
        double wait_ms_total = 0.0;
        double wait_ms_max = 0.0;
        double hold_ms_total = 0.0;
        double hold_ms_max = 0.0;
    };

    class ExecutionLock {
    public:
        class ScopedUnlock {
        public:
            explicit ScopedUnlock(ExecutionLock& lock)
                : lock_(lock) {
                lock_.unlock();
            }

            ScopedUnlock(const ScopedUnlock&) = delete;
            ScopedUnlock& operator=(const ScopedUnlock&) = delete;

            ~ScopedUnlock() {
                lock_.lock();
            }

        private:
            ExecutionLock& lock_;
        };

        explicit ExecutionLock(Context& ctx)
            : ctx_(&ctx),
              lock_(ctx.execution_mutex_, std::defer_lock) {
            lock();
        }

        ExecutionLock(ExecutionLock&& other) noexcept
            : ctx_(other.ctx_),
              lock_(std::move(other.lock_)),
              hold_start_(other.hold_start_) {
            other.ctx_ = nullptr;
        }

        ExecutionLock(const ExecutionLock&) = delete;
        ExecutionLock& operator=(const ExecutionLock&) = delete;
        ExecutionLock& operator=(ExecutionLock&&) = delete;

        ~ExecutionLock() {
            if (ctx_ && lock_.owns_lock()) {
                unlock();
            }
        }

        void unlock() {
            if (!ctx_ || !lock_.owns_lock()) {
                return;
            }
            const auto now = Clock::now();
            const double hold_ms =
                std::chrono::duration<double, std::milli>(now - hold_start_).count();
            ctx_->record_execution_hold(hold_ms);
            lock_.unlock();
        }

        void lock() {
            if (!ctx_ || lock_.owns_lock()) {
                return;
            }
            const auto wait_start = Clock::now();
            lock_.lock();
            const auto acquired = Clock::now();
            const double wait_ms =
                std::chrono::duration<double, std::milli>(acquired - wait_start).count();
            ctx_->record_execution_wait(wait_ms);
            hold_start_ = acquired;
        }

        ScopedUnlock scoped_unlock() { return ScopedUnlock(*this); }

    private:
        using Clock = std::chrono::steady_clock;
        Context* ctx_ = nullptr;
        std::unique_lock<std::mutex> lock_;
        Clock::time_point hold_start_{};
    };

    Context()
        : Context(sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order()}) {}

    explicit Context(sycl::queue queue) {
        q_ = std::move(queue);
        eng_ = dnnl::sycl_interop::make_engine(q_.get_device(), q_.get_context());
        stream_ = dnnl::sycl_interop::make_stream(eng_, q_);
    }

    sycl::queue& queue() { return q_; }
    dnnl::engine& engine() { return eng_; }
    dnnl::stream& stream() { return stream_; }
    ExecutionLock lock_execution() { return ExecutionLock(*this); }

    ExecutionStats execution_stats() const {
        std::lock_guard<std::mutex> lock(execution_stats_mutex_);
        return execution_stats_;
    }

    void reset_execution_stats() {
        std::lock_guard<std::mutex> lock(execution_stats_mutex_);
        execution_stats_ = ExecutionStats{};
    }

    // USM Device memory allocation
    void* alloc_device(size_t bytes) {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
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
        std::lock_guard<std::mutex> lock(alloc_mutex_);
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

    size_t current_allocated_bytes() const {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        return current_allocated_bytes_;
    }

    size_t peak_allocated_bytes() const {
        std::lock_guard<std::mutex> lock(alloc_mutex_);
        return peak_allocated_bytes_;
    }

private:
    void record_execution_wait(double wait_ms) {
        std::lock_guard<std::mutex> lock(execution_stats_mutex_);
        ++execution_stats_.lock_count;
        execution_stats_.wait_ms_total += wait_ms;
        if (wait_ms > execution_stats_.wait_ms_max) {
            execution_stats_.wait_ms_max = wait_ms;
        }
    }

    void record_execution_hold(double hold_ms) {
        std::lock_guard<std::mutex> lock(execution_stats_mutex_);
        execution_stats_.hold_ms_total += hold_ms;
        if (hold_ms > execution_stats_.hold_ms_max) {
            execution_stats_.hold_ms_max = hold_ms;
        }
    }

    sycl::queue q_;
    dnnl::engine eng_;
    dnnl::stream stream_;
    std::mutex execution_mutex_;
    mutable std::mutex execution_stats_mutex_;
    ExecutionStats execution_stats_;
    mutable std::mutex alloc_mutex_;
    std::unordered_map<void*, size_t> alloc_bytes_;
    size_t current_allocated_bytes_ = 0;
    size_t peak_allocated_bytes_ = 0;
};
