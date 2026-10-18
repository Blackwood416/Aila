#pragma once

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <memory>
#include <typeindex>
#include "memory/AllocationBudget.hpp"
#include "memory/LargePages.hpp"
#include "core/AsyncErrorState.hpp"
#include "core/ExecutionTrace.hpp"
#include "core/UncachedHostAllocation.hpp"

// ============================================================
// SYCL + oneDNN 运行时上下文
// ============================================================
class Context {
    using Trace = aila::diagnostics::ExecutionTrace;
    using TraceScope = aila::diagnostics::ExecutionScope;
    using TraceStage = aila::diagnostics::ExecutionStage;
public:
    Context() : Context(nullptr) {}
    explicit Context(Trace* trace, bool enableProfiling = false)
        : trace_(trace), async_errors_(std::make_shared<aila::AsyncErrorState>()),
          q_(sycl::default_selector_v,
             [errors = async_errors_](sycl::exception_list exceptions) noexcept {
                 for (const auto& error : exceptions) errors->Record(error);
             },
             enableProfiling
                 ? sycl::property_list{sycl::property::queue::in_order(),
                                       sycl::property::queue::enable_profiling()}
                 : sycl::property_list{sycl::property::queue::in_order()}) {
        TraceScope stage(trace_, TraceStage::ContextInitialize);
        eng_ = dnnl::sycl_interop::make_engine(q_.get_device(), q_.get_context());
        stream_ = dnnl::sycl_interop::make_stream(eng_, q_);
        stage.Complete();
        CheckTrace();
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    ~Context() {
        // This marks body cleanup only. Queue/stream destruction follows the
        // body; the owning DeviceState keeps the writer alive beyond that.
        TraceScope stage(trace_, TraceStage::ContextRelease);
        // Execution-domain owners record failures before teardown. Legacy
        // Context users still need the queue drained before cached USM dies.
        try { synchronize(); } catch (...) {}
        release_execution_resources();
        stage.Complete();
    }

    sycl::queue& queue() { async_errors_->Rethrow(); CheckTrace(); return q_; }
    dnnl::engine& engine() { async_errors_->Rethrow(); CheckTrace(); return eng_; }
    dnnl::stream& stream() { async_errors_->Rethrow(); CheckTrace(); return stream_; }

    // USM Device memory allocation
    void* alloc_device(size_t bytes) {
        return allocate(bytes, false);
    }

    void* alloc_host(size_t bytes) {
        return allocate(bytes, true);
    }

    // Level Zero host USM with the uncached bias, required for device code that
    // polls host-published doorbells. Counted against the pinned-host budget.
    void* alloc_host_uncached(size_t bytes) {
        async_errors_->Rethrow();
        CheckTrace();
        if (!bytes) {
            throw std::invalid_argument(
                "Uncached host allocation must have positive size");
        }
        host_budget_.Check(bytes, "Uncached host");
        const auto allocation =
            aila::memory::AllocateUncachedHost(q_.get_context(), bytes);
        bool recorded = false;
        try {
            const auto inserted = alloc_bytes_.emplace(
                allocation.ptr,
                Allocation{bytes, AllocationKind::UncachedHost,
                           allocation.native_context});
            if (!inserted.second) {
                throw std::runtime_error(
                    "Uncached host allocation returned a duplicate pointer");
            }
            recorded = true;
            host_budget_.Acquire(bytes, "Uncached host");
            return allocation.ptr;
        } catch (...) {
            if (recorded) {
                alloc_bytes_.erase(allocation.ptr);
            }
            aila::memory::FreeUncachedHost(
                allocation.native_context, allocation.ptr);
            throw;
        }
    }

    // Explicitly resident CPU storage. This pointer is not USM and must not be
    // dereferenced by a kernel. Account the rounded physical allocation.
    void* alloc_large_host(size_t bytes) {
        async_errors_->Rethrow();
        CheckTrace();
        bytes = aila::memory::LargePageAllocationSize(bytes);
        host_budget_.Check(bytes, "Large-page host");
        void* ptr = aila::memory::AllocateLargePages(bytes);
        try {
            alloc_bytes_.emplace(
                ptr, Allocation{bytes, AllocationKind::LargePageHost});
            host_budget_.Acquire(bytes, "Large-page host");
            return ptr;
        } catch (...) {
            alloc_bytes_.erase(ptr);
            aila::memory::FreeLargePages(ptr);
            throw;
        }
    }

    void free_device(void* ptr) {
        if (!ptr) return;
        auto it = alloc_bytes_.find(ptr);
        aila::diagnostics::ExecutionFields fields;
        AllocationKind kind = AllocationKind::Device;
        void* native_context = nullptr;
        if (it != alloc_bytes_.end()) {
            fields.bytes = it->second.bytes;
            fields.memory = it->second.IsHost()
                                ? aila::diagnostics::TraceMemory::PinnedHost
                                : aila::diagnostics::TraceMemory::Device;
            kind = it->second.kind;
            native_context = it->second.native_context;
        }
        TraceScope stage(trace_, TraceStage::UsmFree, fields);
        if (it != alloc_bytes_.end()) {
            auto& budget = it->second.IsHost() ? host_budget_ : device_budget_;
            budget.Release(it->second.bytes);
            alloc_bytes_.erase(it);
        }
        switch (kind) {
        case AllocationKind::LargePageHost:
            aila::memory::FreeLargePages(ptr);
            break;
        case AllocationKind::UncachedHost:
            aila::memory::FreeUncachedHost(native_context, ptr);
            break;
        case AllocationKind::Device:
        case AllocationKind::PinnedHost:
            sycl::free(ptr, q_);
            break;
        }
        stage.Complete(); // Never throw a trace error from a resource free.
    }

    void free_host(void* ptr) { free_device(ptr); }

    void set_allocation_limits(size_t deviceBytes, size_t pinnedHostBytes) {
        if ((deviceBytes && device_budget_.Current() > deviceBytes) ||
            (pinnedHostBytes && host_budget_.Current() > pinnedHostBytes)) {
            throw aila::memory::AllocationLimitExceeded("Existing");
        }
        device_budget_.SetLimit(deviceBytes);
        host_budget_.SetLimit(pinnedHostBytes);
    }

    void check_device_allocation(size_t bytes) const {
        device_budget_.Check(bytes, "Device");
    }

    // One instance per Context, never per process/thread. All accesses use the
    // same external execution lock as queue submission and mutable bindings.
    template <typename T>
    T& workspace() {
        const std::type_index key(typeid(T));
        auto it = workspaces_.find(key);
        if (it == workspaces_.end()) {
            it = workspaces_.emplace(key, std::make_shared<T>(*this)).first;
        }
        return *static_cast<T*>(it->second.get());
    }

    // User-mode oneDNN scratch shared by sequential submissions on the in-order
    // queue. Rebind on every execute: a larger primitive can grow this buffer.
    dnnl::memory scratchpad_memory(const dnnl::memory::desc& md) {
        const size_t bytes = md.get_size();
        if (!bytes) return {};
        if (bytes > scratchpad_capacity_) {
            synchronize();
            void* replacement = alloc_device(bytes);
            free_device(scratchpad_);
            scratchpad_ = replacement;
            scratchpad_capacity_ = bytes;
        }
        return dnnl::sycl_interop::make_memory(
            md, eng_, dnnl::sycl_interop::memory_kind::usm, scratchpad_);
    }

    // Requires a drained queue. This also drops primitive caches owned by a
    // workspace; surviving models rebind shared scratch before their next use.
    void release_execution_resources() {
        TraceScope stage(trace_, TraceStage::WorkspaceRelease);
        workspaces_.clear();
        free_device(scratchpad_);
        scratchpad_ = nullptr;
        scratchpad_capacity_ = 0;
        stage.Complete();
    }

    // Decode has a dedicated primitive. Bound additional shape specializations
    // so varying prompt lengths cannot accumulate one primitive per request.
    template <typename Cache>
    void make_primitive_cache_room(Cache& cache) {
        constexpr size_t maxEntries = 16;
        if (cache.size() >= maxEntries) {
            synchronize();
            cache.erase(cache.begin());
        }
    }

    // Host -> Device copy (blocking, for synchronous operations)
    void memcpy_h2d(void* dst, const void* src, size_t bytes) {
        TraceScope stage(trace_, TraceStage::CopyH2DWait);
        queue().memcpy(dst, src, bytes).wait_and_throw();
        async_errors_->Rethrow();
        stage.Complete();
        CheckTrace();
    }

    // Device -> Host copy (blocking, for synchronous operations)
    void memcpy_d2h(void* dst, const void* src, size_t bytes) {
        TraceScope stage(trace_, TraceStage::CopyD2HWait);
        queue().memcpy(dst, src, bytes).wait_and_throw();
        async_errors_->Rethrow();
        stage.Complete();
        CheckTrace();
    }

    // Async Host -> Device copy (non-blocking, for pipeline)
    sycl::event memcpy_h2d_async(void* dst, const void* src, size_t bytes) {
        return queue().memcpy(dst, src, bytes);  // No wait!
    }

    // Async Device -> Host copy (non-blocking, for pipeline)
    sycl::event memcpy_d2h_async(void* dst, const void* src, size_t bytes) {
        return queue().memcpy(dst, src, bytes);  // No wait!
    }

    void synchronize() {
        // Always attempt to drain, including after a recorded fault. A custom
        // async handler records rather than terminates; explicitly rethrow so
        // DeviceRuntime can fault the domain and stop the remaining requests.
        TraceScope stage(trace_, TraceStage::QueueWait);
        async_errors_->DrainAndRethrow([&] { q_.wait_and_throw(); });
        stage.Complete();
        CheckTrace(); // Check only after drain, preserving any device first cause.
    }

    size_t current_allocated_bytes() const { return device_budget_.Current(); }
    size_t peak_allocated_bytes() const { return device_budget_.Peak(); }
    size_t current_pinned_host_bytes() const { return host_budget_.Current(); }
    size_t peak_pinned_host_bytes() const { return host_budget_.Peak(); }
    size_t device_allocation_count() const { return device_budget_.Count(); }
    size_t pinned_host_allocation_count() const { return host_budget_.Count(); }
    size_t device_allocation_limit() const { return device_budget_.Limit(); }
    size_t pinned_host_allocation_limit() const { return host_budget_.Limit(); }

private:
    // Borrowed from DeviceState, whose writer is destroyed after this Context.
    Trace* trace_ = nullptr;
    void CheckTrace() const { if (trace_) trace_->Check(); }
    // Declared before q_: it also remains alive in the handler's shared capture.
    std::shared_ptr<aila::AsyncErrorState> async_errors_;
    sycl::queue q_;
    dnnl::engine eng_;
    dnnl::stream stream_;
    enum class AllocationKind : unsigned char {
        Device,
        PinnedHost,
        UncachedHost,
        LargePageHost,
    };
    struct Allocation {
        size_t bytes;
        AllocationKind kind;
        void* native_context = nullptr;

        bool IsHost() const noexcept { return kind != AllocationKind::Device; }
    };
    std::unordered_map<void*, Allocation> alloc_bytes_;
    aila::memory::AllocationBudget device_budget_, host_budget_;
    std::unordered_map<std::type_index, std::shared_ptr<void>> workspaces_;
    void* scratchpad_ = nullptr;
    size_t scratchpad_capacity_ = 0;

    void* allocate(size_t bytes, bool host) {
        async_errors_->Rethrow();
        CheckTrace();
        if (!bytes) throw std::invalid_argument("USM allocation must have positive size");
        auto& budget = host ? host_budget_ : device_budget_;
        const char* kind = host ? "Pinned host" : "Device";
        budget.Check(bytes, kind);
        void* ptr = nullptr;
        try {
            ptr = host ? sycl::malloc_host(bytes, q_) : sycl::malloc_device(bytes, q_);
            if (!ptr) throw std::runtime_error("USM allocation failed: " + std::to_string(bytes) + " bytes");
            alloc_bytes_.emplace(
                ptr, Allocation{bytes, host ? AllocationKind::PinnedHost
                                            : AllocationKind::Device});
            budget.Acquire(bytes, kind);
            return ptr;
        } catch (...) {
            if (ptr) sycl::free(ptr, q_);
            throw;
        }
    }
};
