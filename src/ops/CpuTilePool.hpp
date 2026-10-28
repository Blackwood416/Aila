#pragma once

#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <immintrin.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ops::cpu {

inline const std::vector<int>& PhysicalCoreIds() {
    static const std::vector<int> cores = [] {
        std::vector<int> ids;
#ifdef _WIN32
        DWORD length = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
        if (length != 0) {
            std::vector<char> buffer(length);
            if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) {
                for (DWORD offset = 0; offset < length;) {
                    const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(
                        buffer.data() + offset);
                    if (entry->Size == 0) break;
                    if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount > 0) {
                        const GROUP_AFFINITY& group = entry->Processor.GroupMask[0];
                        for (int bit = 0; bit < 64; ++bit)
                            if (group.Mask & (KAFFINITY(1) << bit)) {
                                ids.push_back(int(group.Group) * 64 + bit);
                                break;
                            }
                    }
                    offset += entry->Size;
                }
            }
        }
        if (ids.empty())
            for (WORD group = 0; group < GetActiveProcessorGroupCount(); ++group) {
                const DWORD count = GetActiveProcessorCount(group);
                for (DWORD index = 0; index < count && index < 64; ++index)
                    ids.push_back(int(group) * 64 + int(index));
            }
#else
        for (unsigned index = 0; index < std::thread::hardware_concurrency(); ++index)
            ids.push_back(int(index));
#endif
        return ids;
    }();
    return cores;
}

inline int PhysicalCoreCount() { return int(PhysicalCoreIds().size()); }

inline void PinCurrentThreadToCore(int core_index) {
    if (core_index < 0) return;
#if defined(_WIN32)
    const auto& ids = PhysicalCoreIds();
    const int logical_cpu = (core_index < int(ids.size())) ? ids[size_t(core_index)] : core_index;
    GROUP_AFFINITY target{};
    target.Group = WORD(unsigned(logical_cpu) / 64u);
    target.Mask = KAFFINITY(1) << (unsigned(logical_cpu) & 63u);
    if (SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr)) {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
#else
    (void)core_index;
#endif
}

inline int TilePoolWorkerCore(int worker_index, int first_core) {
    if (worker_index < 0) return -1;
    const int total = PhysicalCoreCount();
    if (total < 6) return -1;
    const int begin = std::clamp(first_core, 2, total - 1);
    return begin + (worker_index % (total - begin));
}

inline bool TilePoolPinningEnabled() {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_CPU_POOL_PIN", true);
    return enabled;
}

inline int TilePoolSpinMicroseconds() {
    static const int spin_us = std::clamp(
        aila::env::read_int_raw("AILA_Q4E_CPU_POOL_SPIN_US", 20000), 0, 100000);
    return spin_us;
}

// Session-owned synchronous tile executor. All workers are created before
// inference; the caller also claims work. A published callback and its stack
// scratch remain borrowed until every worker acknowledges the generation.
class CpuTilePool {
public:
    explicit CpuTilePool(int threads, int first_core = 3) {
        if (threads < 1 || threads > 18) throw std::invalid_argument(
            "CPU tile pool requires 1..18 threads");
        const bool pin = TilePoolPinningEnabled();
        try {
            workers_.reserve(threads - 1);
            for (int i = 1; i < threads; ++i) {
                const int core = pin ? TilePoolWorkerCore(i - 1, first_core) : -1;
                workers_.emplace_back([this, core] {
                    PinCurrentThreadToCore(core);
                    Worker();
                });
            }
        } catch (...) { Stop(); throw; }
    }
    ~CpuTilePool() { Stop(); }
    CpuTilePool(const CpuTilePool&) = delete;
    CpuTilePool& operator=(const CpuTilePool&) = delete;

    template<class Body> void ParallelFor(int tiles, const Body& body) {
        if (tiles <= 0) return;
        // Recursive use must be rejected before changing the active job.
        if (active_.exchange(true, std::memory_order_acq_rel))
            throw std::logic_error("CPU tile pool already executing a batch");
        tileCount_ = tiles;
        callbackState_ = &body;
        callback_ = [](const void* state, int first, int last) {
            (*static_cast<const Body*>(state))(first, last);
        };
        error_ = {};
        failed_.store(false, std::memory_order_relaxed);
        next_.store(0, std::memory_order_relaxed);
        remaining_.store(static_cast<int>(workers_.size()) + 1,
            std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_seq_cst);
        if (sleepers_.load(std::memory_order_seq_cst) != 0) {
            std::lock_guard<std::mutex> lock(waitMutex_);
            workCv_.notify_all();
        }
        Consume();
        Complete();
        const int spin_us = TilePoolSpinMicroseconds();
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::microseconds(std::max(1000, spin_us));
        unsigned wait_iter = 0;
        while (remaining_.load(std::memory_order_acquire)) {
            _mm_pause();
            if ((++wait_iter & 1023u) == 0 &&
                std::chrono::steady_clock::now() >= deadline) break;
        }
        if (remaining_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(waitMutex_);
            doneWaiters_.fetch_add(1, std::memory_order_seq_cst);
            doneCv_.wait(lock, [&] {
                return remaining_.load(std::memory_order_acquire) == 0;
            });
            doneWaiters_.fetch_sub(1, std::memory_order_relaxed);
        }
        callbackState_ = nullptr;
        callback_ = nullptr;
        active_.store(false, std::memory_order_release);
        if (error_) std::rethrow_exception(error_);
    }
private:
    void Consume() noexcept {
        static const int s_claim_tiles = std::clamp(
            aila::env::read_int_raw("AILA_Q4E_CPU_POOL_CLAIM", 1), 1, 16);
        while (!failed_.load(std::memory_order_acquire)) {
            const int first = next_.fetch_add(s_claim_tiles, std::memory_order_relaxed);
            if (first >= tileCount_) return;
            try { callback_(callbackState_, first, std::min(first + s_claim_tiles, tileCount_)); }
            catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex_);
                if (!error_) error_ = std::current_exception();
                failed_.store(true, std::memory_order_release);
                return;
            }
        }
    }
    void Complete() noexcept {
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (doneWaiters_.load(std::memory_order_seq_cst) != 0) {
                std::lock_guard<std::mutex> lock(waitMutex_);
                doneCv_.notify_one();
            }
        }
    }
    void Worker() noexcept {
        const int spin_us = TilePoolSpinMicroseconds();
        uint64_t seen = 0;
        while (true) {
            uint64_t generation = generation_.load(std::memory_order_acquire);
            if (generation == seen && spin_us > 0 &&
                !stopping_.load(std::memory_order_acquire)) {
                const auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(spin_us);
                unsigned iteration = 0;
                while ((generation = generation_.load(std::memory_order_acquire)) == seen &&
                       !stopping_.load(std::memory_order_acquire)) {
                    _mm_pause();
                    if ((++iteration & 1023u) == 0 &&
                        std::chrono::steady_clock::now() >= deadline) break;
                }
            }
            if (generation == seen && !stopping_.load(std::memory_order_acquire)) {
                std::unique_lock<std::mutex> lock(waitMutex_);
                sleepers_.fetch_add(1, std::memory_order_seq_cst);
                workCv_.wait(lock, [&] {
                    return stopping_.load(std::memory_order_acquire) ||
                        generation_.load(std::memory_order_seq_cst) != seen;
                });
                sleepers_.fetch_sub(1, std::memory_order_relaxed);
                generation = generation_.load(std::memory_order_acquire);
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            seen = generation;
            Consume();
            Complete();
        }
    }
    void Stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(waitMutex_);
            stopping_.store(true, std::memory_order_seq_cst);
            generation_.fetch_add(1, std::memory_order_seq_cst);
            workCv_.notify_all();
        }
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    std::vector<std::thread> workers_;
    std::mutex waitMutex_, errorMutex_;
    std::condition_variable workCv_, doneCv_;
    alignas(64) std::atomic<uint64_t> generation_{0};
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> remaining_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    alignas(64) std::atomic<int> doneWaiters_{0};
    std::atomic<bool> stopping_{false}, active_{false}, failed_{false};
    int tileCount_ = 0;
    const void* callbackState_ = nullptr;
    void (*callback_)(const void*, int, int) = nullptr;
    std::exception_ptr error_;
};

// The synchronous coordinator lends its own session pool only to this call.
// Shadows explicitly clear the scope, and concurrent sessions have different
// pools. No worker accesses this thread-local pointer.
inline thread_local CpuTilePool* currentTilePool = nullptr;
class CpuTilePoolScope {
public:
    explicit CpuTilePoolScope(CpuTilePool* pool) : previous_(currentTilePool) {
        currentTilePool = pool;
    }
    ~CpuTilePoolScope() { currentTilePool = previous_; }
private:
    CpuTilePool* previous_;
};
}
