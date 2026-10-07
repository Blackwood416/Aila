#pragma once

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

namespace ops::cpu {

// Session-owned synchronous tile executor. All workers are created before
// inference; the caller also claims work. A published callback and its stack
// scratch remain borrowed until every worker acknowledges the generation.
class CpuTilePool {
public:
    explicit CpuTilePool(int threads) {
        if (threads < 1 || threads > 18) throw std::invalid_argument(
            "CPU tile pool requires 1..18 threads");
        try {
            workers_.reserve(threads - 1);
            for (int i = 1; i < threads; ++i)
                workers_.emplace_back([this] { Worker(); });
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
        {
            // Pair with the wait mutex to prevent a missed wake-up between
            // predicate inspection and parking. Release publishes job fields.
            std::lock_guard<std::mutex> lock(waitMutex_);
            generation_.fetch_add(1, std::memory_order_release);
        }
        workCv_.notify_all();
        Consume();
        Complete();
        auto deadline = std::chrono::steady_clock::now() +
            std::chrono::microseconds(1000);
        while (remaining_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) _mm_pause();
        if (remaining_.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(waitMutex_);
            doneCv_.wait(lock, [&] {
                return remaining_.load(std::memory_order_acquire) == 0;
            });
        }
        callbackState_ = nullptr;
        callback_ = nullptr;
        active_.store(false, std::memory_order_release);
        if (error_) std::rethrow_exception(error_);
    }
private:
    void Consume() noexcept {
        while (!failed_.load(std::memory_order_acquire)) {
            const int first = next_.fetch_add(4, std::memory_order_relaxed);
            if (first >= tileCount_) return;
            try { callback_(callbackState_, first, std::min(first + 4, tileCount_)); }
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
            std::lock_guard<std::mutex> lock(waitMutex_);
            doneCv_.notify_one();
        }
    }
    void Worker() noexcept {
        uint64_t seen = 0;
        while (true) {
            auto deadline = std::chrono::steady_clock::now() +
                std::chrono::microseconds(1000);
            uint64_t generation;
            while ((generation = generation_.load(std::memory_order_acquire)) == seen &&
                   !stopping_.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) _mm_pause();
            if (generation == seen) {
                std::unique_lock<std::mutex> lock(waitMutex_);
                workCv_.wait(lock, [&] {
                    return stopping_.load(std::memory_order_acquire) ||
                        generation_.load(std::memory_order_acquire) != seen;
                });
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
            stopping_.store(true, std::memory_order_release);
        }
        workCv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    std::vector<std::thread> workers_;
    std::mutex waitMutex_, errorMutex_;
    std::condition_variable workCv_, doneCv_;
    alignas(64) std::atomic<uint64_t> generation_{0};
    alignas(64) std::atomic<int> next_{0};
    alignas(64) std::atomic<int> remaining_{0};
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
