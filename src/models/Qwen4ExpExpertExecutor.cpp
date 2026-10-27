#include "models/Qwen4ExpExpertExecutor.hpp"
#include "models/Qwen4ExpHostArena.hpp"
#include "models/Qwen4ExpMoeCopyQueue.hpp"
#include "utils/EnvUtils.hpp"
#include "ops/CpuTilePool.hpp"
#include "profile/Profiling.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <array>
#include <atomic>
#include <immintrin.h>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <cstring>
#include <tbb/task_arena.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#include <psapi.h>
#endif

namespace aila {
namespace {
uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct CpuProcessSample {
    uint32_t faults = 0;
    uint64_t user_ns = 0, kernel_ns = 0;
    bool valid = false;
};
CpuProcessSample sample_cpu_process() {
    CpuProcessSample sample;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) &&
        GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        const auto ns = [](const FILETIME& value) {
            return ((uint64_t(value.dwHighDateTime) << 32) | value.dwLowDateTime) * 100;
        };
        sample.faults = counters.PageFaultCount;
        sample.user_ns = ns(user); sample.kernel_ns = ns(kernel); sample.valid = true;
    }
#endif
    return sample;
}
void compute_expert(const ops::MoeExpertSlotHost& slot, const bf16* x, bf16* h, float* out, int hidden, int intermediate) {
    static const bool gpu_order = aila::env::read_flag("AILA_Q4E_CPU_GPU_ORDER", false);
    ops::cpu::cpu_expert_forward(slot, x, out, 1, h, nullptr, 1.0f, false, hidden, intermediate, hidden, 64,
        gpu_order ? ops::cpu::CpuKernelBackend::GpuOrderAvx2 : ops::cpu::CpuKernelBackend::Auto);
}
}

// One execution slot, no queued backlog. Only the coordinator owns cache leases.
struct Qwen4ExpExpertExecutor::CpuWorker {
    struct Job {
        std::array<ops::cpu::CpuExpertTask, ops::cpu::kMaxExpertBatchTasks> tasks;
        // Optional diagnostic refs owned by the coordinator until collect().
        const ops::cpu::CpuExpertTask* reference_tasks = nullptr;
        int count = 0, hidden = 0, intermediate = 0, tokens = 0;
        const bf16* x = nullptr;
        bf16* h = nullptr;
        float* out = nullptr;
        sycl::event activation;
        sycl::event route_event;
        sycl::event output_reader;
        sycl::event transfer_reader;
        const int* route_ids = nullptr;
        const int* route_selected = nullptr;
        const ops::MoeExpertSlotHost* route_sources = nullptr;
        bool route_sources_resident = false;
        std::array<bool,512> route_cpu_eligible{};
        int route_top_k = 0, route_experts = 0;
        bool resident_prepared = false;
        std::array<bool, ops::cpu::kMaxExpertBatchTasks> mapped_tasks{};
        // Optional grouped schedule (AILA_Q4E_VERIFY_CPU_MERGE): group g owns
        // tasks group_members[groups[g].member_begin .. +member_count). An empty
        // group_count keeps the exact ungrouped cpu_expert_batch_forward call.
        std::array<ops::cpu::CpuExpertGroup, ops::cpu::kMaxExpertBatchTasks> groups{};
        std::array<int, ops::cpu::kMaxExpertBatchTasks> group_members{};
        int group_count = 0;
    };
    struct Result {
        uint64_t activation_ns = 0, begin_ns = 0, end_ns = 0;
        uint64_t route_begin_ns = 0, route_end_ns = 0;
        int route_count = 0;
        std::array<int,ops::cpu::kMaxExpertBatchTasks> route_ranks{}, route_ids{};
        uint64_t prefetch_begin_ns = 0, prefetch_ns = 0, prefetch_calls = 0;
        uint64_t prefetch_ranges = 0, prefetch_bytes = 0, prefetch_failures = 0;
        uint64_t touch_begin_ns = 0, touch_ns = 0, touch_pages = 0;
        uint64_t diagnostic_batches = 0, diagnostic_failures = 0;
        uint64_t verify_batches = 0;
        uint64_t a64_verify_batches = 0;
        uint64_t scale_cache_bytes = 0, scale_cache_hits = 0, scale_cache_misses = 0;
        uint64_t scale_cache_fills = 0, scale_cache_evictions = 0, scale_cache_fallbacks = 0;
        uint64_t scale_cache_begin_ns = 0, scale_cache_prepare_ns = 0;
        uint64_t process_faults = 0, process_user_ns = 0, process_kernel_ns = 0;
        ops::cpu::CpuExpertPhaseTimings phases{};
        std::exception_ptr error;
    };
    std::mutex mutex;
    std::condition_variable cv;
    bool pending = false, busy = false, done = false, stop = false;
    // Signals only shorten waits; mutex/cv still own job/result publication.
    // Distinct cache lines keep the reader and writer polling independent.
    alignas(64) std::atomic<bool> pendingSignal{false};
    alignas(64) std::atomic<bool> doneSignal{false};
    const int spinUs = std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_HANDOFF_SPIN_US", 20000), 0, 50000);
    Job job;
    Result result;
    tbb::task_arena arena;
    std::unique_ptr<ops::cpu::CpuTilePool> tilePool;
    bool batched;
    bool mmap_prefetch;
    bool mmap_touch;
    bool diagnostic;
    bool verify_q8;
    bool verify_a64;
    bool scale_shift_logged = false;
    bool activation64_logged = false;
    // Diagnostic shadow buffers are allocated once, before the worker starts.
    std::vector<bf16> verify_h;
    std::vector<float> verify_out;
    static constexpr size_t kGateScaleCount = size_t(2) * ops::cpu::kIntermediateDim * ops::cpu::kHiddenDim / 64;
    static constexpr size_t kDownScaleCount = size_t(ops::cpu::kHiddenDim) * ops::cpu::kIntermediateDim / 64;
    static constexpr size_t kScaleCount = kGateScaleCount + kDownScaleCount;
    struct ScaleCacheEntry {
        ops::MoeExpertSlotHost source{};
        uint64_t pinned_job = 0;
        int next = -1;
        bool valid = false, referenced = false;
    };
    // Only run() accesses these after construction. Payload addresses never
    // move, and entries referenced by any task in this job cannot be evicted.
    std::vector<float> scale_cache_data;
    std::vector<ScaleCacheEntry> scale_cache_entries;
    std::vector<int> scale_cache_buckets;
    size_t scale_cache_clock = 0;
    uint64_t scale_cache_job = 0;
    std::thread thread;

    explicit CpuWorker(int threads) : arena(threads), batched(aila::env::read_flag("AILA_Q4E_CPU_BATCH", false)),
        mmap_prefetch(aila::env::read_flag("AILA_Q4E_CPU_MMAP_PREFETCH", false)),
        mmap_touch(aila::env::read_flag("AILA_Q4E_CPU_MMAP_TOUCH", false)),
        diagnostic(aila::env::read_flag("AILA_Q4E_CPU_DIAGNOSTIC", false)),
        verify_q8(aila::env::read_flag("AILA_Q4E_CPU_Q8_VERIFY", false)),
        verify_a64(aila::env::read_flag("AILA_Q4E_CPU_A64_VERIFY", false)),
        verify_h((verify_q8 || verify_a64) ? ops::cpu::kMaxExpertBatchTasks * ops::cpu::kIntermediateDim : 0),
        verify_out((verify_q8 || verify_a64) ? ops::cpu::kMaxExpertBatchTasks * ops::cpu::kHiddenDim : 0) {
        if (verify_a64 && (!aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8", true) ||
                          !aila::env::read_flag("AILA_Q4E_CPU_Q8_ACTIVATION64", true)))
            throw std::invalid_argument("CPU A64 verification requires the NF4Q8 Activation64 primary backend");
        init_scale_cache();
        if (aila::env::read_flag("AILA_Q4E_CPU_TILE_POOL", true)) {
            tilePool = std::make_unique<ops::cpu::CpuTilePool>(threads);
            AILA_LOG_INFO("[Q4E CpuTilePool] threads=%d caller_participates=1 claim_tiles=%d bounded_spin_us=%d "
                          "pinned=1 coordinator_core=2 worker_core_begin=3 physical_cores=%d",
                threads, std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_POOL_CLAIM", 1), 1, 16),
                std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_POOL_SPIN_US", 20000), 0, 100000),
                ops::cpu::PhysicalCoreCount());
        }
        if (spinUs) AILA_LOG_INFO("[Q4E CpuHandoffSpin] spin_us=%d mutex_publication=1 bounded_idle=1", spinUs);
        thread = std::thread([this] { run(); });
    }
    ~CpuWorker() {
        drain();
        { std::lock_guard<std::mutex> lock(mutex); stop = true;
          if (spinUs) pendingSignal.store(true, std::memory_order_release); }
        cv.notify_one();
        thread.join();
    }
    void submit(const Job& next) {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy) throw std::logic_error("CPU overlap slot still busy");
        job = next; result = {}; busy = pending = true; done = false;
        if (spinUs) {
            doneSignal.store(false, std::memory_order_relaxed);
            pendingSignal.store(true, std::memory_order_release);
        }
        cv.notify_one();
    }
    bool complete() { std::lock_guard<std::mutex> lock(mutex); return busy && done; }
    Result finish() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!busy) return {};
        if (spinUs && !done) {
            lock.unlock();
            spin_until(doneSignal);
            lock.lock();
        }
        cv.wait(lock, [&] { return done; });
        auto completed = result;
        busy = false;
        if (completed.error) std::rethrow_exception(completed.error);
        return completed;
    }
    void drain() noexcept { try { finish(); } catch (...) {} }
    void spin_until(const std::atomic<bool>& signal) const {
        if (!spinUs || signal.load(std::memory_order_acquire)) return;
        const uint64_t deadline = steady_ns() + uint64_t(spinUs) * 1000;
        unsigned iteration = 0;
        while (!signal.load(std::memory_order_acquire)) {
            _mm_pause();
            // Amortize clock reads while retaining a bounded idle interval.
            if ((++iteration & 63u) == 0 && steady_ns() >= deadline) break;
        }
    }
    void init_scale_cache() {
        static const int mb = std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_SCALE_CACHE_MB", 0), 0, 2048);
        if (!mb) return;
        const size_t budget = size_t(mb) * 1024 * 1024;
        // A power-of-two hash table at least twice capacity takes fewer than
        // four ints per entry. Include both payload and index overhead in cap.
        const size_t count = budget / (kScaleCount * sizeof(float) + sizeof(ScaleCacheEntry) + 4 * sizeof(int));
        if (!count) return;
        size_t buckets = 1;
        while (buckets < count * 2) buckets *= 2;
        scale_cache_data.resize(count * kScaleCount);
        scale_cache_entries.resize(count);
        scale_cache_buckets.assign(buckets, -1);
        AILA_LOG_INFO("[Q4E CpuScaleCacheEnabled] requested_mb=%d slots=%llu bytes=%llu packed_copy=0",
            mb, static_cast<unsigned long long>(count), static_cast<unsigned long long>(scale_cache_bytes()));
    }
    uint64_t scale_cache_bytes() const {
        return scale_cache_data.size() * sizeof(float) + scale_cache_entries.size() * sizeof(ScaleCacheEntry) +
            scale_cache_buckets.size() * sizeof(int);
    }
    size_t scale_cache_bucket(const ops::MoeExpertSlotHost& source) const {
        uint64_t hash = uint64_t(reinterpret_cast<uintptr_t>(source.gate_up_packed)) ^
            (uint64_t(reinterpret_cast<uintptr_t>(source.down_packed)) >> 3);
        hash ^= hash >> 33; hash *= UINT64_C(0xff51afd7ed558ccd);
        hash ^= hash >> 33; hash *= UINT64_C(0xc4ceb9fe1a85ec53); hash ^= hash >> 33;
        return size_t(hash) & (scale_cache_buckets.size() - 1);
    }
    static bool same_nested(const ops::MoeNestedScaleHost& a, const ops::MoeNestedScaleHost& b) {
        return a.codes == b.codes && a.nested_absmax == b.nested_absmax &&
            a.nested_quant_map == b.nested_quant_map && a.blocksize == b.blocksize &&
            std::memcmp(&a.offset, &b.offset, sizeof(float)) == 0;
    }
    int find_scales(const ops::MoeExpertSlotHost& source) const {
        for (int i = scale_cache_buckets[scale_cache_bucket(source)]; i >= 0; i = scale_cache_entries[i].next) {
            const auto& key = scale_cache_entries[i].source;
            if (key.gate_up_packed == source.gate_up_packed && key.down_packed == source.down_packed &&
                same_nested(key.gate_up_nested, source.gate_up_nested) && same_nested(key.down_nested, source.down_nested)) return i;
        }
        return -1;
    }
    static bool cacheable_nested(const ops::MoeNestedScaleHost& nested, size_t count) {
        // Mapped source validation owns array extents. Cache only its complete,
        // positive-block metadata contract; anything else retains normal reads.
        return nested.codes && nested.nested_absmax && nested.nested_quant_map && nested.blocksize > 0 &&
            count % size_t(nested.blocksize) == 0;
    }
    int claim_scales(Result& completed) {
        // Second-chance clock: pinned entries are never selected, even when the
        // cache is smaller than the job. Two rounds clear old reference bits.
        const size_t count = scale_cache_entries.size();
        for (size_t attempt = 0; attempt < count * 2; ++attempt) {
            const size_t index = scale_cache_clock;
            scale_cache_clock = (scale_cache_clock + 1) % count;
            auto& entry = scale_cache_entries[index];
            if (entry.pinned_job == scale_cache_job) continue;
            if (entry.valid && entry.referenced) { entry.referenced = false; continue; }
            if (entry.valid) {
                int* link = &scale_cache_buckets[scale_cache_bucket(entry.source)];
                while (*link >= 0 && *link != int(index)) link = &scale_cache_entries[*link].next;
                if (*link == int(index)) *link = entry.next;
                ++completed.scale_cache_evictions;
            }
            return int(index);
        }
        return -1;
    }
    void prepare_cached_scales(Job& current, Result& completed) {
        if (scale_cache_entries.empty() || !current.resident_prepared || current.count <= 0 ||
            current.count > ops::cpu::kMaxExpertBatchTasks || current.hidden != ops::cpu::kHiddenDim ||
            current.intermediate != ops::cpu::kIntermediateDim) return;
        completed.scale_cache_bytes = scale_cache_bytes();
        completed.scale_cache_begin_ns = steady_ns();
        if (++scale_cache_job == 0) {
            for (auto& entry : scale_cache_entries) entry.pinned_job = 0;
            ++scale_cache_job;
        }
        std::array<int, ops::cpu::kMaxExpertBatchTasks> indices;
        indices.fill(-1);
        std::array<bool, ops::cpu::kMaxExpertBatchTasks> eligible{};
        // Pin all hits first: a miss must not evict a later task's cached entry.
        for (int c = 0; c < current.count; ++c) {
            const auto& source = current.tasks[c].slot;
            eligible[c] = current.mapped_tasks[c] && source.gate_up_packed && source.down_packed &&
                cacheable_nested(source.gate_up_nested, kGateScaleCount) && cacheable_nested(source.down_nested, kDownScaleCount);
            if (!eligible[c]) continue;
            indices[c] = find_scales(source);
            if (indices[c] >= 0) {
                auto& entry = scale_cache_entries[indices[c]];
                entry.pinned_job = scale_cache_job; entry.referenced = true;
                ++completed.scale_cache_hits;
            } else ++completed.scale_cache_misses;
        }
        const auto fill = [](float* dst, const ops::MoeNestedScaleHost& nested, size_t count) {
            const size_t blocksize = size_t(nested.blocksize);
            for (size_t first = 0, group = 0; first < count; first += blocksize, ++group) {
                const float scale = nested.nested_absmax[group];
                for (size_t i = first; i < first + blocksize; ++i) {
                    // Match load_host_absmax's explicit two-step f32 rounding.
                    volatile float product = nested.nested_quant_map[nested.codes[i]] * scale;
                    dst[i] = product + nested.offset;
                }
            }
        };
        for (int c = 0; c < current.count; ++c) {
            if (!eligible[c]) continue;
            const auto source = current.tasks[c].slot;
            int index = indices[c];
            if (index < 0) {
                // An earlier duplicate task may already have filled this key.
                index = find_scales(source);
                if (index < 0) {
                    index = claim_scales(completed);
                    if (index < 0) { ++completed.scale_cache_fallbacks; continue; }
                    auto& entry = scale_cache_entries[index];
                    float* data = scale_cache_data.data() + size_t(index) * kScaleCount;
                    fill(data, source.gate_up_nested, kGateScaleCount);
                    fill(data + kGateScaleCount, source.down_nested, kDownScaleCount);
                    entry.source = source; entry.valid = true;
                    const size_t bucket = scale_cache_bucket(source);
                    entry.next = scale_cache_buckets[bucket]; scale_cache_buckets[bucket] = index;
                    ++completed.scale_cache_fills;
                }
            }
            auto& entry = scale_cache_entries[index];
            entry.pinned_job = scale_cache_job; entry.referenced = true;
            auto& flat = current.tasks[c].slot;
            flat.gate_up_absmax = scale_cache_data.data() + size_t(index) * kScaleCount;
            flat.down_absmax = flat.gate_up_absmax + kGateScaleCount;
            flat.gate_up_nested = {}; flat.down_nested = {};
        }
        completed.scale_cache_prepare_ns = steady_ns() - completed.scale_cache_begin_ns;
    }
    void prepare_mapped_sources(const Job& current, Result& completed) {
#ifdef _WIN32
        if ((!mmap_prefetch && !mmap_touch) || !current.resident_prepared || current.count <= 0 ||
            current.count > ops::cpu::kMaxExpertBatchTasks || current.hidden != ops::cpu::kHiddenDim ||
            current.intermediate != ops::cpu::kIntermediateDim) return;
        // Only this layer's admitted immutable sources. The worker can block in
        // the OS while the coordinator submits GPU work. Neither mode pins pages.
        WIN32_MEMORY_RANGE_ENTRY ranges[ops::cpu::kMaxExpertBatchTasks * 8];
        ULONG_PTR range_count = 0;
        uint64_t bytes = 0;
        auto append = [&](const void* address, size_t length) {
            if (!address || !length) return;
            for (ULONG_PTR i = 0; i < range_count; ++i)
                if (ranges[i].VirtualAddress == address && ranges[i].NumberOfBytes == length) return;
            ranges[range_count++] = {const_cast<void*>(address), length};
            bytes += length;
        };
        auto matrix = [&](const uint8_t* packed, const float* absmax,
                          const ops::MoeNestedScaleHost& nested, int rows, int columns) {
            append(packed, size_t(rows) * columns / 2);
            const size_t scales = size_t(rows) * columns / 64;
            if (nested.codes) {
                append(nested.codes, scales);
                const size_t blocksize = nested.blocksize > 0 ? size_t(nested.blocksize) : 256;
                append(nested.nested_absmax, ((scales + blocksize - 1) / blocksize) * sizeof(float));
                append(nested.nested_quant_map, 256 * sizeof(float));
            } else append(absmax, scales * sizeof(float));
        };
        for (int c = 0; c < current.count; ++c) {
            if (!current.mapped_tasks[c]) continue;
            const auto& source = current.tasks[c].slot;
            matrix(source.gate_up_packed, source.gate_up_absmax, source.gate_up_nested,
                2 * current.intermediate, current.hidden);
            matrix(source.down_packed, source.down_absmax, source.down_nested,
                current.hidden, current.intermediate);
        }
        if (!range_count) return;
        if (mmap_prefetch) {
            // Advisory physical-cache hint; failures leave ordinary reads intact.
            completed.prefetch_begin_ns = steady_ns();
            const BOOL accepted = PrefetchVirtualMemory(GetCurrentProcess(), range_count, ranges, 0);
            completed.prefetch_ns = steady_ns() - completed.prefetch_begin_ns;
            completed.prefetch_calls = 1;
            completed.prefetch_ranges = range_count;
            completed.prefetch_bytes = bytes;
            completed.prefetch_failures = accepted ? 0 : 1;
        }
        if (mmap_touch) {
            completed.touch_begin_ns = steady_ns();
            static const size_t page_size = [] {
                SYSTEM_INFO info{};
                GetSystemInfo(&info);
                return info.dwPageSize ? size_t(info.dwPageSize) : size_t(4096);
            }();
            for (ULONG_PTR i = 0; i < range_count; ++i) {
                const auto* start = static_cast<const uint8_t*>(ranges[i].VirtualAddress);
                const size_t length = ranges[i].NumberOfBytes;
                // Touch the leading page at the actual span start, never at its
                // rounded-down address (which may be outside short metadata).
                (void)*static_cast<const volatile uint8_t*>(start);
                ++completed.touch_pages;
                const size_t next_page = page_size - (reinterpret_cast<uintptr_t>(start) % page_size);
                for (size_t offset = next_page; offset < length; offset += page_size) {
                    (void)*static_cast<const volatile uint8_t*>(start + offset);
                    ++completed.touch_pages;
                }
            }
            completed.touch_ns = steady_ns() - completed.touch_begin_ns;
        }
#else
        (void)current; (void)completed;
#endif
    }
    void execute_compute(Job& current, Result& completed) {
        std::array<ops::cpu::CpuExpertTask, ops::cpu::kMaxExpertBatchTasks>
            original_tasks;
        if (verify_q8 || verify_a64)
            std::copy_n(current.tasks.begin(), current.count,
                        original_tasks.begin());
        arena.execute([&] {
            ops::cpu::CpuTilePoolScope poolScope(tilePool.get());
            if (!current.count) return;
            if (verify_a64 && !current.resident_prepared)
                throw std::invalid_argument("CPU A64 verification requires resident Activation64 tasks");
            if (batched || current.resident_prepared) {
                if (current.hidden != ops::cpu::kHiddenDim || current.intermediate != ops::cpu::kIntermediateDim)
                    throw std::invalid_argument("CPU batch requires real Q4E geometry");
                static const bool gpu_order = aila::env::read_flag("AILA_Q4E_CPU_GPU_ORDER", false);
                static const bool prepared = aila::env::read_flag("AILA_Q4E_CPU_PREPARED_INPUT", false);
                static const bool nf4_q8 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8", true);
                static const bool nf4_q8_block64 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8_BLOCK64", false);
                static const bool nf4_q8_row4 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8_ROW4", false);
                static const bool nf4_scale_shift = aila::env::read_flag("AILA_Q4E_CPU_SCALE_SHIFT", true);
                static const bool nf4_q8_activation64 = aila::env::read_flag("AILA_Q4E_CPU_Q8_ACTIVATION64", true);
                if (current.resident_prepared && nf4_q8 && nf4_q8_activation64 && verify_q8)
                    throw std::invalid_argument("Activation64 uses different Q8 arithmetic; H8 bitwise verification is incompatible");
                if (verify_a64 && (!nf4_q8 || !nf4_q8_activation64))
                    throw std::invalid_argument("CPU A64 verification requires the NF4Q8 Activation64 primary backend");
                if (current.resident_prepared && nf4_q8 && nf4_q8_activation64 && !activation64_logged) {
                    AILA_LOG_INFO("[Q4E CpuActivation64Enabled] activation_block=64 merged_integer_halves=1");
                    activation64_logged = true;
                }
                if (current.resident_prepared && nf4_q8 && nf4_scale_shift && !scale_shift_logged) {
                    AILA_LOG_INFO("[Q4E CpuScaleShiftEnabled] nested_block256=shift8 other_blocks=division");
                    scale_shift_logged = true;
                }
                const auto batch_backend = current.resident_prepared
                    ? (nf4_q8 ? (nf4_q8_activation64 ? ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2 : nf4_q8_block64 ? ops::cpu::CpuKernelBackend::Nf4Q8Block64Avx2 : nf4_q8_row4 ? ops::cpu::CpuKernelBackend::Nf4Q8Row4Avx2 : ops::cpu::CpuKernelBackend::Nf4Q8Avx2)
                              : ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2)
                    : (gpu_order ? (prepared ? ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2 : ops::cpu::CpuKernelBackend::GpuOrderAvx2)
                                 : ops::cpu::CpuKernelBackend::Avx2);
                if (current.group_count > 0) {
                    ops::cpu::cpu_expert_batch_forward_groups(current.groups.data(), current.group_count,
                        current.group_members.data(), current.tasks.data(), current.count,
                        current.x, current.tokens, current.h, current.out, nullptr, batch_backend,
                        diagnostic ? &completed.phases : nullptr, nf4_scale_shift);
                } else {
                    ops::cpu::cpu_expert_batch_forward(current.tasks.data(), current.count, current.x, current.tokens, current.h, current.out, nullptr,
                        batch_backend, diagnostic ? &completed.phases : nullptr, nf4_scale_shift);
                }
                if (verify_q8 && current.resident_prepared && nf4_q8) {
                    ops::cpu::CpuTilePoolScope originalPool(nullptr);
                    const auto* reference = current.reference_tasks ? current.reference_tasks :
                        original_tasks.data();
                    ops::cpu::cpu_expert_batch_forward(reference, current.count,
                        current.x, current.tokens, verify_h.data(), verify_out.data(), nullptr,
                        ops::cpu::CpuKernelBackend::Nf4Q8Avx2);
                    if (std::memcmp(current.h, verify_h.data(), current.count * current.intermediate * sizeof(bf16)) ||
                        std::memcmp(current.out, verify_out.data(), current.count * current.hidden * sizeof(float)))
                        throw std::runtime_error("CPU Q8 candidate differs bitwise from H8 on actual model activation");
                    completed.verify_batches = 1;
                }
                if (verify_a64) {
                    ops::cpu::CpuTilePoolScope originalPool(nullptr);
                    const auto* reference = current.reference_tasks ? current.reference_tasks : original_tasks.data();
                    ops::cpu::cpu_expert_batch_forward(reference, current.count,
                        current.x, current.tokens, verify_h.data(), verify_out.data(), nullptr,
                        ops::cpu::CpuKernelBackend::Nf4Q8Activation64Avx2, nullptr, false, false, false, false);
                    if (std::memcmp(current.h, verify_h.data(), current.count * current.intermediate * sizeof(bf16)))
                        throw std::runtime_error("CPU A64 candidate differs bitwise from original A64 BF16 h on actual model activation");
                    if (std::memcmp(current.out, verify_out.data(), current.count * current.hidden * sizeof(float)))
                        throw std::runtime_error("CPU A64 candidate differs bitwise from original A64 F32 out on actual model activation");
                    completed.a64_verify_batches = 1;
                }
            } else for (int c = 0; c < current.count; ++c)
                compute_expert(current.tasks[c].slot, current.x + current.tasks[c].token * current.hidden,
                    current.h + c * current.intermediate, current.out + c * current.hidden,
                    current.hidden, current.intermediate);
        });
    }
    void execute_direct(Job current, Result& completed) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (busy || pending)
                throw std::logic_error("CPU direct execution requires an idle worker");
        }
        prepare_mapped_sources(current, completed);
        prepare_cached_scales(current, completed);
        execute_compute(current, completed);
    }
    void run() {
        if (ops::cpu::TilePoolPinningEnabled()) {
            ops::cpu::PinCurrentThreadToCore(ops::cpu::PhysicalCoreCount() >= 4 ? 2 : -1);
        }
        for (;;) {
            Job current;
            if (spinUs) spin_until(pendingSignal);
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return pending || stop; });
                if (stop) return;
                current = job; pending = false;
                if (spinUs) pendingSignal.store(false, std::memory_order_relaxed);
            }
            Result completed;
            try {
                if (current.route_ids) {
                    completed.route_begin_ns = steady_ns();
                    current.route_event.wait_and_throw();
                    completed.route_end_ns = steady_ns();
                    // The classify event publishes both pinned arrays. Keep the
                    // original rank order, just as StartResidentCold does.
                    for (int rank = 0; rank < current.tokens * current.route_top_k; ++rank) {
                        const int id = current.route_ids[rank];
                        if (id < 0 || id >= current.route_experts)
                            throw std::logic_error("Invalid asynchronous resident route expert");
                        if ((current.route_selected && current.route_selected[rank] >= 0) || !current.route_cpu_eligible[id]) continue;
                        const auto& source = current.route_sources[id];
                        if (!source.gate_up_packed || !source.down_packed) continue;
                        const int c = current.count++;
                        current.tasks[c] = {source,rank / current.route_top_k};
                        current.mapped_tasks[c] = !current.route_sources_resident;
                        completed.route_ranks[c] = rank;
                        completed.route_ids[c] = id;
                    }
                    completed.route_count = current.count;
                }
                prepare_mapped_sources(current, completed);
                // Preserve immutable nested refs before rebinding only scale
                // metadata. Async route jobs have no coordinator reference array.
                std::array<ops::cpu::CpuExpertTask, ops::cpu::kMaxExpertBatchTasks> original_tasks;
                if (verify_q8 || verify_a64)
                    std::copy_n(current.tasks.begin(), current.count, original_tasks.begin());
                prepare_cached_scales(current, completed);
                const auto begin = steady_ns();
                if (current.activation != sycl::event{}) current.activation.wait_and_throw();
                completed.activation_ns = steady_ns() - begin;
                // Host direct-read merge may still own this output arena. Wait
                // on the worker, after route/input publication, never on the
                // enqueue thread that is submitting the next layer's GPU work.
                if (current.output_reader != sycl::event{}) current.output_reader.wait_and_throw();
                if (current.transfer_reader != sycl::event{}) current.transfer_reader.wait_and_throw();
                const auto before = diagnostic ? sample_cpu_process() : CpuProcessSample{};
                completed.begin_ns = steady_ns();
                execute_compute(current, completed);
                completed.end_ns = steady_ns();
                if (diagnostic && current.count) {
                    const auto after = sample_cpu_process();
                    completed.diagnostic_batches = 1;
                    if (before.valid && after.valid && after.user_ns >= before.user_ns && after.kernel_ns >= before.kernel_ns) {
                        completed.process_faults = uint32_t(after.faults - before.faults);
                        completed.process_user_ns = after.user_ns - before.user_ns;
                        completed.process_kernel_ns = after.kernel_ns - before.kernel_ns;
                    } else completed.diagnostic_failures = 1;
                }
            } catch (...) { completed.error = std::current_exception(); }
            { std::lock_guard<std::mutex> lock(mutex); result = completed; done = true;
              if (spinUs) doneSignal.store(true, std::memory_order_release); }
            cv.notify_one();
        }
    }
};

Qwen4ExpExpertExecutor::Qwen4ExpExpertExecutor() = default;
bool Qwen4ExpExpertExecutor::cpu_work_complete_for_testing() const {
    return cpu_worker_ && cpu_worker_->complete();
}

bool Qwen4ExpExpertExecutor::CopyColdStagingHostSlots(
    const uint8_t* const* sources, uint8_t* destination, size_t slot_bytes,
    int count, bool allow_parallel) const {
    if (!sources || !destination || !slot_bytes || count < 1 || count > 16)
        throw std::invalid_argument("Invalid Qwen4Exp staging host-copy batch");
    for (int i = 0; i < count; ++i)
        if (!sources[i]) throw std::invalid_argument("Qwen4Exp staging source is null");

    const auto copy_one = [&](int i) {
        std::memcpy(destination + size_t(i) * slot_bytes, sources[i], slot_bytes);
    };
    if (!allow_parallel || count == 1 || !cpu_worker_ || cpu_worker_->tilePool ||
        cpu_worker_->arena.max_concurrency() <= 1) {
        for (int i = 0; i < count; ++i) copy_one(i);
        return false;
    }

    // This is the same bounded arena already used by CpuWorker::run. Do not
    // create a second scheduler; each task owns a disjoint destination slot.
    cpu_worker_->arena.execute([&] {
        tbb::parallel_for(tbb::blocked_range<int>(0, count, 1),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i != range.end(); ++i) copy_one(i);
            });
    });
    return true;
}

Qwen4ExpExpertExecutor::~Qwen4ExpExpertExecutor() {
    cleanup();
}

void Qwen4ExpExpertExecutor::cleanup() {
    cpu_worker_.reset();
    cold_source_slots_={}; cold_count_=0; cold_mapped_count_=0; cold_cached_count_=0;
    arena_verify_pending_=false;
    arena_cold_rows_pending_=false; arena_cold_rows_joined_=false;
    arena_cold_rows_=0; arena_cold_rows_next_row_=0;
    if (ctx_) {
        try { ctx_->synchronize(); } catch (...) {}
    }
    if (ctx_) {
        if (x_host_) { ctx_->free_host(x_host_); x_host_ = nullptr; }
        if (cpu_out_host_) { ctx_->free_host(cpu_out_host_); cpu_out_host_ = nullptr; }
        if (cpu_result_packet_host_) { ctx_->free_host(cpu_result_packet_host_); cpu_result_packet_host_=nullptr; }
        if (cpu_result_packet_dev_) { ctx_->free_device(cpu_result_packet_dev_); cpu_result_packet_dev_=nullptr; }
        if (scratch_h_) { ctx_->free_host(scratch_h_); scratch_h_ = nullptr; }
    }
    initialized_ = false;
    cpu_result_packet_active_=false;
    cpu_host_result_active_=false;cpu_host_rows_={};cpu_result_host_read_=false;
    cpu_output_reader_event_={};
}

void Qwen4ExpExpertExecutor::init(
    Context& ctx,
    int max_seq_len,
    int top_k,
    int hidden_size,
    int intermediate_size) {
    cleanup();

    ctx_ = &ctx;
    max_seq_len_ = std::max(1, max_seq_len);
    top_k_ = std::max(1, top_k);
    hidden_size_ = hidden_size > 0 ? hidden_size : 2560;
    intermediate_size_ = intermediate_size > 0 ? intermediate_size : 640;

    const int env_max_cpu = aila::env::read_int_raw("AILA_Q4E_CPU_MAX_EXPERTS", 10);
    max_cpu_credits_ = std::clamp(env_max_cpu, 0, ops::cpu::kMaxExpertBatchTasks);
    const int env_max_tokens = aila::env::read_int("AILA_Q4E_CPU_MAX_TOKENS", 4);
    max_cpu_tokens_ = std::max(1, std::min(4, env_max_tokens));

    static const bool result_packet=aila::env::read_flag("AILA_Q4E_CPU_RESULT_PACKET",false);
    static const bool result_host_read=aila::env::read_flag("AILA_Q4E_CPU_RESULT_HOST_READ",!result_packet);
    if (result_packet && result_host_read)
        throw std::invalid_argument("CPU_RESULT_HOST_READ and CPU_RESULT_PACKET are mutually exclusive");
    cpu_result_host_read_=result_host_read && top_k_>0 && top_k_<=16 &&
        ctx_->queue().get_device().has(sycl::aspect::usm_host_allocations);
    if(cpu_result_host_read_) AILA_LOG_INFO("[Q4E CpuResultHostRead] enabled=1 resident_b1=1 topk=%d merge_reader_fence=1",top_k_);
    x_host_ = static_cast<bf16*>(ctx_->alloc_host(4ULL * hidden_size_ * sizeof(bf16)));
    cpu_out_host_ = static_cast<float*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * hidden_size_ * sizeof(float)));
    scratch_h_ = static_cast<bf16*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * intermediate_size_ * sizeof(bf16)));
    if(result_packet && top_k_<=16) {
        const size_t bytes=16*sizeof(int)+16ULL*hidden_size_*sizeof(float);
        cpu_result_packet_host_=ctx_->alloc_host(bytes);
        cpu_result_packet_dev_=ctx_->alloc_device(bytes);
    }

    prev_cpu_h2d_ev_ = sycl::event{};

    stats_.reset();
    straggler_penalty_active_ = false;
    cpu_retry_countdown_ = 0;
    if (aila::env::read_flag("AILA_Q4E_CPU_OVERLAP", true) || aila::env::read_int("AILA_Q4E_COLD_MODE",0)>0) {
        const int threads = std::clamp(
            aila::env::read_int("AILA_Q4E_CPU_THREADS", std::clamp(ops::cpu::PhysicalCoreCount() - 2, 1, 16)),
            1, 18);
        cpu_worker_ = std::make_unique<CpuWorker>(threads);
    }
    initialized_ = true;
}

sycl::queue* Qwen4ExpExpertExecutor::copy_queue() const {
    if (!q4e::moe_copy_via_copyq()) return nullptr;
    sycl::queue* q = copy_q_override_;
    static std::atomic<bool> logged{false};
    if (!logged.load(std::memory_order_relaxed) &&
        !logged.exchange(true, std::memory_order_relaxed)) {
        if (q) {
            AILA_LOG_INFO("[Q4E MoeCopyViaCopyQ] enabled=1 source=owner "
                          "host_copies=x_d2h+cpu_result_h2d (compute-queue barrier ordered)");
        } else {
            AILA_LOG_INFO("[Q4E MoeCopyViaCopyQ] enabled=1 but no copy queue is "
                          "available; host copies stay on the compute queue");
        }
    }
    return q;
}

void Qwen4ExpExpertExecutor::StageResidentInput(const bf16* x_dev, int rows,
                                                const sycl::event* producer) {
    if(rows<1 || rows>max_cpu_tokens_ || cold_count_ || arena_verify_pending_ || cpu_host_result_active_)throw std::logic_error("CPU cold slot has an unfinished batch or invalid rows");
    static const bool deferArenaWait = aila::env::read_flag("AILA_Q4E_CPU_DEFER_ARENA_WAIT", false);
    if(!deferArenaWait && resident_input_event_!=sycl::event{})resident_input_event_.wait_and_throw();
    resident_input_event_=q4e::submit_host_copy(*ctx_,copy_queue(),x_host_,x_dev,
        size_t(rows)*hidden_size_*sizeof(bf16),producer);
}

int Qwen4ExpExpertExecutor::StartArenaCold(const Qwen4ExpHostArena& arena, int layer,
    const int* ids, const int* ranks, int count, const bf16* x_dev, int row_token, int tokens) {
    if (!cpu_worker_ || count<=0) return 0;
    if (!ids || !ranks || count>16 || tokens<1 || tokens>max_cpu_tokens_ ||
        row_token<0 || row_token>=tokens || cold_count_ ||
        arena_verify_pending_ || cpu_host_result_active_)
        throw std::logic_error("Invalid native arena CPU batch");
    if (tokens >= 2) {
        static const bool s_merge = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_MERGE", false);
        if (s_merge && !merge_inactive_logged_) {
            AILA_LOG_INFO("[Q4E VerifyCpuMerge] inactive: window job requires AILA_Q4E_VERIFY_CPU_BATCH=1");
            merge_inactive_logged_ = true;
        }
    }
    try {
        const bool defer_transfer_wait =
            q4e::moe_copy_via_copyq() && copy_queue() != nullptr;
        sycl::event transfer_reader{};
        if (defer_transfer_wait) {
            transfer_reader = prev_cpu_h2d_ev_;
        } else if (prev_cpu_h2d_ev_!=sycl::event{}) {
            prev_cpu_h2d_ev_.wait_and_throw();
            prev_cpu_h2d_ev_={};
        }
        cold_layer_=layer;
        cold_count_=std::min(count,max_cpu_credits_);
        cold_cached_count_=cold_count_; cold_mapped_count_=0;
        if (!cold_count_) return 0;
        CpuWorker::Job job;
        job.count=cold_count_; job.hidden=hidden_size_; job.intermediate=intermediate_size_; job.tokens=tokens;
        job.x=x_host_; job.h=scratch_h_; job.out=cpu_out_host_; job.resident_prepared=true;
        job.output_reader=cpu_output_reader_event_;
        job.transfer_reader=transfer_reader;
        job.activation=resident_input_event_!=sycl::event{} ? resident_input_event_ :
            (x_dev ? q4e::submit_host_copy(*ctx_,copy_queue(),x_host_,x_dev,
                size_t(hidden_size_)*size_t(tokens)*sizeof(bf16)) : sycl::event{});
        for (int c=0;c<cold_count_;++c) {
            const auto source=arena.GetExpertSlotHost(layer,ids[c]);
            if (!source.gate_up_packed || !source.down_packed || ranks[c]<0 || ranks[c]>=top_k_)
                throw std::logic_error("CPU expert has no native RAM owner or rank");
            cold_source_slots_[c]=source; cold_ids_[c]=ids[c]; cold_ranks_[c]=ranks[c];
            job.tasks[c]={source,row_token};
        }
        cold_submit_ns_=steady_ns(); cpu_worker_->submit(job);
        if (defer_transfer_wait) prev_cpu_h2d_ev_={};
        stats_.cpu_tasks_dispatched+=cold_count_;
        stats_.cpu_cached_tasks_dispatched+=cold_count_;
        return cold_count_;
    } catch (...) { AbortResidentCold(); throw; }
}

int Qwen4ExpExpertExecutor::RunGraphColdPacket(
    const Qwen4ExpHostArena& arena, int layer, const int* ids,
    const int* ranks, int count, const bf16* x_host, int* rank_to_row,
    float* packet_rows) {
    if (!initialized_ || !cpu_worker_ || !ids || !ranks || !x_host ||
        !rank_to_row || !packet_rows)
        throw std::logic_error("Invalid graph CPU packet request");
    if (count < 1 || count > 16 || count > max_cpu_credits_)
        throw std::logic_error("Graph CPU packet count exceeds fixed capacity");
    if (cold_count_ || arena_verify_pending_ || cpu_host_result_active_)
        throw std::logic_error("Graph CPU packet requires an idle executor");

    std::memcpy(
        x_host_, x_host,
        static_cast<std::size_t>(hidden_size_) * sizeof(bf16));

    CpuWorker::Job job;
    job.count = count;
    job.hidden = hidden_size_;
    job.intermediate = intermediate_size_;
    job.tokens = 1;
    job.x = x_host_;
    job.h = scratch_h_;
    job.out = packet_rows;
    job.resident_prepared = true;
    std::fill_n(rank_to_row, 16, -1);
    for (int c = 0; c < count; ++c) {
        const auto source = arena.GetExpertSlotHost(layer, ids[c]);
        if (!source.gate_up_packed || !source.down_packed ||
            ranks[c] < 0 || ranks[c] >= top_k_)
            throw std::logic_error(
                "Graph CPU expert has no native RAM owner or rank");
        job.tasks[c] = {source, 0};
        rank_to_row[ranks[c]] = c;
    }
    CpuWorker::Result result;
    cpu_worker_->execute_direct(job, result);
    if (result.error) std::rethrow_exception(result.error);
    return count;
}

sycl::event Qwen4ExpExpertExecutor::JoinResidentCold(float* output, const sycl::event* order_dep) {
    cpu_result_packet_active_=false;
    if(!cold_count_)return {};
    const auto join=steady_ns();
    const auto result=cpu_worker_->finish();
    stats_.cpu_join_wait_ns+=steady_ns()-join;
    stats_.cpu_activation_wait_ns+=result.activation_ns;
    stats_.cpu_mmap_prefetch_ns+=result.prefetch_ns;
    stats_.cpu_mmap_prefetch_calls+=result.prefetch_calls;
    stats_.cpu_mmap_prefetch_ranges+=result.prefetch_ranges;
    stats_.cpu_mmap_prefetch_bytes+=result.prefetch_bytes;
    stats_.cpu_mmap_prefetch_failures+=result.prefetch_failures;
    stats_.cpu_mmap_touch_ns+=result.touch_ns;
    stats_.cpu_mmap_touch_pages+=result.touch_pages;
    stats_.cpu_scale_cache_bytes=std::max(stats_.cpu_scale_cache_bytes,result.scale_cache_bytes);
    stats_.cpu_scale_cache_hits+=result.scale_cache_hits;
    stats_.cpu_scale_cache_misses+=result.scale_cache_misses;
    stats_.cpu_scale_cache_fills+=result.scale_cache_fills;
    stats_.cpu_scale_cache_evictions+=result.scale_cache_evictions;
    stats_.cpu_scale_cache_fallbacks+=result.scale_cache_fallbacks;
    stats_.cpu_scale_cache_prepare_ns+=result.scale_cache_prepare_ns;
    stats_.cpu_diagnostic_batches+=result.diagnostic_batches;
    stats_.cpu_verify_batches+=result.verify_batches;
    stats_.cpu_a64_verify_batches+=result.a64_verify_batches;
    stats_.cpu_diagnostic_failures+=result.diagnostic_failures;
    stats_.cpu_process_faults+=result.process_faults;
    stats_.cpu_process_user_ns+=result.process_user_ns;
    stats_.cpu_process_kernel_ns+=result.process_kernel_ns;
    stats_.cpu_gate_up_ns+=result.phases.gate_up_ns;
    stats_.cpu_quantize_ns+=result.phases.quantize_ns;
    stats_.cpu_down_ns+=result.phases.down_ns;
    stats_.cpu_gate_chunks+=result.phases.gate_chunks;
    stats_.cpu_down_chunks+=result.phases.down_chunks;
    stats_.cpu_compute_ns+=result.end_ns-result.begin_ns;
    const uint64_t before_compute_ns=result.begin_ns-cold_submit_ns_;
    stats_.cold_queue_ns+=before_compute_ns-std::min(result.activation_ns,before_compute_ns);
    for(int c=0;c<cold_count_;++c)if(cpu_observer_) {
        RouteRankEntry entry;entry.rank=cold_ranks_[c];entry.expert_id=cold_ids_[c];entry.is_gpu=false;
        cpu_observer_(cpu_observer_state_,*ctx_,cold_layer_,entry,cold_source_slots_[c],x_host_,
            scratch_h_+size_t(c)*intermediate_size_,cpu_out_host_+size_t(c)*hidden_size_);
    }
    if(cpu_result_host_read_) {
        cpu_host_rows_.values=cpu_out_host_;cpu_host_rows_.rows=cold_count_;
        cpu_host_rows_.rank_to_row.fill(-1);
        for(int c=0;c<cold_count_;++c)cpu_host_rows_.rank_to_row[cold_ranks_[c]]=c;
        cpu_host_result_active_=true;
    } else if(cpu_result_packet_host_) {
        auto* ranks=static_cast<int*>(cpu_result_packet_host_);
        std::fill_n(ranks,16,-1);
        for(int c=0;c<cold_count_;++c)ranks[cold_ranks_[c]]=c;
        const size_t payload=size_t(cold_count_)*hidden_size_*sizeof(float);
        std::memcpy(ranks+16,cpu_out_host_,payload);
        const size_t bytes=16*sizeof(int)+payload;
        prev_cpu_h2d_ev_=q4e::submit_host_copy(*ctx_,copy_queue(),
            cpu_result_packet_dev_,cpu_result_packet_host_,bytes,order_dep);
        ++stats_.cpu_result_copies;stats_.cpu_result_bytes+=bytes;
        cpu_result_packet_active_=true;
    } else {
        sycl::queue* copy_q=copy_queue();
        sycl::event order_fence{};
        const sycl::event* order_fence_ptr=nullptr;
        if(order_dep){order_fence_ptr=order_dep;}
        else if(copy_q){order_fence=q4e::host_copy_fence(*ctx_,copy_q);order_fence_ptr=&order_fence;}
        for(int c=0;c<cold_count_;) {
            int end=c+1;while(end<cold_count_ && cold_ranks_[end]==cold_ranks_[end-1]+1)++end;
            const size_t bytes=size_t(end-c)*hidden_size_*sizeof(float);
            prev_cpu_h2d_ev_=q4e::submit_host_copy(*ctx_,copy_q,
                output+size_t(cold_ranks_[c])*hidden_size_,
                cpu_out_host_+size_t(c)*hidden_size_,bytes,order_fence_ptr);
            ++stats_.cpu_result_copies;stats_.cpu_result_bytes+=bytes;c=end;
        }
    }
    stats_.cpu_tasks_completed+=cold_count_;++stats_.cpu_async_layers;
    stats_.cpu_mapped_tasks_completed+=cold_mapped_count_;
    stats_.cpu_cached_tasks_completed+=cold_cached_count_;
    stats_.cpu_max_batch_tasks=std::max(stats_.cpu_max_batch_tasks,uint64_t(cold_count_));
    cold_source_slots_={};cold_count_=0;cold_mapped_count_=0;cold_cached_count_=0;
    return prev_cpu_h2d_ev_;
}

bool Qwen4ExpExpertExecutor::CanExecuteArenaColdRows(int rows) const {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_BATCH", false);
    return enabled && initialized_ && cpu_worker_ && rows >= 2 && rows <= 4 &&
        rows <= max_cpu_tokens_ && hidden_size_ == ops::cpu::kHiddenDim &&
        intermediate_size_ == ops::cpu::kIntermediateDim &&
        !cpu_result_host_read_ && !cpu_result_packet_host_ &&
        !cold_count_ && !arena_verify_pending_ && !arena_cold_rows_pending_;
}

int Qwen4ExpExpertExecutor::StartArenaColdRows(const Qwen4ExpHostArena& arena, int layer, int rows,
    const int* ids, const int* ranks, const int* row_task_begin, const int* row_task_count,
    int task_count, const bf16* x_dev) {
    if (!CanExecuteArenaColdRows(rows))
        throw std::logic_error("Invalid multi-row native arena CPU batch");
    if (!ids || !ranks || !row_task_begin || !row_task_count || !x_dev ||
        task_count < 1 || task_count > ops::cpu::kMaxExpertBatchTasks ||
        layer < 0 || layer >= arena.NumLayers())
        throw std::logic_error("Invalid multi-row native arena CPU batch");
    try {
        if (prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
        prev_cpu_h2d_ev_ = {};
        cold_layer_ = layer;
        cold_count_ = task_count;
        cold_cached_count_ = task_count; cold_mapped_count_ = 0;
        CpuWorker::Job job;
        job.count = task_count; job.hidden = hidden_size_; job.intermediate = intermediate_size_;
        job.tokens = rows;
        job.x = x_host_; job.h = scratch_h_; job.out = cpu_out_host_; job.resident_prepared = true;
        job.output_reader = cpu_output_reader_event_;
        job.activation = resident_input_event_ != sycl::event{} ? resident_input_event_ :
            q4e::submit_host_copy(*ctx_, copy_queue(), x_host_, x_dev,
                size_t(hidden_size_) * size_t(rows) * sizeof(bf16));
        int offset = 0;
        for (int r = 0; r < rows; ++r) {
            const int begin = row_task_begin[r], count = row_task_count[r];
            if (begin != offset || count < 0 || begin + count > task_count)
                throw std::logic_error("Multi-row CPU task spans are not a strict partition");
            arena_cold_row_begin_[r] = begin; arena_cold_row_count_[r] = count;
            for (int c = 0; c < count; ++c) {
                const int i = begin + c;
                const auto source = arena.GetExpertSlotHost(layer, ids[i]);
                if (!source.gate_up_packed || !source.down_packed || ranks[i] < 0 || ranks[i] >= top_k_)
                    throw std::logic_error("CPU expert has no native RAM owner or rank");
                cold_source_slots_[i] = source; cold_ids_[i] = ids[i]; cold_ranks_[i] = ranks[i];
                job.tasks[i] = {source, r};
            }
            offset += count;
        }
        if (offset != task_count)
            throw std::logic_error("Multi-row CPU task spans are not a strict partition");
        if (task_count > 1) {
            static const bool s_merge = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_MERGE", false);
            if (s_merge) {
                std::array<int, ops::cpu::kMaxExpertBatchTasks> pair_group{}, group_ids{}, group_counts{};
                int groups = 0;
                for (int i = 0; i < task_count; ++i) {
                    int g = -1;
                    for (int k = 0; k < groups; ++k)
                        if (group_ids[k] == cold_ids_[i]) { g = k; break; }
                    if (g < 0) { g = groups++; group_ids[g] = cold_ids_[i]; group_counts[g] = 0; }
                    pair_group[i] = g; ++group_counts[g];
                }
                bool mergeable = groups < task_count;
                for (int g = 0; g < groups && mergeable; ++g)
                    mergeable = group_counts[g] <= 4;
                if (mergeable) {
                    std::array<int, ops::cpu::kMaxExpertBatchTasks> group_begin{};
                    int cursor = 0;
                    for (int g = 0; g < groups; ++g) { group_begin[g] = cursor; cursor += group_counts[g]; }
                    std::array<int, ops::cpu::kMaxExpertBatchTasks> fill = group_begin;
                    for (int i = 0; i < task_count; ++i) {
                        const int g = pair_group[i];
                        job.group_members[fill[g]++] = i;
                    }
                    for (int g = 0; g < groups; ++g) {
                        job.groups[g].slot = cold_source_slots_[job.group_members[group_begin[g]]];
                        job.groups[g].member_begin = group_begin[g];
                        job.groups[g].member_count = group_counts[g];
                    }
                    job.group_count = groups;
                }
                merge_stat_pairs_ += static_cast<uint64_t>(task_count);
                merge_stat_groups_ += static_cast<uint64_t>(groups);
                ++merge_stat_builds_;
                if (merge_stat_builds_ <= 4 || (merge_stat_builds_ % 48) == 0) {
                    const double window_f = 1.0 - static_cast<double>(groups) / static_cast<double>(task_count);
                    const double cum_f = merge_stat_pairs_ > 0
                        ? 1.0 - static_cast<double>(merge_stat_groups_) / static_cast<double>(merge_stat_pairs_)
                        : 0.0;
                    AILA_LOG_INFO("[Q4E VerifyCpuMerge] build=%llu layer=%d pairs=%d groups=%d f=%.4f cum_pairs=%llu cum_groups=%llu cum_f=%.4f",
                        static_cast<unsigned long long>(merge_stat_builds_), layer, task_count, groups, window_f,
                        static_cast<unsigned long long>(merge_stat_pairs_),
                        static_cast<unsigned long long>(merge_stat_groups_), cum_f);
                }
            }
        }
        arena_cold_rows_ = rows; arena_cold_rows_next_row_ = 0;
        arena_cold_rows_joined_ = false; arena_cold_rows_pending_ = true;
        cold_submit_ns_ = steady_ns();
        cpu_worker_->submit(job);
        stats_.cpu_tasks_dispatched += task_count;
        stats_.cpu_cached_tasks_dispatched += task_count;
        if (!arena_cold_rows_logged_) {
            AILA_LOG_INFO("[Q4E ArenaColdRowsEnabled] rows=%d tasks=%d capacity=%d",
                rows, task_count, ops::cpu::kMaxExpertBatchTasks);
            arena_cold_rows_logged_ = true;
        }
        return task_count;
    } catch (...) { AbortResidentCold(); throw; }
}

sycl::event Qwen4ExpExpertExecutor::JoinArenaColdRowsRow(float* rank_output, int row,
                                                        const sycl::event* order_dep) {
    if (!rank_output || !arena_cold_rows_pending_ || row != arena_cold_rows_next_row_ ||
        row < 0 || row >= arena_cold_rows_)
        throw std::logic_error("Invalid multi-row native arena CPU join");
    if (!arena_cold_rows_joined_) {
        const auto join = steady_ns();
        const auto result = cpu_worker_->finish();
        stats_.cpu_join_wait_ns += steady_ns() - join;
        stats_.cpu_activation_wait_ns += result.activation_ns;
        stats_.cpu_compute_ns += result.end_ns - result.begin_ns;
        stats_.cpu_mmap_prefetch_ns += result.prefetch_ns;
        stats_.cpu_mmap_touch_ns += result.touch_ns;
        stats_.cpu_scale_cache_bytes = std::max(stats_.cpu_scale_cache_bytes, result.scale_cache_bytes);
        stats_.cpu_scale_cache_hits += result.scale_cache_hits;
        stats_.cpu_scale_cache_misses += result.scale_cache_misses;
        stats_.cpu_scale_cache_prepare_ns += result.scale_cache_prepare_ns;
        stats_.cpu_diagnostic_batches += result.diagnostic_batches;
        stats_.cpu_diagnostic_failures += result.diagnostic_failures;
        stats_.cpu_verify_batches += result.verify_batches;
        stats_.cpu_a64_verify_batches += result.a64_verify_batches;
        stats_.cpu_process_faults += result.process_faults;
        stats_.cpu_process_user_ns += result.process_user_ns;
        stats_.cpu_process_kernel_ns += result.process_kernel_ns;
        stats_.cpu_gate_up_ns += result.phases.gate_up_ns;
        stats_.cpu_quantize_ns += result.phases.quantize_ns;
        stats_.cpu_down_ns += result.phases.down_ns;
        stats_.cpu_gate_chunks += result.phases.gate_chunks;
        stats_.cpu_down_chunks += result.phases.down_chunks;
        for (int c = 0; c < cold_count_; ++c) if (cpu_observer_) {
            int task_row = 0;
            while (task_row + 1 < arena_cold_rows_ &&
                   c >= arena_cold_row_begin_[task_row] + arena_cold_row_count_[task_row]) ++task_row;
            RouteRankEntry entry;
            entry.token_idx = task_row; entry.rank = cold_ranks_[c];
            entry.expert_id = cold_ids_[c]; entry.is_gpu = false;
            cpu_observer_(cpu_observer_state_, *ctx_, cold_layer_, entry, cold_source_slots_[c],
                x_host_ + size_t(task_row) * hidden_size_,
                scratch_h_ + size_t(c) * intermediate_size_, cpu_out_host_ + size_t(c) * hidden_size_);
        }
        arena_cold_rows_joined_ = true;
    }
    const int begin = arena_cold_row_begin_[row], count = arena_cold_row_count_[row];
    sycl::event upload{};
    sycl::queue* row_copy_q = copy_queue();
    sycl::event row_fence{};
    const sycl::event* row_fence_ptr = nullptr;
    if (order_dep) row_fence_ptr = order_dep;
    else if (row_copy_q) { row_fence = q4e::host_copy_fence(*ctx_, row_copy_q); row_fence_ptr = &row_fence; }
    for (int c = begin; c < begin + count;) {
        int end = c + 1;
        while (end < begin + count && cold_ranks_[end] == cold_ranks_[end - 1] + 1) ++end;
        const size_t bytes = size_t(end - c) * hidden_size_ * sizeof(float);
        upload = q4e::submit_host_copy(*ctx_, row_copy_q,
            rank_output + size_t(cold_ranks_[c]) * hidden_size_,
            cpu_out_host_ + size_t(c) * hidden_size_, bytes, row_fence_ptr);
        ++stats_.cpu_result_copies; stats_.cpu_result_bytes += bytes;
        c = end;
    }
    if (upload != sycl::event{}) prev_cpu_h2d_ev_ = upload;
    ++arena_cold_rows_next_row_;
    if (row == arena_cold_rows_ - 1) {
        stats_.cpu_tasks_completed += cold_count_; ++stats_.cpu_async_layers;
        stats_.cpu_max_batch_tasks = std::max(stats_.cpu_max_batch_tasks, uint64_t(cold_count_));
        cold_source_slots_ = {};
        cold_count_ = 0; cold_mapped_count_ = 0; cold_cached_count_ = 0;
        arena_cold_rows_pending_ = false;
        arena_cold_rows_ = 0; arena_cold_rows_next_row_ = 0;
    }
    return upload;
}

void Qwen4ExpExpertExecutor::FinishResidentCpuRead(const sycl::event& merge) {
    if(!cpu_host_result_active_) return;
    if(merge==sycl::event{})throw std::logic_error("CPU host result requires an actual merge reader event");
    cpu_output_reader_event_=merge;
    cpu_host_result_active_=false;
}

void Qwen4ExpExpertExecutor::AbortResidentCold() noexcept {
    if(cpu_worker_)cpu_worker_->drain();
    try {if(ctx_)ctx_->synchronize();}catch(...){}
    cold_source_slots_={};cold_count_=0;cold_mapped_count_=0;cold_cached_count_=0;
    cpu_result_packet_active_=false;
    cpu_host_result_active_=false;cpu_host_rows_={};cpu_output_reader_event_={};
    arena_verify_pending_=false;
    arena_cold_rows_pending_=false;arena_cold_rows_joined_=false;
    arena_cold_rows_=0;arena_cold_rows_next_row_=0;
}

bool Qwen4ExpExpertExecutor::CanExecuteArenaVerifyRows(const Qwen4ExpHostArena& arena, int rows, int top_k) const {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_ARENA_VERIFY_ROWS", false);
    return enabled && initialized_ && cpu_worker_ && rows >= 2 && rows <= 4 && rows <= max_cpu_tokens_ &&
        top_k == top_k_ && top_k <= 10 && rows * top_k <= ops::cpu::kMaxExpertBatchTasks &&
        hidden_size_ == ops::cpu::kHiddenDim && intermediate_size_ == ops::cpu::kIntermediateDim &&
        arena.IsAllocated() && arena.NumLayers() > 0 && arena.NumExpertsPerLayer() > 0 &&
        arena.NumExpertsPerLayer() <= 512 &&
        !cold_count_ && !arena_verify_pending_ &&
        !cpu_result_packet_host_ && !cpu_result_host_read_ && !cpu_host_result_active_;
}

int Qwen4ExpExpertExecutor::StartArenaVerifyRows(const Qwen4ExpHostArena& arena, int layer, int rows,
    const int* ids, const sycl::event& route_event) {
    if (!CanExecuteArenaVerifyRows(arena, rows, top_k_) || !ids || layer < 0 || layer >= arena.NumLayers() ||
        resident_input_event_ == sycl::event{})
        throw std::logic_error("Invalid arena multi-row verification route");
    const int experts = arena.NumExpertsPerLayer();
    try {
        static const bool deferArenaWait = aila::env::read_flag("AILA_Q4E_CPU_DEFER_ARENA_WAIT", false);
        const sycl::event transferReader = deferArenaWait ? prev_cpu_h2d_ev_ : sycl::event{};
        if (!deferArenaWait && prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
        prev_cpu_h2d_ev_ = {};
        cold_layer_ = layer; cold_source_slots_ = {}; cold_mapped_count_ = 0; cold_cached_count_ = 0;
        arena_verify_sources_.resize(size_t(experts));
        int predicted = 0;
        for (int e = 0; e < experts; ++e) {
            arena_verify_sources_[size_t(e)] = arena.GetExpertSlotHost(layer, e);
        }
        CpuWorker::Job job;
        job.hidden = hidden_size_; job.intermediate = intermediate_size_; job.tokens = rows;
        job.x = x_host_; job.h = scratch_h_; job.out = cpu_out_host_; job.resident_prepared = true;
        job.activation = resident_input_event_; job.route_event = route_event;
        job.output_reader = cpu_output_reader_event_; job.transfer_reader = transferReader;
        job.route_ids = ids; job.route_selected = nullptr;
        job.route_top_k = top_k_; job.route_experts = experts;
        job.route_sources_resident = true;
        job.route_sources = arena_verify_sources_.data();
        for (int e = 0; e < experts; ++e)
            job.route_cpu_eligible[e] = arena.GetExpertPointer(layer, e) != nullptr;
        for (int rank = 0; rank < rows * top_k_; ++rank) {
            const int id = ids[rank];
            if (id >= 0 && id < experts && job.route_cpu_eligible[id] &&
                arena_verify_sources_[size_t(id)].gate_up_packed &&
                arena_verify_sources_[size_t(id)].down_packed) ++predicted;
        }
        cold_submit_ns_ = steady_ns();
        arena_verify_pending_ = true;
        arena_verify_arena_ = &arena;
        cpu_worker_->submit(job);
        if (!arena_verify_logged_) {
            AILA_LOG_INFO("[Q4E ArenaVerifyRowsEnabled] layer=%d rows=%d topk=%d experts=%d capacity=%d",
                layer, rows, top_k_, experts, ops::cpu::kMaxExpertBatchTasks);
            arena_verify_logged_ = true;
        }
        return predicted;
    } catch (...) { AbortResidentCold(); throw; }
}

sycl::event Qwen4ExpExpertExecutor::JoinArenaVerifyRows(float* rank_output) {
    if (!rank_output) throw std::invalid_argument("Arena verification needs a rank output buffer");
    if (!arena_verify_pending_) return {};
    const auto join = steady_ns();
    const auto result = cpu_worker_->finish();
    cold_count_ = result.route_count;
    cold_mapped_count_ = 0; cold_cached_count_ = 0;
    std::copy_n(result.route_ranks.begin(), cold_count_, cold_ranks_.begin());
    std::copy_n(result.route_ids.begin(), cold_count_, cold_ids_.begin());
    for (int c = 0; c < cold_count_; ++c)
        cold_source_slots_[c] = arena_verify_arena_
            ? arena_verify_arena_->GetExpertSlotHost(cold_layer_, cold_ids_[c]) : ops::MoeExpertSlotHost{};
    arena_verify_pending_ = false;
    stats_.cpu_route_wait_ns += result.route_end_ns - result.route_begin_ns;
    stats_.cpu_join_wait_ns += steady_ns() - join;
    if (!cold_count_) { cold_source_slots_ = {}; return {}; }
    ++stats_.cpu_route_async_layers;
    stats_.cpu_activation_wait_ns += result.activation_ns;
    stats_.cpu_mmap_prefetch_ns += result.prefetch_ns;
    stats_.cpu_mmap_prefetch_calls += result.prefetch_calls;
    stats_.cpu_mmap_prefetch_ranges += result.prefetch_ranges;
    stats_.cpu_mmap_prefetch_bytes += result.prefetch_bytes;
    stats_.cpu_mmap_prefetch_failures += result.prefetch_failures;
    stats_.cpu_mmap_touch_ns += result.touch_ns;
    stats_.cpu_mmap_touch_pages += result.touch_pages;
    stats_.cpu_scale_cache_bytes = std::max(stats_.cpu_scale_cache_bytes, result.scale_cache_bytes);
    stats_.cpu_scale_cache_hits += result.scale_cache_hits;
    stats_.cpu_scale_cache_misses += result.scale_cache_misses;
    stats_.cpu_scale_cache_fills += result.scale_cache_fills;
    stats_.cpu_scale_cache_evictions += result.scale_cache_evictions;
    stats_.cpu_scale_cache_fallbacks += result.scale_cache_fallbacks;
    stats_.cpu_scale_cache_prepare_ns += result.scale_cache_prepare_ns;
    stats_.cpu_diagnostic_batches += result.diagnostic_batches;
    stats_.cpu_diagnostic_failures += result.diagnostic_failures;
    stats_.cpu_verify_batches += result.verify_batches;
    stats_.cpu_a64_verify_batches += result.a64_verify_batches;
    stats_.cpu_process_faults += result.process_faults;
    stats_.cpu_process_user_ns += result.process_user_ns;
    stats_.cpu_process_kernel_ns += result.process_kernel_ns;
    stats_.cpu_gate_up_ns += result.phases.gate_up_ns;
    stats_.cpu_quantize_ns += result.phases.quantize_ns;
    stats_.cpu_down_ns += result.phases.down_ns;
    stats_.cpu_gate_chunks += result.phases.gate_chunks;
    stats_.cpu_down_chunks += result.phases.down_chunks;
    stats_.cpu_compute_ns += result.end_ns - result.begin_ns;
    for (int c = 0; c < cold_count_; ++c) if (cpu_observer_) {
        RouteRankEntry entry;
        entry.token_idx = cold_ranks_[c] / top_k_; entry.rank = cold_ranks_[c] % top_k_;
        entry.expert_id = cold_ids_[c]; entry.is_gpu = false;
        cpu_observer_(cpu_observer_state_, *ctx_, cold_layer_, entry, cold_source_slots_[c],
            x_host_ + size_t(entry.token_idx) * hidden_size_,
            scratch_h_ + size_t(c) * intermediate_size_, cpu_out_host_ + size_t(c) * hidden_size_);
    }
    for (int c = 0; c < cold_count_;) {
        int end = c + 1;
        while (end < cold_count_ && cold_ranks_[end] == cold_ranks_[end - 1] + 1) ++end;
        const size_t bytes = size_t(end - c) * hidden_size_ * sizeof(float);
        prev_cpu_h2d_ev_ = ctx_->queue().memcpy(
            rank_output + size_t(cold_ranks_[c]) * hidden_size_, cpu_out_host_ + size_t(c) * hidden_size_, bytes);
        ++stats_.cpu_result_copies; stats_.cpu_result_bytes += bytes;
        c = end;
    }
    stats_.cold_eligible += cold_count_;
    stats_.cpu_tasks_dispatched += cold_count_; stats_.cpu_tasks_completed += cold_count_;
    stats_.cpu_max_batch_tasks = std::max(stats_.cpu_max_batch_tasks, uint64_t(cold_count_));
    ++stats_.cpu_async_layers;
    cold_source_slots_ = {}; cold_count_ = 0;
    return prev_cpu_h2d_ev_;
}

} // namespace aila
