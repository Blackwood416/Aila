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
    const int spinUs = std::clamp(aila::env::read_int_raw("AILA_Q4E_CPU_HANDOFF_SPIN_US", 0), 0, 5000);
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
        if (verify_a64 && (!aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8", false) ||
                          !aila::env::read_flag("AILA_Q4E_CPU_Q8_ACTIVATION64", false)))
            throw std::invalid_argument("CPU A64 verification requires the NF4Q8 Activation64 primary backend");
        init_scale_cache();
        if (aila::env::read_flag("AILA_Q4E_CPU_TILE_POOL", false)) {
            tilePool = std::make_unique<ops::cpu::CpuTilePool>(threads);
            AILA_LOG_INFO("[Q4E CpuTilePool] threads=%d caller_participates=1 claim_tiles=4 bounded_spin_us=1000", threads);
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
    void run() {
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
                current.activation.wait_and_throw();
                completed.activation_ns = steady_ns() - begin;
                // Host direct-read merge may still own this output arena. Wait
                // on the worker, after route/input publication, never on the
                // enqueue thread that is submitting the next layer's GPU work.
                if (current.output_reader != sycl::event{}) current.output_reader.wait_and_throw();
                if (current.transfer_reader != sycl::event{}) current.transfer_reader.wait_and_throw();
                const auto before = diagnostic ? sample_cpu_process() : CpuProcessSample{};
                completed.begin_ns = steady_ns();
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
                        static const bool nf4_q8 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8", false);
                        static const bool nf4_q8_block64 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8_BLOCK64", false);
                        static const bool nf4_q8_row4 = aila::env::read_flag("AILA_Q4E_CPU_NF4_Q8_ROW4", false);
                        static const bool nf4_scale_shift = aila::env::read_flag("AILA_Q4E_CPU_SCALE_SHIFT", false);
                        static const bool nf4_q8_activation64 = aila::env::read_flag("AILA_Q4E_CPU_Q8_ACTIVATION64", false);
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
                            // Original mapped nested scales, division loader,
                            // original A64 single-row dot and scalar SwiGLU.
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
    async_route_pending_=false; async_route_collected_=false; async_route_count_=0;
    verify_async_rows_=0; verify_async_layer_=-1;
    async_route_logged_=false;
    arena_cold_rows_pending_=false; arena_cold_rows_joined_=false;
    arena_cold_rows_=0; arena_cold_rows_next_row_=0;
    async_route_sources_.clear(); async_route_experts_=0;
    if (ctx_) {
        try { ctx_->synchronize(); } catch (...) {}
    }
    active_cpu_leases_.clear();
    if (ctx_) {
        if (x_host_) { ctx_->free_host(x_host_); x_host_ = nullptr; }
        if (cpu_out_host_) { ctx_->free_host(cpu_out_host_); cpu_out_host_ = nullptr; }
        if (cpu_result_packet_host_) { ctx_->free_host(cpu_result_packet_host_); cpu_result_packet_host_=nullptr; }
        if (cpu_result_packet_dev_) { ctx_->free_device(cpu_result_packet_dev_); cpu_result_packet_dev_=nullptr; }
        if (scratch_h_) { ctx_->free_host(scratch_h_); scratch_h_ = nullptr; }
        if (gpu_slots_host_) { ctx_->free_host(gpu_slots_host_); gpu_slots_host_ = nullptr; }
        if (gpu_slots_dev_) { ctx_->free_device(gpu_slots_dev_); gpu_slots_dev_ = nullptr; }
        if (gpu_ranks_host_) { ctx_->free_host(gpu_ranks_host_); gpu_ranks_host_ = nullptr; }
        if (gpu_ranks_dev_) { ctx_->free_device(gpu_ranks_dev_); gpu_ranks_dev_ = nullptr; }
        if (gpu_tokens_host_) { ctx_->free_host(gpu_tokens_host_); gpu_tokens_host_ = nullptr; }
        if (gpu_tokens_dev_) { ctx_->free_device(gpu_tokens_dev_); gpu_tokens_dev_ = nullptr; }
        if (weights_dev_) { ctx_->free_device(weights_dev_); weights_dev_ = nullptr; }
        if (down_acc_dev_) { ctx_->free_device(down_acc_dev_); down_acc_dev_ = nullptr; }
        if (xq_dev_) { ctx_->free_device(xq_dev_); xq_dev_ = nullptr; }
        if (xs_dev_) { ctx_->free_device(xs_dev_); xs_dev_ = nullptr; }
        if (hq_dev_) { ctx_->free_device(hq_dev_); hq_dev_ = nullptr; }
        if (hs_dev_) { ctx_->free_device(hs_dev_); hs_dev_ = nullptr; }
        q8_adaptive_ = false;
    }
    initialized_ = false;
    cpu_result_packet_active_=false;
    cpu_host_result_active_=false;cpu_host_rows_={};cpu_result_host_read_=false;
    cpu_output_reader_event_={};
}

void Qwen4ExpExpertExecutor::init(
    Context& ctx,
    Qwen4ExpExpertCache& cache,
    int max_seq_len,
    int top_k,
    int hidden_size,
    int intermediate_size) {
    cleanup();

    ctx_ = &ctx;
    cache_ = &cache;
    max_seq_len_ = std::max(1, max_seq_len);
    top_k_ = std::max(1, top_k);
    hidden_size_ = hidden_size > 0 ? hidden_size : 2560;
    intermediate_size_ = intermediate_size > 0 ? intermediate_size : 640;

    const int env_max_cpu = aila::env::read_int_raw("AILA_Q4E_CPU_MAX_EXPERTS", 2);
    max_cpu_credits_ = std::clamp(env_max_cpu, 0, ops::cpu::kMaxExpertBatchTasks);
    cpu_placement_ = std::clamp(aila::env::read_int("AILA_Q4E_CPU_PLACEMENT", 0), 0, 2);
    cpu_fixed_credits_ = aila::env::read_flag("AILA_Q4E_CPU_FIXED_CREDITS", false);
    cpu_batch_prepare_ = aila::env::read_flag("AILA_Q4E_CPU_BATCH_PREPARE", false);
    cpu_wait_prefetch_ = aila::env::read_flag("AILA_Q4E_CPU_WAIT_PREFETCH", false);
    cpu_mmap_cold_ = aila::env::read_flag("AILA_Q4E_CPU_MMAP_COLD", false);
    cpu_coalesce_results_ = aila::env::read_flag("AILA_Q4E_CPU_COALESCE_RESULTS", false);
    const int env_max_tokens = aila::env::read_int("AILA_Q4E_CPU_MAX_TOKENS", 4);
    max_cpu_tokens_ = std::max(1, std::min(4, env_max_tokens));

    const size_t max_entries = static_cast<size_t>(max_seq_len_) * top_k_;
    active_plan_.entries.reserve(max_entries);
    active_plan_.ready_gpu_indices.reserve(max_entries);
    active_plan_.pending_gpu_indices.reserve(max_entries);
    active_plan_.cpu_indices.reserve(max_entries);
    active_plan_.top_k = top_k_;

    active_cpu_leases_.reserve(ops::cpu::kMaxExpertBatchTasks);

    static const bool result_packet=aila::env::read_flag("AILA_Q4E_CPU_RESULT_PACKET",false);
    static const bool result_host_read=aila::env::read_flag("AILA_Q4E_CPU_RESULT_HOST_READ",false);
    if (result_packet && result_host_read)
        throw std::invalid_argument("CPU_RESULT_HOST_READ and CPU_RESULT_PACKET are mutually exclusive");
    cpu_result_host_read_=result_host_read && top_k_>0 && top_k_<=16 &&
        ctx_->queue().get_device().has(sycl::aspect::usm_host_allocations);
    if(cpu_result_host_read_) AILA_LOG_INFO("[Q4E CpuResultHostRead] enabled=1 resident_b1=1 topk=%d merge_reader_fence=1",top_k_);
    // Preallocate pinned host and device buffers
    x_host_ = static_cast<bf16*>(ctx_->alloc_host(4ULL * hidden_size_ * sizeof(bf16)));
    cpu_out_host_ = static_cast<float*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * hidden_size_ * sizeof(float)));
    scratch_h_ = static_cast<bf16*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * intermediate_size_ * sizeof(bf16)));
    if(result_packet && top_k_<=16) {
        const size_t bytes=16*sizeof(int)+16ULL*hidden_size_*sizeof(float);
        cpu_result_packet_host_=ctx_->alloc_host(bytes);
        cpu_result_packet_dev_=ctx_->alloc_device(bytes);
    }

    const size_t max_gpu_tasks = static_cast<size_t>(max_seq_len_) * top_k_;
    gpu_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_host(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_device(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_ranks_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_ranks_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    gpu_tokens_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_tokens_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    weights_dev_ = static_cast<float*>(ctx_->alloc_device(max_gpu_tasks * sizeof(float)));

    const size_t max_down_acc = 4ULL * top_k_ * hidden_size_;
    down_acc_dev_ = static_cast<float*>(ctx_->alloc_device(max_down_acc * sizeof(float)));

    // Opt-in W4A8 adaptive decode. The Q8 variant only exists for the
    // production expert geometry, so any other shape silently keeps the BF16
    // kernels and never allocates these arenas.
    q8_adaptive_ = aila::env::read_flag("AILA_Q4E_GPU_Q8_ADAPTIVE", false) &&
        ctx_->queue().get_device().is_gpu() && hidden_size_ == 2560 &&
        intermediate_size_ == 640 && hidden_size_ % 64 == 0 && intermediate_size_ % 64 == 0 &&
        top_k_ >= 1 && top_k_ <= 16;
    if (q8_adaptive_) {
        const size_t max_tokens = 4;
        xq_dev_ = static_cast<int8_t*>(ctx_->alloc_device(max_tokens * size_t(hidden_size_)));
        xs_dev_ = static_cast<float*>(
            ctx_->alloc_device(max_tokens * size_t(hidden_size_ / 64) * sizeof(float)));
        hq_dev_ = static_cast<int8_t*>(
            ctx_->alloc_device(max_gpu_tasks * size_t(intermediate_size_)));
        hs_dev_ = static_cast<float*>(ctx_->alloc_device(
            max_gpu_tasks * size_t(intermediate_size_ / 64) * sizeof(float)));
        AILA_LOG_INFO("[Q4E Adaptive Q8] enabled=1 hidden=%d intermediate=%d top_k=%d "
                      "activation_block=64 experts_capacity=%zu",
            hidden_size_, intermediate_size_, top_k_, max_gpu_tasks);
    }

    const size_t max_pending = static_cast<size_t>(max_seq_len_) * top_k_;
    pending_experts_buf_.assign(max_pending, 0);
    assigned_slots_buf_.assign(max_pending, -1);
    prev_cpu_h2d_ev_ = sycl::event{};

    stats_.reset();
    slot_seen_.assign(cache_->num_slots(), false);
    straggler_penalty_active_ = false;
    consecutive_stragglers_ = 0;
    cpu_retry_countdown_ = 0;
    cold_budget_ns_ = 1000.0 * std::clamp(aila::env::read_int("AILA_Q4E_COLD_BUDGET_US",1000),100,10000);
    cold_policy_=std::clamp(aila::env::read_int_raw("AILA_Q4E_COLD_POLICY",0),0,3);
    cold_pcie_num_=std::clamp(aila::env::read_int_raw("AILA_Q4E_COLD_PCIE_NUM",141),0,256);
    for (int n=0;n<=16;++n) cold_cost_ns_[n]=100000.0+400000.0*n;
    if (aila::env::read_flag("AILA_Q4E_CPU_OVERLAP", false) || aila::env::read_int("AILA_Q4E_COLD_MODE",0)>0) {
        const int threads = std::clamp(aila::env::read_int("AILA_Q4E_CPU_THREADS", 2), 1, 18);
        cpu_worker_ = std::make_unique<CpuWorker>(threads);
    }
    initialized_ = true;
}

sycl::queue* Qwen4ExpExpertExecutor::copy_queue() const {
    if (!q4e::moe_copy_via_copyq()) return nullptr;
    sycl::queue* q = copy_q_override_;
    if (!q && cache_) q = cache_->CopyQueue();
    static std::atomic<bool> logged{false};
    if (!logged.load(std::memory_order_relaxed) &&
        !logged.exchange(true, std::memory_order_relaxed)) {
        if (q) {
            AILA_LOG_INFO("[Q4E MoeCopyViaCopyQ] enabled=1 source=%s "
                          "host_copies=x_d2h+cpu_result_h2d (compute-queue barrier ordered)",
                copy_q_override_ ? "owner" : "expert_cache");
        } else {
            AILA_LOG_INFO("[Q4E MoeCopyViaCopyQ] enabled=1 but no copy queue is "
                          "available; host copies stay on the compute queue");
        }
    }
    return q;
}

void Qwen4ExpExpertExecutor::StageResidentInput(const bf16* x_dev, int rows) {
    if(rows<1 || rows>max_cpu_tokens_ || verify_async_rows_ || cold_count_ || async_route_pending_ || async_route_collected_ || arena_verify_pending_ || cpu_host_result_active_)throw std::logic_error("CPU cold slot has an unfinished batch or invalid rows");
    static const bool deferArenaWait = aila::env::read_flag("AILA_Q4E_CPU_DEFER_ARENA_WAIT", false);
    // The unfinished-job guard above excludes CPU input readers. Copies on
    // the same in-order queue retain GPU write ordering without a host wait.
    if(!deferArenaWait && resident_input_event_!=sycl::event{})resident_input_event_.wait_and_throw();
    // The returned event stays the worker's activation gate (job.activation),
    // so a copy-queue D2H is still fully ordered before the first x_host_ read.
    resident_input_event_=q4e::submit_host_copy(*ctx_,copy_queue(),x_host_,x_dev,
        size_t(rows)*hidden_size_*sizeof(bf16));
}

void Qwen4ExpExpertExecutor::PrepareAsyncResidentSources(int layers, int experts) {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_CPU_ASYNC_ROUTE", false) ||
        aila::env::read_flag("AILA_Q4E_VERIFY_ASYNC_ROUTE", false);
    if (!enabled || !cache_ || layers <= 0 || experts != 512 || cache_->host_pool_enabled() ||
        !cpu_mmap_cold_ || cache_->cpu_packed_source_slots() ||
        hidden_size_ != ops::cpu::kHiddenDim || intermediate_size_ != ops::cpu::kIntermediateDim) return;
    async_route_experts_=experts;
    if (cache_->large_page_resident()) return; // Use its live ownership table, never cached mmap descriptors.
    async_route_sources_.resize(size_t(layers)*experts);
    for (int l=0;l<layers;++l) for (int e=0;e<experts;++e)
        async_route_sources_[size_t(l)*experts+e]=cache_->MappedExpertHostSource(l,e);
}

bool Qwen4ExpExpertExecutor::CanStartAsyncResidentRoute(int layer) const {
    return initialized_ && cpu_worker_ && cache_ && cache_->frozen() && !cache_->host_pool_enabled() &&
        cpu_mmap_cold_ && cold_policy_==1 && top_k_<=16 && max_cpu_credits_>=top_k_ &&
        !cache_->cpu_packed_source_slots() && !verify_async_rows_ && !cold_count_ && !async_route_pending_ && !async_route_collected_ && !arena_verify_pending_ && !cpu_host_result_active_ &&
        layer>=0 && async_route_experts_==512 &&
        (cache_->ResidentLayerSources(layer) || (size_t(layer)+1)*async_route_experts_<=async_route_sources_.size());
}

void Qwen4ExpExpertExecutor::StartAsyncResidentRoute(int layer, const int* ids, const int* selected,
    const sycl::event& route_event) {
    if (!CanStartAsyncResidentRoute(layer) || !ids || !selected || resident_input_event_==sycl::event{})
        throw std::logic_error("Invalid asynchronous resident CPU route service");
    try {
        // The previous CPU result transfer still reads these pinned arenas.
        static const bool deferArenaWait = aila::env::read_flag("AILA_Q4E_CPU_DEFER_ARENA_WAIT", false);
        const sycl::event transferReader = deferArenaWait ? prev_cpu_h2d_ev_ : sycl::event{};
        if (!deferArenaWait && prev_cpu_h2d_ev_!=sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
        prev_cpu_h2d_ev_={}; active_cpu_leases_.clear(); cold_layer_=layer;
        cold_source_slots_={}; cold_mapped_count_=0; cold_cached_count_=0;
        CpuWorker::Job job;
        job.hidden=hidden_size_;job.intermediate=intermediate_size_;job.tokens=1;
        job.x=x_host_;job.h=scratch_h_;job.out=cpu_out_host_;job.resident_prepared=true;
        job.activation=resident_input_event_;job.route_event=route_event;
        job.output_reader=cpu_output_reader_event_;
        job.transfer_reader=transferReader;
        job.route_ids=ids;job.route_selected=selected;
        job.route_top_k=top_k_;job.route_experts=async_route_experts_;
        job.route_sources_resident=cache_->large_page_resident();
        job.route_sources=job.route_sources_resident ? cache_->ResidentLayerSources(layer)
            : async_route_sources_.data()+size_t(layer)*async_route_experts_;
        // Cache and leases remain coordinator-owned. Frozen/no-prefetch/no-swap
        // dispatch cannot turn these misses into GPU-ready inputs while in flight.
        for(int e=0;e<async_route_experts_;++e) {
            static const bool lightRoute = aila::env::read_flag("AILA_Q4E_CPU_LIGHT_ROUTE", false);
            if (lightRoute) job.route_cpu_eligible[e] = !cache_->HasGpuRouteSource(layer,e);
            else {
                const auto residency=cache_->query_residency(layer,e);
                job.route_cpu_eligible[e]=!residency.gpu_ready && !residency.gpu_copying;
            }
        }
        cold_submit_ns_=steady_ns();
        async_route_pending_=true;
        cpu_worker_->submit(job);
        if (!async_route_logged_) {
            AILA_LOG_INFO("[Q4E AsyncRouteEnabled] layer=%d topk=%d host_ram=0 mmap=1 policy=1 credits=%d",
                layer,top_k_,max_cpu_credits_);
            async_route_logged_=true;
        }
    } catch(...) {AbortResidentCold();throw;}
}

void Qwen4ExpExpertExecutor::RetireAsyncResidentRoute(GpuReadyLease& route) {
    if (!async_route_collected_ || async_route_pending_ || route.layer!=cold_layer_)
        throw std::logic_error("Asynchronous CPU route was not collected");
    int gpu_count=0, consumed=0;
    for (int i=0;i<route.cold_count;++i) {
        bool cpu=false;
        for (int c=0;c<async_route_count_;++c) if (route.cold_ranks[i]==async_route_ranks_[c]) {
            if (route.cold_ids[i]!=async_route_ids_[c])
                throw std::logic_error("Asynchronous CPU route binding changed expert");
            cpu=true; ++consumed; break;
        }
        if (!cpu) {route.cold_ids[gpu_count]=route.cold_ids[i];route.cold_ranks[gpu_count++]=route.cold_ranks[i];}
    }
    if (consumed!=async_route_count_)
        throw std::logic_error("Asynchronous CPU route binding changed placement");
    route.cold_count=gpu_count;
    async_route_collected_=false;async_route_count_=0;
}

int Qwen4ExpExpertExecutor::StartResidentCold(GpuReadyLease& route, const bf16* x_dev) {
    if (!cpu_worker_ || !cache_->frozen() || route.cold_count>16 || cold_count_ || async_route_pending_ || async_route_collected_ || arena_verify_pending_ || cpu_host_result_active_)
        throw std::logic_error("Invalid resident CPU cold service");
    if (prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
    prev_cpu_h2d_ev_={}; active_cpu_leases_.clear(); cold_layer_=route.layer;
    cold_source_slots_={}; cold_mapped_count_=0; cold_cached_count_=0;
    int gpu_count=0;
    std::array<bool, ops::cpu::kMaxExpertBatchTasks> mapped_tasks{};
    const bool retry=cold_policy_==2 && (++cold_batches_seen_%32==0);
    try {
        std::array<HostExpertLease,16> chosen;
        std::array<ops::MoeExpertSlotHost,16> chosen_sources{};
        std::array<bool,16> mapped{};
        std::array<bool,16> cached{};
        int selected=0;
        static const bool low_rank=aila::env::read_flag("AILA_Q4E_COLD_LOW_RANK",false);
        for(int scan=0;scan<route.cold_count;++scan) {
            const int i=low_rank && cold_policy_!=3?route.cold_count-1-scan:scan;
            // Strata reserves the last floor(nmiss * fraction) routed misses
            // for GPU PCIe work. CPU-ineligible host misses also fall back to GPU.
            if(cold_policy_==3 && i>=route.cold_count-((route.cold_count*cold_pcie_num_)>>8))continue;
            const int e=route.cold_ids[i];
            const auto residency=cache_->query_residency(route.layer,e);
            if (!residency.gpu_ready && !residency.gpu_copying) {
                const auto source=!residency.host_ready && cpu_mmap_cold_
                    ? cache_->CpuExpertHostSource(route.layer,e,&cached[i]) : ops::MoeExpertSlotHost{};
                if (!residency.host_ready && !source.gate_up_packed) continue;
                ++stats_.cold_eligible;
                if(selected<std::min(16,max_cpu_credits_) &&
                    (cold_policy_==1 || cold_policy_==3 || retry || cold_cost_ns_[selected+1]<=cold_budget_ns_)) {
                    if (residency.host_ready) {
                        auto lease=cache_->acquire_host_lease(route.layer,e);
                        if (lease.is_valid()) {chosen_sources[i]=lease.slot_host(); chosen[i]=std::move(lease); ++selected;}
                    } else {chosen_sources[i]=source; mapped[i]=!cached[i]; ++selected;}
                } else ++stats_.cold_budget_rejected;
            }
        }
        for(int i=0;i<route.cold_count;++i) {
            const int e=route.cold_ids[i],rank=route.cold_ranks[i];
            if(chosen_sources[i].gate_up_packed) {
                mapped_tasks[cold_count_]=mapped[i];
                cold_source_slots_[cold_count_]=chosen_sources[i];
                cold_ranks_[cold_count_]=rank;cold_ids_[cold_count_++]=e;
                if (chosen[i].is_valid()) active_cpu_leases_.push_back(std::move(chosen[i]));
                cold_mapped_count_+=mapped[i];
                cold_cached_count_+=cached[i];
            } else {route.cold_ids[gpu_count]=e;route.cold_ranks[gpu_count++]=rank;}
        }
        route.cold_count=gpu_count;
        if(!cold_count_) return 0;
        CpuWorker::Job job;job.count=cold_count_;job.hidden=hidden_size_;job.intermediate=intermediate_size_;job.tokens=1;
        job.x=x_host_;job.h=scratch_h_;job.out=cpu_out_host_;job.resident_prepared=true;
        job.output_reader=cpu_output_reader_event_;
        job.mapped_tasks=mapped_tasks;
        job.activation=resident_input_event_!=sycl::event{} ? resident_input_event_ :
            q4e::submit_host_copy(*ctx_,copy_queue(),x_host_,x_dev,size_t(hidden_size_)*sizeof(bf16));
        for(int c=0;c<cold_count_;++c)job.tasks[c]={cold_source_slots_[c],0};
        if((cpu_worker_->verify_q8 || cpu_worker_->verify_a64) && cold_cached_count_) {
            for(int c=0;c<cold_count_;++c) {
                const auto original=cache_->MappedExpertHostSource(cold_layer_,cold_ids_[c]);
                if(!original.gate_up_packed || !original.down_packed)
                    throw std::logic_error("Cached CPU source has no valid original shadow reference");
                cold_reference_tasks_[c]={original,0};
            }
            job.reference_tasks=cold_reference_tasks_.data();
        }
        cold_submit_ns_=steady_ns();cpu_worker_->submit(job);
        cache_->TraceColdCpu(cold_layer_,cold_ids_.data(),cold_ranks_.data(),cold_count_);
        stats_.cpu_tasks_dispatched+=cold_count_;
        stats_.cpu_mapped_tasks_dispatched+=cold_mapped_count_;
        stats_.cpu_cached_tasks_dispatched+=cold_cached_count_;
        if(cache_->tracing())cache_->trace_host_interval("cold_cpu_submit",cold_layer_,cold_submit_ns_,steady_ns());
        return cold_count_;
    } catch(...) {AbortResidentCold();throw;}
}

int Qwen4ExpExpertExecutor::StartArenaCold(const Qwen4ExpHostArena& arena, int layer,
    const int* ids, const int* ranks, int count, const bf16* x_dev, int row_token, int tokens) {
    if (!cpu_worker_ || count<=0) return 0;
    if (!ids || !ranks || count>16 || tokens<1 || tokens>max_cpu_tokens_ ||
        row_token<0 || row_token>=tokens || cold_count_ || async_route_pending_ ||
        async_route_collected_ || arena_verify_pending_ || cpu_host_result_active_)
        throw std::logic_error("Invalid native arena CPU batch");
    // The grouped schedule (AILA_Q4E_VERIFY_CPU_MERGE) lives in the window-wide
    // job that StartArenaColdRows builds, which the caller only reaches under
    // AILA_Q4E_VERIFY_CPU_BATCH. A multi-row per-row call therefore means the
    // merge flag cannot act; report that once instead of silently no-oping.
    if (tokens >= 2) {
        static const bool s_merge = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_MERGE", false);
        if (s_merge && !merge_inactive_logged_) {
            AILA_LOG_INFO("[Q4E VerifyCpuMerge] inactive: window job requires AILA_Q4E_VERIFY_CPU_BATCH=1");
            merge_inactive_logged_ = true;
        }
    }
    try {
        if (prev_cpu_h2d_ev_!=sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
        prev_cpu_h2d_ev_={}; cold_layer_=layer;
        cold_count_=std::min(count,max_cpu_credits_);
        cold_cached_count_=cold_count_; cold_mapped_count_=0;
        if (!cold_count_) return 0;
        CpuWorker::Job job;
        job.count=cold_count_; job.hidden=hidden_size_; job.intermediate=intermediate_size_; job.tokens=tokens;
        job.x=x_host_; job.h=scratch_h_; job.out=cpu_out_host_; job.resident_prepared=true;
        job.output_reader=cpu_output_reader_event_;
        job.activation=resident_input_event_!=sycl::event{} ? resident_input_event_ :
            q4e::submit_host_copy(*ctx_,copy_queue(),x_host_,x_dev,
                size_t(hidden_size_)*size_t(tokens)*sizeof(bf16));
        for (int c=0;c<cold_count_;++c) {
            const auto source=arena.GetExpertSlotHost(layer,ids[c]);
            if (!source.gate_up_packed || !source.down_packed || ranks[c]<0 || ranks[c]>=top_k_)
                throw std::logic_error("CPU expert has no native RAM owner or rank");
            cold_source_slots_[c]=source; cold_ids_[c]=ids[c]; cold_ranks_[c]=ranks[c];
            job.tasks[c]={source,row_token};
        }
        cold_submit_ns_=steady_ns(); cpu_worker_->submit(job);
        stats_.cpu_tasks_dispatched+=cold_count_;
        stats_.cpu_cached_tasks_dispatched+=cold_count_;
        return cold_count_;
    } catch (...) { AbortResidentCold(); throw; }
}

sycl::event Qwen4ExpExpertExecutor::JoinResidentCold(float* output, const sycl::event* order_dep) {
    cpu_result_packet_active_=false;
    if(!cold_count_ && !async_route_pending_)return {};
    const auto join=steady_ns();
    const auto result=cpu_worker_->finish();
    if (async_route_pending_) {
        // finish() is the publication fence for rank/source metadata and CPU output.
        cold_count_=result.route_count;
        cold_cached_count_=cache_->large_page_resident() ? cold_count_ : 0;
        cold_mapped_count_=cold_count_-cold_cached_count_;
        std::copy_n(result.route_ranks.begin(), cold_count_, cold_ranks_.begin());
        std::copy_n(result.route_ids.begin(), cold_count_, cold_ids_.begin());
        for(int c=0;c<cold_count_;++c)
            cold_source_slots_[c]=cache_->large_page_resident()
                ? cache_->CpuExpertHostSource(cold_layer_,cold_ids_[c])
                : async_route_sources_[size_t(cold_layer_)*async_route_experts_+cold_ids_[c]];
        // The async resident route is single-row (count==1), so cold_count_<=16
        // fits the fixed legacy reconciliation arrays.
        std::copy_n(cold_ranks_.begin(),cold_count_,async_route_ranks_.begin());
        std::copy_n(cold_ids_.begin(),cold_count_,async_route_ids_.begin());
        async_route_count_=cold_count_;
        async_route_pending_=false;async_route_collected_=true;
        ++stats_.cpu_route_async_layers;
        stats_.cpu_route_wait_ns+=result.route_end_ns-result.route_begin_ns;
        stats_.cold_eligible+=cold_count_;
        stats_.cpu_tasks_dispatched+=cold_count_;stats_.cpu_mapped_tasks_dispatched+=cold_mapped_count_;
        stats_.cpu_cached_tasks_dispatched+=cold_cached_count_;
        cache_->TraceColdCpu(cold_layer_,cold_ids_.data(),cold_ranks_.data(),cold_count_);
        if(cache_->tracing())cache_->trace_host_interval("cold_cpu_route_wait",cold_layer_,result.route_begin_ns,result.route_end_ns);
    }
    stats_.cpu_join_wait_ns+=steady_ns()-join;
    if (!cold_count_) return {};
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
    const auto completion_ns=result.end_ns-cold_submit_ns_;
    const uint64_t before_compute_ns=result.begin_ns-cold_submit_ns_;
    const uint64_t event_wait_ns=result.activation_ns+result.route_end_ns-result.route_begin_ns;
    stats_.cold_queue_ns+=before_compute_ns-std::min(event_wait_ns,before_compute_ns);
    // Recoverable policy uses CPU-owned service time, including source warmup.
    // Activation/host scheduling waits stay separate; flags off preserve its
    // previous compute-only value.
    const double measured=cold_policy_==2?
        double(result.end_ns-result.begin_ns+result.prefetch_ns+result.touch_ns+result.scale_cache_prepare_ns):double(completion_ns);
    cold_cost_ns_[cold_count_]=0.8*cold_cost_ns_[cold_count_]+0.2*measured;
    if(cold_policy_==2) {
        // A fresh completed batch also informs larger, stale buckets. Retries
        // remain bounded by max_cpu_credits_, with only one batch outstanding.
        for(int n=cold_count_+1;n<=16;++n)
            cold_cost_ns_[n]=0.95*cold_cost_ns_[n]+0.05*measured*n/cold_count_;
    }
    if(cache_->tracing()) {
        if (result.prefetch_calls) cache_->trace_host_interval("cold_cpu_mmap_prefetch",cold_layer_,
            result.prefetch_begin_ns,result.prefetch_begin_ns+result.prefetch_ns);
        if (result.touch_pages) cache_->trace_host_interval("cold_cpu_mmap_touch",cold_layer_,
            result.touch_begin_ns,result.touch_begin_ns+result.touch_ns);
        if (result.scale_cache_prepare_ns) cache_->trace_host_interval("cold_cpu_scale_cache_prepare",cold_layer_,
            result.scale_cache_begin_ns,result.scale_cache_begin_ns+result.scale_cache_prepare_ns);
        cache_->trace_host_interval("cold_cpu_compute",cold_layer_,result.begin_ns,result.end_ns);
        cache_->trace_host_interval("cold_cpu_join",cold_layer_,join,steady_ns());
        cache_->TraceDeviceEvent("cold_input_copy",cold_layer_,resident_input_event_);
    }
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
        // One immutable transfer: fixed rank-to-packed map then CPU output rows.
        // StartResidentCold drains the previous transfer before this host arena
        // can be reused; the in-order merge reads device bytes before next write.
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
        // One compute-queue fence covers the whole coalesced rank-span batch.
        sycl::queue* copy_q=copy_queue();
        sycl::event order_fence{};
        const sycl::event* order_fence_ptr=nullptr;
        if(order_dep){order_fence_ptr=order_dep;}
        else if(copy_q){order_fence=q4e::host_copy_fence(*ctx_,copy_q);order_fence_ptr=&order_fence;}
        for(int c=0;c<cold_count_;) {
            int end=c+1;while(end<cold_count_ && cold_ranks_[end]==cold_ranks_[end-1]+1)++end;
            const size_t bytes=size_t(end-c)*hidden_size_*sizeof(float);
            // Returned as prev_cpu_h2d_ev_ so the in-order merge (and next
            // layer's StartArenaCold drain) takes the copy event as dependency.
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
    active_cpu_leases_.clear();cold_source_slots_={};cold_count_=0;cold_mapped_count_=0;cold_cached_count_=0;
    return prev_cpu_h2d_ev_;
}

bool Qwen4ExpExpertExecutor::CanExecuteArenaColdRows(int rows) const {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_VERIFY_CPU_BATCH", false);
    // The packet and host-read result modes publish a single-row view, so the
    // batched window keeps the default per-row rank-span upload and is disabled
    // whenever either of those layouts is selected.
    return enabled && initialized_ && cpu_worker_ && rows >= 2 && rows <= 4 &&
        rows <= max_cpu_tokens_ && hidden_size_ == ops::cpu::kHiddenDim &&
        intermediate_size_ == ops::cpu::kIntermediateDim &&
        !cpu_result_host_read_ && !cpu_result_packet_host_ &&
        !cold_count_ && !async_route_pending_ && !async_route_collected_ &&
        !verify_async_rows_ && !arena_verify_pending_ && !arena_cold_rows_pending_ &&
        active_cpu_leases_.empty();
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
                // Each task reads its own row of the staged input; the task
                // index (not the row) owns the output row.
                job.tasks[i] = {source, r};
            }
            offset += count;
        }
        if (offset != task_count)
            throw std::logic_error("Multi-row CPU task spans are not a strict partition");
        // Opt-in C1 grouped schedule (AILA_Q4E_VERIFY_CPU_MERGE): tasks are
        // already a strict row-major partition of (row, expert) pairs, so the
        // same expert selected by several rows can be expressed as one group
        // whose members are those pair indices. Everything downstream (row
        // spans, cold_ranks_, per-row rank-span upload, observer) keeps the
        // pair layout; only the worker's tile loop is reordered.
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
                    // Flattened member lists keep each group's pair indices
                    // contiguous; the pair table itself is untouched.
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
        if (cache_->tracing()) {
            cache_->trace_host_interval("cold_cpu_compute", cold_layer_, result.begin_ns, result.end_ns);
            cache_->trace_host_interval("cold_cpu_join", cold_layer_, join, steady_ns());
            cache_->TraceDeviceEvent("cold_input_copy", cold_layer_, resident_input_event_);
        }
        arena_cold_rows_joined_ = true;
    }
    // Publish this row's CPU rank spans into its own accumulator slice; the
    // in-order queue keeps later rows from overwriting it before the merge.
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
        active_cpu_leases_.clear(); cold_source_slots_ = {};
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
    active_cpu_leases_.clear();cold_source_slots_={};cold_count_=0;cold_mapped_count_=0;cold_cached_count_=0;
    cpu_result_packet_active_=false;
    cpu_host_result_active_=false;cpu_host_rows_={};cpu_output_reader_event_={};
    async_route_pending_=false;async_route_collected_=false;async_route_count_=0;
    verify_async_rows_=0;verify_async_layer_=-1;
    arena_verify_pending_=false;
    arena_cold_rows_pending_=false;arena_cold_rows_joined_=false;
    arena_cold_rows_=0;arena_cold_rows_next_row_=0;
}

bool Qwen4ExpExpertExecutor::CanExecuteVerifyRows(int rows, int top_k) const {
    return initialized_ && cpu_worker_ && cache_->frozen() && !cache_->host_pool_enabled() &&
        cpu_mmap_cold_ && rows >= 2 && rows <= 4 && rows <= max_seq_len_ && rows <= max_cpu_tokens_ &&
        top_k == top_k_ && top_k <= 10 && max_cpu_credits_ >= top_k &&
        hidden_size_ == ops::cpu::kHiddenDim && intermediate_size_ == ops::cpu::kIntermediateDim &&
        !cold_count_ && !async_route_pending_ && !async_route_collected_ && active_cpu_leases_.empty() &&
        !arena_verify_pending_ &&
        !cpu_result_packet_host_ && !cpu_result_host_read_ && !cpu_host_result_active_;
}

void Qwen4ExpExpertExecutor::StartAsyncVerifyRoute(int layer, int rows, const int* ids, const sycl::event& route_event) {
    if (!CanExecuteVerifyRows(rows, top_k_) || !CanStartAsyncResidentRoute(layer) || verify_async_rows_ ||
        !ids || resident_input_event_ == sycl::event{})
        throw std::logic_error("Invalid asynchronous verification route");
    CpuWorker::Job job;
    job.tokens=rows; job.hidden=hidden_size_; job.intermediate=intermediate_size_;
    job.x=x_host_; job.h=scratch_h_; job.out=cpu_out_host_; job.resident_prepared=true;
    job.activation=resident_input_event_; job.route_event=route_event;
    job.output_reader=cpu_output_reader_event_; job.transfer_reader=prev_cpu_h2d_ev_;
    job.route_ids=ids; job.route_top_k=top_k_; job.route_experts=async_route_experts_;
    job.route_sources_resident=cache_->large_page_resident();
    job.route_sources=job.route_sources_resident ? cache_->ResidentLayerSources(layer)
        : async_route_sources_.data()+size_t(layer)*async_route_experts_;
    // Eligibility is captured on the coordinator. The worker reads only this
    // immutable table and mapped weights; it never acquires cache leases.
    for (int e=0;e<async_route_experts_;++e) job.route_cpu_eligible[e]=!cache_->IsDirectReady(layer,e);
    verify_async_rows_=rows; verify_async_layer_=layer;
    try { cpu_worker_->submit(job); } catch (...) { AbortResidentCold(); throw; }
}

sycl::event Qwen4ExpExpertExecutor::ExecuteVerifyRows(Context& ctx, int layer, const int* ids,
    const float* weights, const bf16* x, bf16* h, bf16* y, int rows, int top_k,
    const bf16* shared_out, const bf16* shared_gate) {
    if (!CanExecuteVerifyRows(rows, top_k))
        throw std::logic_error("CPU verification rows require frozen mapped cold service");
    const bool async_verify = verify_async_rows_ != 0;
    if (async_verify && (verify_async_rows_ != rows || verify_async_layer_ != layer))
        throw std::logic_error("Asynchronous verification route changed shape or layer");
    // This fence includes descriptor/map uploads even for an all-GPU layer.
    if (prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
    if (cpu_output_reader_event_ != sycl::event{}) cpu_output_reader_event_.wait_and_throw();
    auto route = cache_->BindReady(layer, ids, rows * top_k);
    struct Guard {
        Context& ctx;
        CpuWorker& worker;
        bool finished = false;
        ~Guard() {
            if (!finished) {
                worker.drain();
                try { ctx.synchronize(); } catch (...) {}
            }
        }
    } guard{ctx, *cpu_worker_}; // drains before route releases its GPU readers
    CpuWorker::Job job;
    job.count = route.cold_count; job.tokens = rows;
    job.hidden = hidden_size_; job.intermediate = intermediate_size_;
    job.x = x_host_; job.h = scratch_h_; job.out = cpu_out_host_;
    job.resident_prepared = true;
    for (int c = 0; c < job.count; ++c) {
        auto source = cache_->CpuExpertHostSource(layer, route.cold_ids[c]);
        if (!source.gate_up_packed || !source.down_packed)
            throw std::logic_error("CPU verification row has no mapped expert source");
        job.tasks[c] = {source, route.cold_ranks[c] / top_k};
        job.mapped_tasks[c] = !cache_->large_page_resident();
    }
    if (job.count) {
        job.activation = async_verify ? resident_input_event_ :
            q4e::submit_host_copy(ctx, copy_queue(), x_host_, x, size_t(rows) * hidden_size_ * sizeof(bf16));
        if (!async_verify) cpu_worker_->submit(job);
        cache_->TraceColdCpu(layer, route.cold_ids.data(), route.cold_ranks.data(), job.count);
    }
    auto last_upload = ctx.queue().memcpy(weights_dev_, weights, size_t(rows) * top_k * sizeof(float));
    const int ready = route.ready_count;
    if (ready) {
        static const int reuse = aila::env::read_int("AILA_Q4E_VERIFY_GPU_REUSE", 0);
        std::array<int, ops::cpu::kMaxExpertBatchTasks> order{};
        std::array<bool, ops::cpu::kMaxExpertBatchTasks> used{};
        int offset = 0, pairs = 0, quads = 0;
        // Existing expert-major kernels share a weight read across 2/4 tokens.
        // Layout is quads, then pairs, then singles; original rank maps survive.
        for (int group : {4, 2, 1}) {
            if (group > 1 && reuse < group) continue;
            for (int g = 0; g < ready; ++g) if (!used[g]) {
                std::array<int, 4> members{};
                int n = 0;
                for (int j = g; j < ready && n < group; ++j)
                    if (!used[j] && route.ready_slots[j] == route.ready_slots[g]) members[n++] = j;
                if (n != group) continue;
                for (int j = 0; j < n; ++j) { order[offset++] = members[j]; used[members[j]] = true; }
                pairs += group == 2; quads += group == 4;
            }
        }
        for (int g = 0; g < ready; ++g) {
            const int src = order[g];
            gpu_slots_host_[g] = cache_->get_slot_dev(route.ready_slots[src]);
            gpu_ranks_host_[g] = route.ready_ranks[src] % top_k;
            gpu_tokens_host_[g] = route.ready_ranks[src] / top_k;
        }
        ctx.queue().memcpy(gpu_slots_dev_, gpu_slots_host_, size_t(ready) * sizeof(ops::MoeExpertSlotDev));
        ctx.queue().memcpy(gpu_ranks_dev_, gpu_ranks_host_, size_t(ready) * sizeof(int));
        last_upload = ctx.queue().memcpy(gpu_tokens_dev_, gpu_tokens_host_, size_t(ready) * sizeof(int));
        if (reuse >= 2) {
            auto gu = ops::moe_expert_major_prefill_gate_up(ctx, gpu_slots_dev_, gpu_tokens_dev_, gpu_ranks_dev_,
                nullptr, x, h, ready, top_k, hidden_size_, intermediate_size_, 64, nullptr,
                cache_->uses_nested_codes(), pairs, quads);
            ops::moe_expert_major_prefill_down(ctx, gpu_slots_dev_, gpu_tokens_dev_, gpu_ranks_dev_,
                nullptr, h, down_acc_dev_, ready, top_k, intermediate_size_, hidden_size_, 64, &gu,
                cache_->uses_nested_codes(), pairs, quads);
        } else {
            auto gu = ops::moe_decode_gate_up_swiglu_grouped(ctx, gpu_slots_dev_, nullptr,
                x, h, ready, hidden_size_, intermediate_size_, 64, nullptr, gpu_tokens_dev_, cache_->uses_nested_codes());
            ops::moe_decode_down_to_rank_f32(ctx, gpu_slots_dev_, gpu_ranks_dev_, h, down_acc_dev_,
                ready, top_k, intermediate_size_, hidden_size_, 64, gpu_tokens_dev_, nullptr, &gu, cache_->uses_nested_codes());
        }
    }
    if (job.count || async_verify) {
        const auto join = steady_ns();
        const auto result = cpu_worker_->finish();
        if (async_verify) {
            verify_async_rows_=0; verify_async_layer_=-1;
            if (result.route_count != job.count)
                throw std::logic_error("Asynchronous verification changed cold task count");
            for (int c=0;c<job.count;++c)
                if (result.route_ranks[c] != route.cold_ranks[c] || result.route_ids[c] != route.cold_ids[c])
                    throw std::logic_error("Asynchronous verification changed cold placement");
            ++stats_.cpu_route_async_layers;
            stats_.cpu_route_wait_ns += result.route_end_ns-result.route_begin_ns;
        }
        stats_.cpu_join_wait_ns += steady_ns() - join;
        stats_.cpu_activation_wait_ns += result.activation_ns;
        stats_.cpu_compute_ns += result.end_ns - result.begin_ns;
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
        stats_.cpu_a64_verify_batches += result.a64_verify_batches;
        stats_.cpu_verify_batches += result.verify_batches;
        stats_.cpu_diagnostic_batches += result.diagnostic_batches;
        stats_.cpu_diagnostic_failures += result.diagnostic_failures;
        stats_.cpu_process_faults += result.process_faults;
        stats_.cpu_process_user_ns += result.process_user_ns;
        stats_.cpu_process_kernel_ns += result.process_kernel_ns;
        stats_.cpu_gate_up_ns += result.phases.gate_up_ns;
        stats_.cpu_quantize_ns += result.phases.quantize_ns;
        stats_.cpu_down_ns += result.phases.down_ns;
        stats_.cpu_gate_chunks += result.phases.gate_chunks;
        stats_.cpu_down_chunks += result.phases.down_chunks;
        if (cache_->tracing()) {
            cache_->trace_host_interval("cold_cpu_compute", layer, result.begin_ns, result.end_ns);
            cache_->trace_host_interval("cold_cpu_join", layer, join, steady_ns());
            cache_->TraceDeviceEvent("cold_input_copy", layer, job.activation);
        }
        for (int c = 0; c < job.count; ++c) if (cpu_observer_) {
            RouteRankEntry entry;
            entry.token_idx = job.tasks[c].token; entry.rank = route.cold_ranks[c] % top_k;
            entry.expert_id = route.cold_ids[c]; entry.is_gpu = false;
            cpu_observer_(cpu_observer_state_, ctx, layer, entry, job.tasks[c].slot,
                x_host_ + size_t(entry.token_idx) * hidden_size_, scratch_h_ + size_t(c) * intermediate_size_,
                cpu_out_host_ + size_t(c) * hidden_size_);
        }
        for (int c = 0; c < job.count;) {
            int end = c + 1;
            while (end < job.count && route.cold_ranks[end] == route.cold_ranks[end - 1] + 1) ++end;
            const auto submit = steady_ns();
            const size_t bytes = size_t(end - c) * hidden_size_ * sizeof(float);
            last_upload = ctx.queue().memcpy(down_acc_dev_ + size_t(route.cold_ranks[c]) * hidden_size_,
                cpu_out_host_ + size_t(c) * hidden_size_, bytes);
            ++stats_.cpu_result_copies; stats_.cpu_result_bytes += bytes;
            stats_.cpu_result_submit_ns += steady_ns() - submit;
            c = end;
        }
        stats_.cpu_tasks_dispatched += job.count; stats_.cpu_tasks_completed += job.count;
        stats_.cpu_mapped_tasks_dispatched += job.count; stats_.cpu_mapped_tasks_completed += job.count;
        stats_.cold_eligible += job.count;
        stats_.cpu_max_batch_tasks = std::max(stats_.cpu_max_batch_tasks, uint64_t(job.count));
        ++stats_.cpu_async_layers;
    }
    // The in-order queue joins GPU rows and all CPU rank spans before merging.
    auto merge = ops::moe_ordered_accumulate_down_rows(ctx, down_acc_dev_, weights_dev_, y,
        rows, top_k, hidden_size_, shared_out, shared_gate);
    route.Finish(merge);
    prev_cpu_h2d_ev_ = last_upload;
    stats_.gpu_tasks_dispatched += ready;
    ++stats_.layers_hetero_executed;
    guard.finished = true;
    return merge;
}

bool Qwen4ExpExpertExecutor::CanExecuteArenaVerifyRows(const Qwen4ExpHostArena& arena, int rows, int top_k) const {
    static const bool enabled = aila::env::read_flag("AILA_Q4E_ARENA_VERIFY_ROWS", false);
    return enabled && initialized_ && cpu_worker_ && rows >= 2 && rows <= 4 && rows <= max_cpu_tokens_ &&
        top_k == top_k_ && top_k <= 10 && rows * top_k <= ops::cpu::kMaxExpertBatchTasks &&
        hidden_size_ == ops::cpu::kHiddenDim && intermediate_size_ == ops::cpu::kIntermediateDim &&
        arena.IsAllocated() && arena.NumLayers() > 0 && arena.NumExpertsPerLayer() > 0 &&
        arena.NumExpertsPerLayer() <= 512 &&
        !cold_count_ && !async_route_pending_ && !async_route_collected_ && !verify_async_rows_ &&
        !arena_verify_pending_ && active_cpu_leases_.empty() &&
        !cpu_result_packet_host_ && !cpu_result_host_read_ && !cpu_host_result_active_;
}

int Qwen4ExpExpertExecutor::StartArenaVerifyRows(const Qwen4ExpHostArena& arena, int layer, int rows,
    const int* ids, const sycl::event& route_event) {
    if (!CanExecuteArenaVerifyRows(arena, rows, top_k_) || !ids || layer < 0 || layer >= arena.NumLayers() ||
        resident_input_event_ == sycl::event{})
        throw std::logic_error("Invalid arena multi-row verification route");
    const int experts = arena.NumExpertsPerLayer();
    try {
        // The previous CPU result transfer still reads these pinned arenas.
        static const bool deferArenaWait = aila::env::read_flag("AILA_Q4E_CPU_DEFER_ARENA_WAIT", false);
        const sycl::event transferReader = deferArenaWait ? prev_cpu_h2d_ev_ : sycl::event{};
        if (!deferArenaWait && prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
        prev_cpu_h2d_ev_ = {}; active_cpu_leases_.clear();
        cold_layer_ = layer; cold_source_slots_ = {}; cold_mapped_count_ = 0; cold_cached_count_ = 0;
        // The arena is the native RAM complement of the GPU residency table: a
        // non-null owner is a CPU cold expert. Record the immutable per-layer
        // source table once; the worker reads only it and never the arena.
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
        // Native large-page sources need no mmap prefetch, matching StartArenaCold.
        job.route_sources_resident = true;
        job.route_sources = arena_verify_sources_.data();
        for (int e = 0; e < experts; ++e)
            job.route_cpu_eligible[e] = arena.GetExpertPointer(layer, e) != nullptr;
        for (int rank = 0; rank < rows * top_k_; ++rank) {
            const int id = ids[rank];
            // Mirror the worker's own admission test exactly.
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
    // finish() is the publication fence for rank metadata and CPU output rows.
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
    if (!cold_count_) { active_cpu_leases_.clear(); cold_source_slots_ = {}; return {}; }
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
    if (cache_->tracing()) {
        cache_->trace_host_interval("cold_cpu_compute", cold_layer_, result.begin_ns, result.end_ns);
        cache_->trace_host_interval("cold_cpu_join", cold_layer_, join, steady_ns());
        cache_->TraceDeviceEvent("cold_input_copy", cold_layer_, resident_input_event_);
    }
    for (int c = 0; c < cold_count_; ++c) if (cpu_observer_) {
        RouteRankEntry entry;
        entry.token_idx = cold_ranks_[c] / top_k_; entry.rank = cold_ranks_[c] % top_k_;
        entry.expert_id = cold_ids_[c]; entry.is_gpu = false;
        cpu_observer_(cpu_observer_state_, *ctx_, cold_layer_, entry, cold_source_slots_[c],
            x_host_ + size_t(entry.token_idx) * hidden_size_,
            scratch_h_ + size_t(c) * intermediate_size_, cpu_out_host_ + size_t(c) * hidden_size_);
    }
    // The worker emits ranks in ascending order, so contiguous spans coalesce.
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
    active_cpu_leases_.clear(); cold_source_slots_ = {}; cold_count_ = 0;
    return prev_cpu_h2d_ev_;
}

LayerMoePlan& Qwen4ExpExpertExecutor::plan_layer(
    int layer,
    const int* router_host_ids,
    const float* router_host_weights,
    int token_count,
    int top_k,
    bool enable_hetero) {
    if (!initialized_ || !cache_) {
        throw std::logic_error("Qwen4ExpExpertExecutor: not initialized");
    }

    active_plan_.clear();
    active_cpu_leases_.clear();
    active_plan_.layer = layer;
    active_plan_.token_count = token_count;
    active_plan_.top_k = top_k;
    if (token_count <= 0 || !router_host_ids) {
        return active_plan_;
    }
    if (token_count > max_seq_len_ || top_k <= 0 || top_k > top_k_) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: route exceeds execution capacity");
    }

    const int total = token_count * top_k;
    active_plan_.total_tasks = total;

    // H3: Determine effective CPU credits for this layer
    int effective_credits = !cpu_fixed_credits_ && straggler_penalty_active_ ? std::max(0, max_cpu_credits_ - 1) : max_cpu_credits_;
    // A one-credit policy must not remain disabled forever after warmup.
    if (!cpu_fixed_credits_ && enable_hetero && token_count <= max_cpu_tokens_ && straggler_penalty_active_ && max_cpu_credits_ == 1) {
        if (cpu_retry_countdown_ > 0) --cpu_retry_countdown_;
        else { effective_credits = 1; cpu_retry_countdown_ = 8; }
    }
    const bool allow_cpu_dispatch = enable_hetero && (token_count <= max_cpu_tokens_) && (effective_credits > 0);
    const bool batch_prepare = allow_cpu_dispatch && cpu_placement_ == 2 && cpu_batch_prepare_;
    static_assert(ops::cpu::kMaxExpertBatchTasks <= Qwen4ExpExpertCache::kMaxHostPrepareTasks);
    std::array<HostExpertLease, ops::cpu::kMaxExpertBatchTasks> prepared_leases;
    std::array<int, ops::cpu::kMaxExpertBatchTasks> prepared_ids{}, prepared_ranks{};
    int prepared_count = 0;
    if (batch_prepare) {
        for (int idx = 0; idx < total && prepared_count < effective_credits; ++idx) {
            const auto res = cache_->query_residency(layer, router_host_ids[idx]);
            if (!cpu_wait_prefetch_ && (res.gpu_copying || (!res.host_ready && res.host_filling))) continue;
            prepared_ids[prepared_count] = router_host_ids[idx];
            prepared_ranks[prepared_count++] = idx;
        }
        if (prepared_count > 0) {
            const auto begin = steady_ns();
            stats_.cpu_host_prepared_experts += cache_->prepare_host_batch(layer, prepared_ids.data(), prepared_count, prepared_leases.data(), cpu_wait_prefetch_);
            stats_.cpu_host_prepare_ns += steady_ns() - begin;
            ++stats_.cpu_host_prepare_batches;
        }
    }

    for (int t = 0; t < token_count; ++t) {
        for (int r = 0; r < top_k; ++r) {
            const int idx = t * top_k + r;
            const int expert_id = router_host_ids[idx];
            const float weight = router_host_weights ? router_host_weights[idx] : 1.0f;

            RouteRankEntry entry;
            entry.token_idx = t;
            entry.rank = r;
            entry.expert_id = expert_id;
            entry.route_weight = weight;
            entry.result_idx = idx;

            auto res = cache_->query_residency(layer, expert_id);
            const bool cpu_credit = allow_cpu_dispatch && active_plan_.cpu_tasks < effective_credits;
            if (!batch_prepare && cpu_credit && cpu_placement_ == 2 && !res.host_ready && !res.host_filling && !res.gpu_copying) {
                const auto begin = steady_ns();
                cache_->prepare_host(layer, expert_id);
                stats_.cpu_host_prepare_ns += steady_ns() - begin;
                res = cache_->query_residency(layer, expert_id);
            }
            int prepared_idx = -1;
            if (batch_prepare) {
                for (int c = 0; c < prepared_count; ++c) if (prepared_ranks[c] == idx) { prepared_idx = c; break; }
            }
            const bool choose_cpu = batch_prepare
                ? prepared_idx >= 0 && prepared_leases[prepared_idx].is_valid()
                : cpu_credit && res.host_ready && !res.gpu_copying && (cpu_placement_ > 0 || !res.gpu_ready);
            if (res.gpu_ready && !choose_cpu) {
                // Hit in GPU cache: always keep on GPU (zero PCIe transfer needed)
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                entry.gpu_generation = res.gpu_generation;
                active_plan_.entries.push_back(entry);
                active_plan_.ready_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.ready_gpu_tasks;
            } else if (choose_cpu) {
                // GPU miss, but Host-Ready and within CPU admission credits!
                // Attempt to acquire thread-safe host lease
                auto lease = batch_prepare ? std::move(prepared_leases[prepared_idx]) : cache_->acquire_host_lease(layer, expert_id);
                if (lease.is_valid()) {
                    entry.is_gpu = false;
                    entry.gpu_slot = -1;
                    active_plan_.entries.push_back(entry);
                    active_plan_.cpu_indices.push_back(idx);
                    active_cpu_leases_.push_back(std::move(lease));
                    ++active_plan_.cpu_tasks;
                } else {
                    // Lease failed (e.g. reader race or eviction): fall back to GPU pending
                    entry.is_gpu = true;
                    entry.gpu_slot = -1;
                    active_plan_.entries.push_back(entry);
                    active_plan_.pending_gpu_indices.push_back(idx);
                    ++active_plan_.gpu_tasks;
                    ++active_plan_.pending_gpu_tasks;
                }
            } else {
                // GPU pending (copying or needs DMA from host/disk)
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                active_plan_.entries.push_back(entry);
                active_plan_.pending_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.pending_gpu_tasks;
            }
        }
    }

    return active_plan_;
}

void Qwen4ExpExpertExecutor::ensure_gpu_pending(LayerMoePlan& plan) {
    if (!cache_ || plan.gpu_tasks == 0) {
        return;
    }
    const auto acquire_start = std::chrono::steady_clock::now();
    for (int idx : plan.ready_gpu_indices) {
        const auto& entry = plan.entries[idx];
        const auto res = cache_->query_residency(plan.layer, entry.expert_id);
        if (!res.gpu_ready || res.gpu_slot != entry.gpu_slot ||
            res.gpu_generation != entry.gpu_generation) {
            throw std::logic_error("Qwen4ExpExpertExecutor: stale ready expert plan");
        }
    }

    int* pending_experts = pending_experts_buf_.data();
    int* assigned_slots = assigned_slots_buf_.data();

    int num_pending = 0;
    for (const auto& entry : plan.entries) {
        if (!entry.is_gpu) continue;
        int eid = entry.expert_id;
        bool found = false;
        for (int i = 0; i < num_pending; ++i) {
            if (pending_experts[i] == eid) {
                found = true;
                break;
            }
        }
        if (!found) {
            pending_experts[num_pending++] = eid;
        }
    }

    for (int i = 0; i < num_pending; ++i) assigned_slots[i] = -1;

    acquire_begin_ns_ = steady_ns();
    sycl::event dma_ev = cache_->acquire(plan.layer, pending_experts,
                                         num_pending,
                                         assigned_slots);
    acquire_end_ns_ = steady_ns();
    plan.final_dma_event = dma_ev;

    // Include ready keys in acquire's reservation set, even when the cache is full.
    for (auto& entry : plan.entries) {
        if (!entry.is_gpu) continue;
        int eid = entry.expert_id;
        for (int i = 0; i < num_pending; ++i) {
            if (pending_experts[i] == eid) {
                entry.gpu_slot = assigned_slots[i];
                entry.gpu_generation = cache_->get_slot_generation(assigned_slots[i]);
                entry.dma_event = cache_->get_slot_dma_event(assigned_slots[i]);
                break;
            }
        }
    }
    stats_.gpu_acquire_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - acquire_start).count();
}

sycl::event Qwen4ExpExpertExecutor::execute_hetero_decode(
    Context& ctx,
    LayerMoePlan& plan,
    const bf16* x_dev,
    bf16* moe_h_dev,
    const float* router_weights_host,
    bf16* y_dev,
    int token_count,
    int top_k,
    const bf16* shared_out,
    const bf16* shared_gate,
    bool uses_nested,
    bool defer_gpu_acquire,
    const ExpertPrefetchRequest* next_prefetch) {
    if (!initialized_) throw std::logic_error("Qwen4ExpExpertExecutor: execution before initialization");
    // CPU admission is independent of the preallocated GPU/merge capacity.
    if (token_count <= 0 || token_count > 4 || token_count > max_seq_len_) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: execution exceeds buffer capacity");
    }
    if (token_count != plan.token_count || top_k != plan.top_k) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: execution shape differs from plan");
    }
    if (plan.cpu_tasks < 0 || plan.cpu_tasks > ops::cpu::kMaxExpertBatchTasks ||
        static_cast<size_t>(plan.cpu_tasks) != active_cpu_leases_.size())
        throw std::logic_error("Qwen4ExpExpertExecutor: CPU lease count differs from plan");
    auto validate_entry = [&](const RouteRankEntry& entry) {
        const auto res = cache_->query_residency(plan.layer, entry.expert_id);
        if (!res.gpu_resident || res.gpu_slot != entry.gpu_slot ||
            res.gpu_generation != entry.gpu_generation) {
            throw std::logic_error("Qwen4ExpExpertExecutor: stale GPU expert plan");
        }
    };
    if (defer_gpu_acquire) {
        for (int idx : plan.ready_gpu_indices) validate_entry(plan.entries[idx]);
    } else {
        for (const auto& entry : plan.entries) {
            if (entry.is_gpu) validate_entry(entry);
        }
    }
    struct ExecutionGuard {
        Context& ctx;
        CpuWorker* worker;
        std::vector<HostExpertLease>& leases;
        bool submitted = false;
        ~ExecutionGuard() {
            if (!submitted) {
                if (worker) worker->drain();
                try { ctx.synchronize(); } catch (...) {}
                leases.clear();
            }
        }
    } execution_guard{ctx, cpu_worker_.get(), active_cpu_leases_};
    bool prefetch_started = false;
    auto start_next_prefetch = [&] {
        if (next_prefetch && !prefetch_started) {
            cache_->prefetch_async(next_prefetch->layer, next_prefetch->expert_ids,
                                   next_prefetch->count, next_prefetch->quota);
            prefetch_started = true;
        }
    };

    // Drain previous layer's CPU result copy if still in-flight to prevent host buffer reuse race
    if(cpu_host_result_active_)throw std::logic_error("CPU host result has no registered merge reader");
    if(cpu_output_reader_event_!=sycl::event{}) {
        cpu_output_reader_event_.wait_and_throw();cpu_output_reader_event_={};
    }
    if (prev_cpu_h2d_ev_ != sycl::event{}) {
        prev_cpu_h2d_ev_.wait();
        prev_cpu_h2d_ev_ = sycl::event{};
    }

    const size_t total_acts = static_cast<size_t>(token_count) * top_k;
    // AILA_Q4E_MOE_COPY_VIA_COPYQ: keep the three per-layer CPU-complement host
    // copies off the compute queue. copyq_active is the single decision point;
    // when it is false every site below is byte-for-byte the previous call.
    const bool copyq_active = copy_queue() != nullptr;
    bool weights_copy_on_copyq = false;
    sycl::event weights_copy_ev{};
    // Copy router weights to device for ordered rank accumulation
    if (copyq_active) {
        weights_copy_ev = q4e::submit_host_copy(ctx, copy_queue(), weights_dev_,
            router_weights_host, total_acts * sizeof(float));
        weights_copy_on_copyq = true;
    } else {
        ctx.queue().memcpy(weights_dev_, router_weights_host, total_acts * sizeof(float));
    }

    // 1. If CPU tasks exist, initiate async fetch of activations x_dev to pinned host memory
    sycl::event x_d2h_ev{};
    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        // Consumed by the CPU worker through job.activation below (and by the
        // synchronous wait at the join), so the copy-queue event is the fence.
        x_d2h_ev = q4e::submit_host_copy(ctx, copy_queue(), x_host_, x_dev,
            static_cast<size_t>(token_count) * hidden_size_ * sizeof(bf16));
    }
    const bool async_cpu = cpu_worker_ && plan.cpu_tasks > 0;

    // Opt-in W4A8 adaptive decode: identical scheduling, signed-Q8 activations.
    // h_rank_offset is the expert-slice origin inside the shared expert-major
    // scratch, so the ready/pending split keeps its existing layout.
    const bool q8 = q8_adaptive_;
    const size_t hq_stride = static_cast<size_t>(intermediate_size_);
    const size_t hs_stride = static_cast<size_t>(intermediate_size_ / 64);
    auto launch_gate_up = [&](const ops::MoeExpertSlotDev* slots, int count, int h_rank_offset,
                              const int* tokens, const sycl::event* dep) -> sycl::event {
        bf16* h = moe_h_dev + static_cast<size_t>(h_rank_offset) * intermediate_size_;
        if (!q8) {
            return ops::moe_decode_gate_up_swiglu_grouped(
                ctx, slots, nullptr, x_dev, h, count, hidden_size_, intermediate_size_, 64,
                dep, tokens, uses_nested);
        }
        return ops::moe_decode_gate_up_swiglu_grouped_q8(
            ctx, slots, xq_dev_, xs_dev_, h, hq_dev_ + static_cast<size_t>(h_rank_offset) * hq_stride,
            hs_dev_ + static_cast<size_t>(h_rank_offset) * hs_stride, count, hidden_size_,
            intermediate_size_, tokens, dep, uses_nested);
    };
    auto launch_down = [&](const ops::MoeExpertSlotDev* slots, const int* ranks, int count,
                           int h_rank_offset, const int* tokens, const sycl::event& dep) -> sycl::event {
        if (!q8) {
            return ops::moe_decode_down_to_rank_f32(
                ctx, slots, ranks, moe_h_dev + static_cast<size_t>(h_rank_offset) * intermediate_size_,
                down_acc_dev_, count, top_k, intermediate_size_, hidden_size_, 64, tokens, nullptr,
                &dep, uses_nested);
        }
        return ops::moe_decode_down_to_rank_f32_q8(
            ctx, slots, ranks, hq_dev_ + static_cast<size_t>(h_rank_offset) * hq_stride,
            hs_dev_ + static_cast<size_t>(h_rank_offset) * hs_stride, down_acc_dev_, count, top_k,
            intermediate_size_, hidden_size_, tokens, &dep, uses_nested);
    };

    acquire_begin_ns_ = acquire_end_ns_ = 0;
    if (async_cpu) {
        CpuWorker::Job job;
        job.count = plan.cpu_tasks; job.hidden = hidden_size_; job.intermediate = intermediate_size_; job.tokens = token_count;
        job.x = x_host_; job.h = scratch_h_; job.out = cpu_out_host_; job.activation = x_d2h_ev;
        for (int c = 0; c < plan.cpu_tasks; ++c)
            job.tasks[c] = {active_cpu_leases_[c].slot_host(), plan.entries[plan.cpu_indices[c]].token_idx};
        cpu_worker_->submit(job);
    }

    // 2. Launch GPU execution FIRST so GPU hardware runs asynchronously while CPU computes
    sycl::event down_ev{};
    if (plan.gpu_tasks > 0) {
        cache_->record_first_consumer_start();
        stats_.gpu_tasks_dispatched += plan.gpu_tasks;
        const int num_slots = cache_->num_slots();
        if (q8) {
            // Quantize every decode row once; both subsets read this arena.
            ops::quantize_moe_input_q8_64_rows(ctx, x_dev, xq_dev_, xs_dev_, token_count, hidden_size_);
        }

        if (plan.ready_gpu_tasks > 0 && plan.pending_gpu_tasks > 0) {
            const int ready_count = plan.ready_gpu_tasks;
            const int pending_count = plan.pending_gpu_tasks;

            // 1. Setup ready subset at indices [0 .. ready_count - 1]
            for (int r = 0; r < ready_count; ++r) {
                int idx = plan.ready_gpu_indices[r];
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid ready gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[r] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[r] = plan.entries[idx].rank;
                gpu_tokens_host_[r] = plan.entries[idx].token_idx;
            }

            const ops::MoeExpertSlotDev* eff_slots = gpu_slots_host_;
            const int* eff_ranks = gpu_ranks_host_;
            const int* eff_tokens = gpu_tokens_host_;
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_, gpu_slots_host_, ready_count * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_, gpu_ranks_host_, ready_count * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_, gpu_tokens_host_, ready_count * sizeof(int));
                eff_slots = gpu_slots_dev_;
                eff_ranks = gpu_ranks_dev_;
                eff_tokens = gpu_tokens_dev_;
            }

            sycl::event gu_ev_ready = launch_gate_up(eff_slots, ready_count, 0, eff_tokens, nullptr);

            sycl::event down_ev_ready = launch_down(eff_slots, eff_ranks, ready_count, 0,
                eff_tokens, gu_ev_ready);

            for (int idx : plan.ready_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev_ready);
            }
            plan.ready_subset_event = down_ev_ready;
            if (defer_gpu_acquire) {
                // Ready compute is already queued while the host services misses.
                ensure_gpu_pending(plan);
                start_next_prefetch();
                cache_->record_first_consumer_start();
                ++stats_.ready_first_layers;
            }

            for (int p = 0; p < pending_count; ++p) {
                int idx = plan.pending_gpu_indices[p];
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid pending gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[ready_count + p] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[ready_count + p] = plan.entries[idx].rank;
                gpu_tokens_host_[ready_count + p] = plan.entries[idx].token_idx;
            }
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_ + ready_count, gpu_slots_host_ + ready_count,
                                   pending_count * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_ + ready_count, gpu_ranks_host_ + ready_count, pending_count * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_ + ready_count, gpu_tokens_host_ + ready_count, pending_count * sizeof(int));
            }

            sycl::event gu_ev_pending = launch_gate_up(eff_slots + ready_count, pending_count,
                ready_count, eff_tokens + ready_count,
                plan.final_dma_event != sycl::event{} ? &plan.final_dma_event : nullptr);

            sycl::event down_ev_pending = launch_down(eff_slots + ready_count, eff_ranks + ready_count,
                pending_count, ready_count, eff_tokens + ready_count, gu_ev_pending);

            for (int idx : plan.pending_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev_pending);
            }

            down_ev = down_ev_pending;
        } else {
            if (defer_gpu_acquire) {
                ensure_gpu_pending(plan);
                start_next_prefetch();
                cache_->record_first_consumer_start();
            }
            int g_idx = 0;
            for (int idx : plan.ready_gpu_indices) {
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid ready gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[g_idx] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
                gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
                ++g_idx;
            }
            for (int idx : plan.pending_gpu_indices) {
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid pending gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[g_idx] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
                gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
                ++g_idx;
            }

            const ops::MoeExpertSlotDev* eff_slots = gpu_slots_host_;
            const int* eff_ranks = gpu_ranks_host_;
            const int* eff_tokens = gpu_tokens_host_;
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_, gpu_slots_host_, plan.gpu_tasks * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_, gpu_ranks_host_, plan.gpu_tasks * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_, gpu_tokens_host_, plan.gpu_tasks * sizeof(int));
                eff_slots = gpu_slots_dev_;
                eff_ranks = gpu_ranks_dev_;
                eff_tokens = gpu_tokens_dev_;
            }

            const sycl::event* dep = (plan.pending_gpu_tasks > 0 && plan.final_dma_event != sycl::event{})
                ? &plan.final_dma_event : nullptr;

            sycl::event gu_ev = launch_gate_up(eff_slots, plan.gpu_tasks, 0, eff_tokens, dep);

            down_ev = launch_down(eff_slots, eff_ranks, plan.gpu_tasks, 0, eff_tokens, gu_ev);

            for (int idx : plan.ready_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
            }
            for (int idx : plan.pending_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
            }
        }
    }

    start_next_prefetch();

    // 3. Concurrent CPU execution for Host-ready miss experts (runs in parallel with GPU!)
    sycl::event last_cpu_h2d_ev{};
    bool cpu_result_copy_issued = false;
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    bool async_join_blocked = false;

    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        auto copy_results = [&](int begin, int end) {
            const auto submit_start = steady_ns();
            const size_t bytes = static_cast<size_t>(end - begin) * hidden_size_ * sizeof(float);
            float* dst = down_acc_dev_ + static_cast<size_t>(plan.cpu_indices[begin]) * hidden_size_;
            const float* src = cpu_out_host_ + static_cast<size_t>(begin) * hidden_size_;
            last_cpu_h2d_ev = q4e::submit_host_copy(ctx, copy_queue(), dst, src, bytes);
            cpu_result_copy_issued = true;
            ++stats_.cpu_result_copies;
            stats_.cpu_result_bytes += bytes;
            stats_.cpu_result_submit_ns += steady_ns() - submit_start;
        };
        if (async_cpu) {
            async_join_blocked = !cpu_worker_->complete();
            const auto join_start = steady_ns();
            const auto completed = cpu_worker_->finish();
            stats_.cpu_join_wait_ns += steady_ns() - join_start;
            stats_.cpu_activation_wait_ns += completed.activation_ns;
            stats_.cpu_compute_ns += completed.end_ns - completed.begin_ns;
            if (acquire_end_ns_ > acquire_begin_ns_) {
                const auto start = std::max(acquire_begin_ns_, completed.begin_ns);
                const auto end = std::min(acquire_end_ns_, completed.end_ns);
                if (end > start) stats_.cpu_acquire_overlap_ns += end - start;
            }
            ++stats_.cpu_async_layers;
        } else {
            if (x_d2h_ev != sycl::event{}) x_d2h_ev.wait_and_throw();
            stats_.cpu_activation_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - cpu_t0).count();
        }

        for (int c = 0; c < plan.cpu_tasks; ++c) {
            const int plan_idx = plan.cpu_indices[c];
            const auto& entry = plan.entries[plan_idx];
            const int t = entry.token_idx;
            const bf16* token_x = x_host_ + static_cast<size_t>(t) * hidden_size_;
            float* out_slot = cpu_out_host_ + static_cast<size_t>(c) * hidden_size_;

            bf16* task_h = scratch_h_ + static_cast<size_t>(c) * intermediate_size_;
            if (!async_cpu) {
                const auto compute_start = steady_ns();
                compute_expert(active_cpu_leases_[c].slot_host(), token_x, task_h, out_slot, hidden_size_, intermediate_size_);
                stats_.cpu_compute_ns += steady_ns() - compute_start;
            }
            if (cpu_observer_) {
                cpu_observer_(cpu_observer_state_, ctx, plan.layer, entry,
                              active_cpu_leases_[c].slot_host(), token_x, task_h, out_slot);
            }

            // Release lease immediately after reading weights
            active_cpu_leases_[c].release();

            if (!cpu_coalesce_results_) copy_results(c, c + 1);
        }
        if (cpu_coalesce_results_) {
            for (int begin = 0; begin < plan.cpu_tasks; ) {
                int end = begin + 1;
                while (end < plan.cpu_tasks && plan.cpu_indices[end] == plan.cpu_indices[end - 1] + 1) ++end;
                copy_results(begin, end);
                begin = end;
            }
        }

        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        const uint64_t cpu_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_t1 - cpu_t0).count();
        stats_.cpu_tasks_dispatched += plan.cpu_tasks;
        stats_.cpu_tasks_completed += plan.cpu_tasks;
        stats_.cpu_max_batch_tasks = std::max(stats_.cpu_max_batch_tasks, static_cast<uint64_t>(plan.cpu_tasks));
        active_cpu_leases_.clear();

        // Async work that is not ready at merge admission exposes a host wait.
        // GPU status sampled after joining cannot classify that earlier overlap.
        // AILA_Q4E_MOE_COPY_VIA_COPYQ also removes this compute-queue status
        // query, and with it the straggler penalty it feeds: the B arm measures
        // that heuristic's value as well.
        if (plan.gpu_tasks > 0 && !copyq_active) {
            const auto gpu_status = down_ev.get_info<sycl::info::event::command_execution_status>();
            if (async_cpu ? async_join_blocked : gpu_status == sycl::info::event_command_status::complete) {
                // Penalize an observed blocking join, or the original synchronous tail.
                stats_.cpu_straggler_count++;
                consecutive_stragglers_++;
                stats_.cpu_straggler_interval_ns += cpu_elapsed_ns;
                if (consecutive_stragglers_ >= 2) {
                    set_straggler_penalty(true);
                }
            } else {
                // No blocking join observed for async, or GPU still active for sync.
                stats_.cpu_overlapped_count++;
                consecutive_stragglers_ = 0;
                set_straggler_penalty(false);
            }
        }
    }

    // 5. Ordered Rank Merge: merges all ranks in strict rank order (0..top_k-1)
    // The merge takes a single dependency pointer, so with copy-queue routing the
    // GPU subset event and both copy events are folded into one barrier; the
    // gate-off path keeps the original down_ev / last_cpu_h2d_ev selection.
    sycl::event merge_dep{};
    const sycl::event* merge_dep_ptr = nullptr;
    if (copyq_active) {
        std::vector<sycl::event> deps;
        if (plan.gpu_tasks > 0) deps.push_back(down_ev);
        if (weights_copy_on_copyq) deps.push_back(weights_copy_ev);
        if (cpu_result_copy_issued) deps.push_back(last_cpu_h2d_ev);
        merge_dep = q4e::join_dependencies(ctx, deps);
        if (!deps.empty()) merge_dep_ptr = &merge_dep;
    } else {
        merge_dep_ptr = plan.gpu_tasks > 0 ? &down_ev
            : (last_cpu_h2d_ev != sycl::event{} ? &last_cpu_h2d_ev : nullptr);
    }
    sycl::event merge_ev = ops::moe_ordered_accumulate_down_rows(
        ctx, down_acc_dev_, weights_dev_, y_dev, token_count, top_k,
        hidden_size_, shared_out, shared_gate, merge_dep_ptr);

    prev_cpu_h2d_ev_ = last_cpu_h2d_ev;
    stats_.layers_hetero_executed++;
    execution_guard.submitted = true;
    return merge_ev;
}

void Qwen4ExpExpertExecutor::record_execution_complete(LayerMoePlan& plan, sycl::event finish_event) {
    if (!cache_) return;
    const int max_slot = cache_->num_slots();
    if (static_cast<int>(slot_seen_.size()) < max_slot) {
        slot_seen_.resize(max_slot, false);
    }
    std::fill(slot_seen_.begin(), slot_seen_.end(), false);

    for (int idx : plan.ready_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !slot_seen_[slot]) {
            slot_seen_[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
    for (int idx : plan.pending_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !slot_seen_[slot]) {
            slot_seen_[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
}

} // namespace aila
