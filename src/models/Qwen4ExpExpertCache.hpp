#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"
#include "utils/SafeTensors.hpp"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

class ExpertFillPool;

enum class ExpertSlotState : uint8_t {
    Empty = 0,
    Filling = 1,
    Copying = 2,
    Ready = 3,
    Faulted = 4
};

struct GpuSlotRecord {
    uint64_t key = UINT64_MAX;
    uint64_t generation = 0;
    ExpertSlotState state = ExpertSlotState::Empty;
    int fill_owner = -1; // -1: none, 0: demand, 1: prefetch
    sycl::event consumer_fence{};
    sycl::event dma_event{};
};

struct HostSlotRecord {
    uint64_t key = UINT64_MAX;
    std::atomic<uint64_t> generation{0};
    int fill_owner = -1; // -1: none, 0: demand, 1: prefetch
    sycl::event source_reader_fence{};
    std::atomic<int32_t> cpu_readers{0};

    HostSlotRecord() = default;
    HostSlotRecord(const HostSlotRecord& o)
        : key(o.key), generation(o.generation.load(std::memory_order_relaxed)),
          fill_owner(o.fill_owner), source_reader_fence(o.source_reader_fence),
          cpu_readers(o.cpu_readers.load(std::memory_order_relaxed)) {}
    HostSlotRecord& operator=(const HostSlotRecord& o) {
        if (this != &o) {
            key = o.key;
            generation.store(o.generation.load(std::memory_order_relaxed), std::memory_order_relaxed);
            fill_owner = o.fill_owner;
            source_reader_fence = o.source_reader_fence;
            cpu_readers.store(o.cpu_readers.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
    HostSlotRecord(HostSlotRecord&& o) noexcept
        : key(o.key), generation(o.generation.load(std::memory_order_relaxed)),
          fill_owner(o.fill_owner), source_reader_fence(std::move(o.source_reader_fence)),
          cpu_readers(o.cpu_readers.load(std::memory_order_relaxed)) {}
    HostSlotRecord& operator=(HostSlotRecord&& o) noexcept {
        if (this != &o) {
            key = o.key;
            generation.store(o.generation.load(std::memory_order_relaxed), std::memory_order_relaxed);
            fill_owner = o.fill_owner;
            source_reader_fence = std::move(o.source_reader_fence);
            cpu_readers.store(o.cpu_readers.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

struct ExpertKey {
    int layer = -1;
    int expert_id = -1;
    bool operator==(const ExpertKey& o) const { return layer == o.layer && expert_id == o.expert_id; }
    bool operator!=(const ExpertKey& o) const { return !(*this == o); }
};

struct ExpertResidency {
    int layer = -1;
    int expert_id = -1;
    bool gpu_resident = false;
    bool gpu_ready = false;
    bool gpu_copying = false;
    int gpu_slot = -1;
    uint64_t gpu_generation = 0;
    sycl::event gpu_dma_event{};

    bool host_resident = false;
    bool host_ready = false;
    bool host_filling = false;
    int host_slot = -1;
    uint64_t host_generation = 0;
    int host_cpu_readers = 0;
};

struct FillTicket {
    int layer = -1;
    int expert_id = -1;
    int hslot = -1;
    uint64_t generation = 0;
    bool ready = false;
};

struct GpuExpertTicket {
    int layer = -1;
    int expert_id = -1;
    int gpu_slot = -1;
    uint64_t gpu_generation = 0;
    sycl::event dma_event{};
    bool ready = false;
};

class Qwen4ExpExpertCache;

class HostExpertLease {
public:
    HostExpertLease() = default;
    HostExpertLease(Qwen4ExpExpertCache* cache, int layer, int expert_id, int hslot, uint64_t gen, ops::MoeExpertSlotHost slot_host)
        : cache_(cache), layer_(layer), expert_id_(expert_id), hslot_(hslot), generation_(gen), slot_host_(slot_host) {}
    ~HostExpertLease();

    HostExpertLease(const HostExpertLease&) = delete;
    HostExpertLease& operator=(const HostExpertLease&) = delete;

    HostExpertLease(HostExpertLease&& o) noexcept;
    HostExpertLease& operator=(HostExpertLease&& o) noexcept;

    bool is_valid() const { return cache_ != nullptr && hslot_ >= 0; }
    explicit operator bool() const { return is_valid(); }
    int layer() const { return layer_; }
    int expert_id() const { return expert_id_; }
    int hslot() const { return hslot_; }
    uint64_t generation() const { return generation_; }
    const ops::MoeExpertSlotHost& slot_host() const { return slot_host_; }

    void release();

private:
    Qwen4ExpExpertCache* cache_ = nullptr;
    int layer_ = -1;
    int expert_id_ = -1;
    int hslot_ = -1;
    uint64_t generation_ = 0;
    ops::MoeExpertSlotHost slot_host_{};
};

struct ExpertCacheStats {
    uint64_t hit_count = 0;
    uint64_t miss_count = 0;
    uint64_t total_transferred_bytes = 0;
    uint64_t fill_ns = 0;     // host-side staging fill (mmap read + optional absmax dequant)
    uint64_t dma_wait_ns = 0; // waiting for h2d copies to drain
    uint64_t prefetch_issued = 0;  // experts staged by prefetch_async
    uint64_t prefetch_used = 0;    // ... of which served a real miss in acquire
    uint64_t prefetch_wait_ns = 0; // acquire waiting on an unfinished prefetch fill
    uint64_t host_pool_hits = 0;   // served directly from hot host RAM pool (zero CPU copy)
    uint64_t host_pool_misses = 0; // required fill into host pool from mmap

    // Phase B metrics (B3, B6)
    uint64_t prefetch_issued_bytes = 0;
    uint64_t prefetch_useful_bytes = 0;
    uint64_t prefetch_cancelled_bytes = 0;
    uint64_t prefetch_wasted_bytes = 0;
    uint64_t demand_exposed_wait_ns = 0;
    uint64_t demand_microbatches = 0;

    double hit_rate() const {
        uint64_t total = hit_count + miss_count;
        return total > 0 ? static_cast<double>(hit_count) / static_cast<double>(total) : 0.0;
    }

    double prefetch_utility_ratio() const {
        return prefetch_issued_bytes > 0
            ? static_cast<double>(prefetch_useful_bytes) / static_cast<double>(prefetch_issued_bytes)
            : 0.0;
    }
};

class Qwen4ExpExpertCache {
public:
    Qwen4ExpExpertCache();
    ~Qwen4ExpExpertCache();

    Qwen4ExpExpertCache(const Qwen4ExpExpertCache&) = delete;
    Qwen4ExpExpertCache& operator=(const Qwen4ExpExpertCache&) = delete;

    Qwen4ExpExpertCache(Qwen4ExpExpertCache&& other) noexcept;
    Qwen4ExpExpertCache& operator=(Qwen4ExpExpertCache&& other) noexcept;

    // Initialize expert cache
    // num_layers: number of MoE layers (e.g. 48)
    // cache_bytes: GPU memory budget for expert cache. If 0, reads AILA_Q4E_EXPERT_CACHE_MB (default 4000 MB).
    bool init(Context& ctx,
              ModelWeights& weights,
              int num_layers,
              size_t cache_bytes = 0,
              int num_experts_per_layer = 512,
              int hidden_size = 2560,
              int intermediate_size = 640,
              int blocksize = 64,
              std::string* error_message = nullptr,
              bool force_nested_codes = false);

    // Override or set layer expert references (useful for testing or manual binding)
    void set_layer_expert_ref(int layer,
                              const Bnb4BitHostWeightRef& gate_up,
                              const Bnb4BitHostWeightRef& down);

    // Acquire slots for layer and expert_ids.
    // count: number of active experts (e.g. top_k = 10)
    // out_slots: filled with slot indices in [0, num_slots - 1]
    // Crucial guarantee: slots acquired in the same batch will never evict each other.
    // A request with more unique experts than GPU slots is rejected before cache state changes.
    // If an exception occurs after a slot binding or DMA submission, the cache becomes poisoned;
    // callers must destroy/reinitialize it before using the GPU cache again.
    // Returns: SYCL event tracking completion of DMA transfers (or default-constructed host event if all hit).
    sycl::event acquire(int layer,
                        const int* expert_ids,
                        int count,
                        int* out_slots);

    // Starts filling predicted non-resident experts into a pinned prefetch staging area on
    // the worker threads and returns immediately. Binds no slot; the next acquire() joins
    // the fill and DMAs staged experts that actually miss instead of filling them again.
    // quota: maximum number of non-resident experts to stage. If <= 0, defaults to count.
    void prefetch_async(int layer, const int* expert_ids, int count, int quota = -1);
    void wait_prefetch();
    void wait_copy();

    // Moves resident experts to the eviction end of the LRU in the given order, so the last
    // one listed is evicted first. Non-resident experts are ignored.
    void demote(int layer, const int* expert_ids, int count);

    // Device slot descriptors for kernel launches
    ops::MoeExpertSlotDev get_slot_dev(int slot_id) const {
        if (faulted_) throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned; reinitialize before use");
        return slots_dev_[slot_id];
    }

    void get_slots_dev(const int* slot_ids, int count, ops::MoeExpertSlotDev* out_slot_devs) const {
        if (faulted_) throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned; reinitialize before use");
        for (int i = 0; i < count; ++i) {
            out_slot_devs[i] = slots_dev_[slot_ids[i]];
        }
    }

    const ExpertCacheStats& stats() const { return stats_; }
    bool contains(int layer, int expert_id) const {
        if (!initialized_ || faulted_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        return layer_expert_to_slot_[expert_flat_index(layer, expert_id)] >= 0;
    }
    bool host_contains(int layer, int expert_id) const {
        if (!initialized_ || !host_pool_enabled_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        return host_layer_expert_to_slot_[expert_flat_index(layer, expert_id)] >= 0;
    }
    bool host_is_protected(int layer, int expert_id) const {
        if (!initialized_ || !host_pool_enabled_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        int hslot = host_layer_expert_to_slot_[expert_flat_index(layer, expert_id)];
        if (hslot < 0 || hslot >= num_host_slots_) return false;
        return host_slot_is_protected_[hslot];
    }
    bool host_is_reserved(int layer, int expert_id) const {
        if (!initialized_ || !host_pool_enabled_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        int hslot = host_layer_expert_to_slot_[expert_flat_index(layer, expert_id)];
        if (hslot < 0 || hslot >= num_host_slots_) return false;
        return host_slot_reserved_[hslot];
    }
    bool host_has_dma_in_flight(int layer, int expert_id) const;
    void cancel_prefetch_for_testing() { cancel_prefetch(); }
    size_t host_protected_count() const { return host_protected_list_.size(); }
    size_t host_probationary_count() const { return host_probationary_list_.size(); }
    size_t host_empty_count() const { return host_empty_slots_.size(); }
    float host_protected_ratio() const { return protected_ratio_; }
    void set_host_protected_ratio(float ratio) { protected_ratio_ = std::max(0.0f, std::min(1.0f, ratio)); }
    void reset_stats() { stats_ = {}; }
    void log_stats(int layer = -1) const;

    ExpertSlotState get_slot_state(int slot_id) const;
    uint64_t get_slot_generation(int slot_id) const;
    ExpertSlotState get_host_slot_state(int hslot) const;
    uint64_t get_host_slot_generation(int hslot) const;
    void record_consumer_reader_fence(int slot_id, sycl::event fence);
    void record_consumer_reader_fences(const int* slot_ids, int count, sycl::event fence);
    void set_cost_aware_eviction(bool enable) { cost_aware_eviction_ = enable; }
    bool cost_aware_eviction() const { return cost_aware_eviction_; }

    int num_slots() const { return num_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    bool uses_nested_codes() const { return compact_nested_; }
    bool is_initialized() const { return initialized_ && !faulted_; }
    bool is_faulted() const { return faulted_; }

    size_t gu_packed_bytes() const { return gu_packed_bytes_; }
    size_t gu_absmax_count() const { return gu_absmax_count_; }
    size_t gu_absmax_bytes() const { return gu_absmax_bytes_; }
    size_t dn_packed_bytes() const { return dn_packed_bytes_; }
    size_t dn_absmax_count() const { return dn_absmax_count_; }
    size_t dn_absmax_bytes() const { return dn_absmax_bytes_; }

    const uint8_t* pinned_staging_ptr() const { return pinned_staging_ptr_; }
    bool host_pool_enabled() const { return host_pool_enabled_; }
    int num_host_slots() const { return num_host_slots_; }
    size_t host_pool_bytes() const { return host_pool_bytes_; }

    ExpertResidency query_residency(int layer, int expert_id) const;
    HostExpertLease acquire_host_lease(int layer, int expert_id);
    void release_host_lease(int hslot, uint64_t gen);
    ops::MoeExpertSlotHost get_slot_host(int hslot) const;
    ops::MoeExpertSlotHost get_expert_slot_host(int layer, int expert_id) const;
    int get_host_slot_cpu_readers(int hslot) const {
        if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return 0;
        return host_slot_records_[hslot].cpu_readers.load(std::memory_order_relaxed);
    }
    FillTicket prepare_host(int layer, int expert_id, int priority = 0);
    GpuExpertTicket ensure_gpu(int layer, int expert_id, int priority = 0);
    GpuExpertTicket ensure_gpu(const HostExpertLease& lease, int priority = 0);

private:
    void cleanup();
    void configure_layer_nested_metadata(int layer);
    void bind_slot_nested_metadata(int slot, int layer, int expert_id);
    // Fills part [part, nparts) of every slot segment for one expert.
    void load_expert_to_staging(int layer, int expert_id, uint8_t* staging_slot,
                                int part = 0, int nparts = 1);

    Context* ctx_ = nullptr;
    const ModelWeights* weights_ = nullptr;
    int num_layers_ = 0;
    int num_experts_per_layer_ = 512;
    int hidden_size_ = 2560;
    int intermediate_size_ = 640;
    int blocksize_ = 64;

    size_t gu_packed_bytes_ = 0;
    size_t gu_absmax_count_ = 0;
    size_t gu_absmax_bytes_ = 0;
    size_t dn_packed_bytes_ = 0;
    size_t dn_absmax_count_ = 0;
    size_t dn_absmax_bytes_ = 0;

    size_t gu_packed_offset_ = 0;
    size_t gu_absmax_offset_ = 0;
    size_t dn_packed_offset_ = 0;
    size_t dn_absmax_offset_ = 0;
    size_t slot_bytes_ = 0;
    bool compact_nested_ = false;

    int num_slots_ = 0;
    size_t dev_pool_bytes_ = 0;
    std::vector<uint8_t*> dev_pool_chunks_; // each <= AILA_Q4E_EXPERT_CHUNK_MB (L0 caps single USM allocs at ~4 GB)
    std::vector<uint8_t*> slot_dev_base_;
    std::vector<uint8_t*> layer_meta_allocs_;

    size_t staging_buffer_bytes_ = 0;
    uint8_t* pinned_staging_ptr_ = nullptr;
    uint8_t* prefetch_staging_ptr_ = nullptr; // prefetch_staging_slots_ slots
    int prefetch_staging_slots_ = 0;

    // Immediate-submission in-order queue for miss DMA (see acquire()).
    std::unique_ptr<sycl::queue> copy_q_;
    sycl::event last_dma_event_{};
    struct PrefetchItem {
        int layer = -1;
        int expert_id = -1;
        int hslot = -1;
        uint64_t expected_generation = 0;
        std::atomic<uint64_t> claimed_parts{0};
        std::atomic<int> parts_remaining{0};
        std::atomic<bool> cancelled{false};
        std::atomic<bool> ready{false};
        bool used = false;

        PrefetchItem() = default;
        PrefetchItem(int l, int e, int h, int parts, uint64_t gen = 0)
            : layer(l), expert_id(e), hslot(h), expected_generation(gen),
              claimed_parts(0), parts_remaining(parts), cancelled(false), ready(false), used(false) {}

        PrefetchItem(const PrefetchItem& o)
            : layer(o.layer), expert_id(o.expert_id), hslot(o.hslot),
              expected_generation(o.expected_generation),
              claimed_parts(o.claimed_parts.load(std::memory_order_relaxed)),
              parts_remaining(o.parts_remaining.load(std::memory_order_relaxed)),
              cancelled(o.cancelled.load(std::memory_order_relaxed)),
              ready(o.ready.load(std::memory_order_relaxed)),
              used(o.used) {}

        PrefetchItem& operator=(const PrefetchItem& o) {
            if (this != &o) {
                layer = o.layer;
                expert_id = o.expert_id;
                hslot = o.hslot;
                expected_generation = o.expected_generation;
                claimed_parts.store(o.claimed_parts.load(std::memory_order_relaxed), std::memory_order_relaxed);
                parts_remaining.store(o.parts_remaining.load(std::memory_order_relaxed), std::memory_order_relaxed);
                cancelled.store(o.cancelled.load(std::memory_order_relaxed), std::memory_order_relaxed);
                ready.store(o.ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
                used = o.used;
            }
            return *this;
        }
    };
    std::vector<PrefetchItem> prefetch_items_;
    bool prefetch_in_flight_ = false;

    std::unique_ptr<std::atomic<uint8_t>[]> host_slot_state_; // 0=Empty, 1=Filling, 2=Ready
    void cancel_prefetch();
    void ensure_prefetch_item_ready(PrefetchItem& item, int prefetch_idx);

    struct LayerWeights {
        Bnb4BitHostWeightRef gate_up{};
        Bnb4BitHostWeightRef down{};
        const float* gu_nested_absmax_dev = nullptr;
        const float* gu_nested_qmap_dev = nullptr;
        const float* dn_nested_absmax_dev = nullptr;
        const float* dn_nested_qmap_dev = nullptr;
        bool valid = false;
    };
    std::vector<LayerWeights> layer_weights_;

    inline int expert_flat_index(int layer, int expert_id) const {
        return layer * num_experts_per_layer_ + expert_id;
    }

    std::list<int> lru_list_;
    std::vector<std::list<int>::iterator> slot_iters_;
    std::vector<int> layer_expert_to_slot_;
    std::vector<uint64_t> slot_to_key_;
    std::vector<bool> slot_reserved_;
    std::vector<ops::MoeExpertSlotDev> slots_dev_;
    std::vector<GpuSlotRecord> gpu_slot_records_;

    ExpertCacheStats stats_{};
    bool initialized_ = false;
    bool faulted_ = false;
    bool cost_aware_eviction_ = true;

    // Host fill parallelism (AILA_Q4E_FILL_THREADS / AILA_Q4E_FILL_PARTS)
    // Hot Host RAM Pool (pinned USM host memory allocated in 1 GB chunks)
    bool host_pool_enabled_ = false;
    size_t host_pool_bytes_ = 0;
    int num_host_slots_ = 0;
    std::vector<void*> host_pool_chunks_;
    std::vector<uint8_t*> host_slot_ptrs_;
    std::vector<int> host_layer_expert_to_slot_;
    std::vector<uint64_t> host_slot_to_key_;
    std::vector<HostSlotRecord> host_slot_records_;
    std::vector<int> host_empty_slots_;
    std::list<int> host_probationary_list_;
    std::list<int> host_protected_list_;
    std::vector<std::list<int>::iterator> host_slot_iters_;
    std::vector<bool> host_slot_is_protected_;
    std::vector<bool> host_slot_reserved_;
    std::vector<bool> host_slot_in_batch_readers_;
    std::vector<int> batch_reserved_hslots_scratch_;
    std::vector<sycl::event> host_slot_dma_events_;
    float protected_ratio_ = 0.75f;
    std::unique_ptr<std::mutex> host_list_mutex_;
    void touch_host_slot(int hslot);
    int pick_victim_hslot();
    void add_batch_reader_hslot(int hslot);

    struct MissInfo {
        int batch_index;
        int layer;
        int expert_id;
        int slot_id;
        const uint8_t* prefilled; // staged by prefetch_async, or null
    };

    // Preallocated scratch to avoid hot-path heap allocations
    std::vector<MissInfo> misses_scratch_;
    std::vector<int> batch_reserved_slots_scratch_;
    std::vector<MissInfo> to_fill_scratch_;
    std::vector<int> assigned_hslots_scratch_;
    std::vector<uint64_t> assigned_hslot_gens_scratch_;
    std::unique_ptr<std::atomic<int>[]> parts_left_scratch_;
    size_t parts_left_capacity_ = 0;

    std::unique_ptr<ExpertFillPool> fill_pool_;
    int fill_parts_ = 8;
};

inline void HostExpertLease::release() {
    if (cache_ && hslot_ >= 0) {
        cache_->release_host_lease(hslot_, generation_);
        cache_ = nullptr;
        hslot_ = -1;
    }
}

inline HostExpertLease::~HostExpertLease() { release(); }

inline HostExpertLease::HostExpertLease(HostExpertLease&& o) noexcept
    : cache_(o.cache_), layer_(o.layer_), expert_id_(o.expert_id_), hslot_(o.hslot_),
      generation_(o.generation_), slot_host_(o.slot_host_) {
    o.cache_ = nullptr;
    o.hslot_ = -1;
}

inline HostExpertLease& HostExpertLease::operator=(HostExpertLease&& o) noexcept {
    if (this != &o) {
        release();
        cache_ = o.cache_;
        layer_ = o.layer_;
        expert_id_ = o.expert_id_;
        hslot_ = o.hslot_;
        generation_ = o.generation_;
        slot_host_ = o.slot_host_;
        o.cache_ = nullptr;
        o.hslot_ = -1;
    }
    return *this;
}
