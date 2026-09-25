#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"
#include "utils/SafeTensors.hpp"

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

class ExpertFillPool;

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

    double hit_rate() const {
        uint64_t total = hit_count + miss_count;
        return total > 0 ? static_cast<double>(hit_count) / static_cast<double>(total) : 0.0;
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
    // Returns: SYCL event tracking completion of DMA transfers (or default-constructed host event if all hit).
    sycl::event acquire(int layer,
                        const int* expert_ids,
                        int count,
                        int* out_slots);

    // Starts filling predicted non-resident experts into a pinned prefetch staging area on
    // the worker threads and returns immediately. Binds no slot; the next acquire() joins
    // the fill and DMAs staged experts that actually miss instead of filling them again.
    void prefetch_async(int layer, const int* expert_ids, int count);
    void wait_prefetch();
    void wait_copy();

    // Moves resident experts to the eviction end of the LRU in the given order, so the last
    // one listed is evicted first. Non-resident experts are ignored.
    void demote(int layer, const int* expert_ids, int count);

    // Device slot descriptors for kernel launches
    ops::MoeExpertSlotDev get_slot_dev(int slot_id) const {
        return slots_dev_[slot_id];
    }

    void get_slots_dev(const int* slot_ids, int count, ops::MoeExpertSlotDev* out_slot_devs) const {
        for (int i = 0; i < count; ++i) {
            out_slot_devs[i] = slots_dev_[slot_ids[i]];
        }
    }

    const ExpertCacheStats& stats() const { return stats_; }
    bool contains(int layer, int expert_id) const {
        if (!initialized_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        return layer_expert_to_slot_[expert_flat_index(layer, expert_id)] >= 0;
    }
    bool host_contains(int layer, int expert_id) const {
        if (!initialized_ || !host_pool_enabled_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
        return host_layer_expert_to_slot_[expert_flat_index(layer, expert_id)] >= 0;
    }
    void reset_stats() { stats_ = {}; }
    void log_stats(int layer = -1) const;

    int num_slots() const { return num_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    bool uses_nested_codes() const { return compact_nested_; }
    bool is_initialized() const { return initialized_; }

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
        std::atomic<uint32_t> claimed_parts{0};
        std::atomic<int> parts_remaining{0};
        std::atomic<bool> cancelled{false};
        std::atomic<bool> ready{false};

        PrefetchItem() = default;
        PrefetchItem(int l, int e, int h, int parts)
            : layer(l), expert_id(e), hslot(h), claimed_parts(0), parts_remaining(parts), cancelled(false), ready(false) {}

        PrefetchItem(const PrefetchItem& o)
            : layer(o.layer), expert_id(o.expert_id), hslot(o.hslot),
              claimed_parts(o.claimed_parts.load(std::memory_order_relaxed)),
              parts_remaining(o.parts_remaining.load(std::memory_order_relaxed)),
              cancelled(o.cancelled.load(std::memory_order_relaxed)),
              ready(o.ready.load(std::memory_order_relaxed)) {}

        PrefetchItem& operator=(const PrefetchItem& o) {
            if (this != &o) {
                layer = o.layer;
                expert_id = o.expert_id;
                hslot = o.hslot;
                claimed_parts.store(o.claimed_parts.load(std::memory_order_relaxed), std::memory_order_relaxed);
                parts_remaining.store(o.parts_remaining.load(std::memory_order_relaxed), std::memory_order_relaxed);
                cancelled.store(o.cancelled.load(std::memory_order_relaxed), std::memory_order_relaxed);
                ready.store(o.ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
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

    ExpertCacheStats stats_{};
    bool initialized_ = false;

    // Host fill parallelism (AILA_Q4E_FILL_THREADS / AILA_Q4E_FILL_PARTS)
    // Hot Host RAM Pool (pinned USM host memory allocated in 1 GB chunks)
    bool host_pool_enabled_ = false;
    size_t host_pool_bytes_ = 0;
    int num_host_slots_ = 0;
    std::vector<void*> host_pool_chunks_;
    std::vector<uint8_t*> host_slot_ptrs_;
    std::vector<int> host_layer_expert_to_slot_;
    std::vector<uint64_t> host_slot_to_key_;
    std::list<int> host_lru_list_;
    std::vector<std::list<int>::iterator> host_slot_iters_;

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
    std::unique_ptr<std::atomic<int>[]> parts_left_scratch_;
    size_t parts_left_capacity_ = 0;

    std::unique_ptr<ExpertFillPool> fill_pool_;
    int fill_parts_ = 8;
};
