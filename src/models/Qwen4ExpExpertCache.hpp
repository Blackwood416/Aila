#pragma once

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/MoeExpertOps.hpp"
#include "utils/SafeTensors.hpp"

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

struct ExpertCacheStats {
    uint64_t hit_count = 0;
    uint64_t miss_count = 0;
    uint64_t total_transferred_bytes = 0;

    double hit_rate() const {
        uint64_t total = hit_count + miss_count;
        return total > 0 ? static_cast<double>(hit_count) / static_cast<double>(total) : 0.0;
    }
};

class Qwen4ExpExpertCache {
public:
    Qwen4ExpExpertCache() = default;
    ~Qwen4ExpExpertCache();

    Qwen4ExpExpertCache(const Qwen4ExpExpertCache&) = delete;
    Qwen4ExpExpertCache& operator=(const Qwen4ExpExpertCache&) = delete;

    Qwen4ExpExpertCache(Qwen4ExpExpertCache&& other) noexcept;
    Qwen4ExpExpertCache& operator=(Qwen4ExpExpertCache&& other) noexcept;

    // Initialize expert cache
    // num_layers: number of MoE layers (e.g. 48)
    // cache_bytes: GPU memory budget for expert cache. If 0, reads AILA_Q4E_EXPERT_CACHE_MB (default 6656 MB).
    bool init(Context& ctx,
              ModelWeights& weights,
              int num_layers,
              size_t cache_bytes = 0,
              int num_experts_per_layer = 512,
              int hidden_size = 2560,
              int intermediate_size = 640,
              int blocksize = 64,
              std::string* error_message = nullptr);

    // Override or set layer expert references (useful for testing or manual binding)
    void set_layer_expert_ref(int layer,
                              const Bnb4BitHostWeightRef& gate_up,
                              const Bnb4BitHostWeightRef& down);

    // Acquire slots for layer and expert_ids.
    // count: number of active experts (e.g. top_k = 10)
    // out_slots: filled with slot indices in [0, num_slots - 1]
    // Crucial guarantee: slots acquired in the same batch will never evict each other.
    void acquire(int layer,
                 const int* expert_ids,
                 int count,
                 int* out_slots);

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
    void reset_stats() { stats_ = {}; }
    void log_stats(int layer = -1) const;

    int num_slots() const { return num_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }
    bool is_initialized() const { return initialized_; }

    size_t gu_packed_bytes() const { return gu_packed_bytes_; }
    size_t gu_absmax_count() const { return gu_absmax_count_; }
    size_t gu_absmax_bytes() const { return gu_absmax_bytes_; }
    size_t dn_packed_bytes() const { return dn_packed_bytes_; }
    size_t dn_absmax_count() const { return dn_absmax_count_; }
    size_t dn_absmax_bytes() const { return dn_absmax_bytes_; }

    const uint8_t* dev_pool_ptr() const { return dev_pool_ptr_; }
    const uint8_t* pinned_staging_ptr() const { return pinned_staging_ptr_; }

private:
    void cleanup();
    void load_expert_to_staging(int layer, int expert_id, uint8_t* staging_slot);

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

    int num_slots_ = 0;
    size_t dev_pool_bytes_ = 0;
    uint8_t* dev_pool_ptr_ = nullptr;

    size_t staging_buffer_bytes_ = 0;
    uint8_t* pinned_staging_ptr_ = nullptr;

    struct LayerWeights {
        Bnb4BitHostWeightRef gate_up{};
        Bnb4BitHostWeightRef down{};
        bool valid = false;
    };
    std::vector<LayerWeights> layer_weights_;

    std::list<int> lru_list_;
    std::vector<std::list<int>::iterator> slot_iters_;
    std::unordered_map<uint64_t, int> key_to_slot_;
    std::vector<uint64_t> slot_to_key_;
    std::vector<bool> slot_reserved_;
    std::vector<ops::MoeExpertSlotDev> slots_dev_;

    ExpertCacheStats stats_{};
    bool initialized_ = false;
};
