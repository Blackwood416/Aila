#include "Qwen4ExpExpertCache.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

Qwen4ExpExpertCache::~Qwen4ExpExpertCache() {
    cleanup();
}

Qwen4ExpExpertCache::Qwen4ExpExpertCache(Qwen4ExpExpertCache&& other) noexcept {
    *this = std::move(other);
}

Qwen4ExpExpertCache& Qwen4ExpExpertCache::operator=(Qwen4ExpExpertCache&& other) noexcept {
    if (this == &other) return *this;
    cleanup();

    ctx_ = other.ctx_;
    weights_ = other.weights_;
    num_layers_ = other.num_layers_;
    num_experts_per_layer_ = other.num_experts_per_layer_;
    hidden_size_ = other.hidden_size_;
    intermediate_size_ = other.intermediate_size_;
    blocksize_ = other.blocksize_;

    gu_packed_bytes_ = other.gu_packed_bytes_;
    gu_absmax_count_ = other.gu_absmax_count_;
    gu_absmax_bytes_ = other.gu_absmax_bytes_;
    dn_packed_bytes_ = other.dn_packed_bytes_;
    dn_absmax_count_ = other.dn_absmax_count_;
    dn_absmax_bytes_ = other.dn_absmax_bytes_;

    gu_packed_offset_ = other.gu_packed_offset_;
    gu_absmax_offset_ = other.gu_absmax_offset_;
    dn_packed_offset_ = other.dn_packed_offset_;
    dn_absmax_offset_ = other.dn_absmax_offset_;
    slot_bytes_ = other.slot_bytes_;

    num_slots_ = other.num_slots_;
    dev_pool_bytes_ = other.dev_pool_bytes_;
    dev_pool_ptr_ = other.dev_pool_ptr_;

    staging_buffer_bytes_ = other.staging_buffer_bytes_;
    pinned_staging_ptr_ = other.pinned_staging_ptr_;

    layer_weights_ = std::move(other.layer_weights_);
    lru_list_ = std::move(other.lru_list_);
    slot_iters_ = std::move(other.slot_iters_);
    key_to_slot_ = std::move(other.key_to_slot_);
    slot_to_key_ = std::move(other.slot_to_key_);
    slot_reserved_ = std::move(other.slot_reserved_);
    slots_dev_ = std::move(other.slots_dev_);
    stats_ = other.stats_;
    initialized_ = other.initialized_;

    other.ctx_ = nullptr;
    other.weights_ = nullptr;
    other.dev_pool_ptr_ = nullptr;
    other.pinned_staging_ptr_ = nullptr;
    other.dev_pool_bytes_ = 0;
    other.staging_buffer_bytes_ = 0;
    other.num_slots_ = 0;
    other.initialized_ = false;

    return *this;
}

void Qwen4ExpExpertCache::cleanup() {
    if (ctx_) {
        try {
            ctx_->synchronize();
        } catch (...) {}
        if (dev_pool_ptr_) {
            ctx_->free_device(dev_pool_ptr_);
            dev_pool_ptr_ = nullptr;
        }
        if (pinned_staging_ptr_) {
            ctx_->free_host(pinned_staging_ptr_);
            pinned_staging_ptr_ = nullptr;
        }
    }
    ctx_ = nullptr;
    weights_ = nullptr;
    dev_pool_bytes_ = 0;
    staging_buffer_bytes_ = 0;
    num_slots_ = 0;
    initialized_ = false;
    layer_weights_.clear();
    lru_list_.clear();
    slot_iters_.clear();
    key_to_slot_.clear();
    slot_to_key_.clear();
    slot_reserved_.clear();
    slots_dev_.clear();
}

bool Qwen4ExpExpertCache::init(Context& ctx,
                              ModelWeights& weights,
                              int num_layers,
                              size_t cache_bytes,
                              int num_experts_per_layer,
                              int hidden_size,
                              int intermediate_size,
                              int blocksize,
                              std::string* error_message) {
    cleanup();

    ctx_ = &ctx;
    weights_ = &weights;
    num_layers_ = num_layers;
    num_experts_per_layer_ = num_experts_per_layer;
    hidden_size_ = hidden_size;
    intermediate_size_ = intermediate_size;
    blocksize_ = blocksize;

    // Expert geometry:
    // gate_up: 2 * intermediate_size rows, in_features cols
    const int gu_rows = 2 * intermediate_size_;
    gu_packed_bytes_ = static_cast<size_t>(gu_rows) * (hidden_size_ / 2);
    gu_absmax_count_ = static_cast<size_t>(gu_rows) * (hidden_size_ / blocksize_);
    gu_absmax_bytes_ = gu_absmax_count_ * sizeof(float);

    // down: hidden_size rows, intermediate_size cols
    const int dn_rows = hidden_size_;
    dn_packed_bytes_ = static_cast<size_t>(dn_rows) * (intermediate_size_ / 2);
    dn_absmax_count_ = static_cast<size_t>(dn_rows) * (intermediate_size_ / blocksize_);
    dn_absmax_bytes_ = dn_absmax_count_ * sizeof(float);

    gu_packed_offset_ = 0;
    gu_absmax_offset_ = gu_packed_bytes_;
    dn_packed_offset_ = gu_absmax_offset_ + gu_absmax_bytes_;
    dn_absmax_offset_ = dn_packed_offset_ + dn_packed_bytes_;
    slot_bytes_ = dn_absmax_offset_ + dn_absmax_bytes_;

    if (slot_bytes_ == 0) {
        if (error_message) *error_message = "Qwen4ExpExpertCache: invalid slot geometry";
        return false;
    }

    if (cache_bytes == 0) {
        int mb = aila::env::read_int("AILA_Q4E_EXPERT_CACHE_MB", 4000);
        if (mb <= 0) mb = 4000;
        cache_bytes = static_cast<size_t>(mb) * 1024 * 1024;
    }

    num_slots_ = std::max(1, static_cast<int>(cache_bytes / slot_bytes_));
    dev_pool_bytes_ = static_cast<size_t>(num_slots_) * slot_bytes_;

    try {
        dev_pool_ptr_ = static_cast<uint8_t*>(ctx_->alloc_device(dev_pool_bytes_));
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Qwen4ExpExpertCache device alloc failed: ") + e.what();
        cleanup();
        return false;
    }

    // Pinned host staging buffer for DDR3 -> PCIe 3.0 DMA
    staging_buffer_bytes_ = std::max<size_t>(slot_bytes_ * 16, static_cast<size_t>(64) * 1024 * 1024);
    if (cache_bytes < staging_buffer_bytes_) {
        staging_buffer_bytes_ = std::max<size_t>(slot_bytes_ * 2, cache_bytes);
    }

    try {
        pinned_staging_ptr_ = static_cast<uint8_t*>(ctx_->alloc_host(staging_buffer_bytes_));
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Qwen4ExpExpertCache pinned host alloc failed: ") + e.what();
        cleanup();
        return false;
    }

    // Setup slot tracking & descriptors
    lru_list_.clear();
    slot_iters_.resize(num_slots_);
    key_to_slot_.clear();
    slot_to_key_.assign(num_slots_, UINT64_MAX);
    slot_reserved_.assign(num_slots_, false);
    slots_dev_.resize(num_slots_);

    for (int s = 0; s < num_slots_; ++s) {
        lru_list_.push_back(s);
        auto it = lru_list_.end();
        --it;
        slot_iters_[s] = it;

        uint8_t* base = dev_pool_ptr_ + static_cast<size_t>(s) * slot_bytes_;
        slots_dev_[s].gate_up_packed = base + gu_packed_offset_;
        slots_dev_[s].gate_up_absmax = reinterpret_cast<const float*>(base + gu_absmax_offset_);
        slots_dev_[s].down_packed = base + dn_packed_offset_;
        slots_dev_[s].down_absmax = reinterpret_cast<const float*>(base + dn_absmax_offset_);
    }

    // Load expert references from weights
    layer_weights_.resize(num_layers_);
    for (int L = 0; L < num_layers_; ++L) {
        std::string gu_name = "model.language_model.layers." + std::to_string(L) + ".mlp.experts.gate_up_proj";
        std::string dn_name = "model.language_model.layers." + std::to_string(L) + ".mlp.experts.down_proj";

        if (weights.has_host(gu_name) && weights.has_host(dn_name)) {
            if (LoadBnb4BitHostRef(weights, gu_name, layer_weights_[L].gate_up, error_message, hidden_size_) &&
                LoadBnb4BitHostRef(weights, dn_name, layer_weights_[L].down, error_message, intermediate_size_)) {
                layer_weights_[L].valid = true;
            }
        }
    }

    stats_ = {};
    initialized_ = true;
    return true;
}

void Qwen4ExpExpertCache::set_layer_expert_ref(int layer,
                                              const Bnb4BitHostWeightRef& gate_up,
                                              const Bnb4BitHostWeightRef& down) {
    if (layer >= static_cast<int>(layer_weights_.size())) {
        layer_weights_.resize(layer + 1);
    }
    layer_weights_[layer].gate_up = gate_up;
    layer_weights_[layer].down = down;
    layer_weights_[layer].valid = true;
}

void Qwen4ExpExpertCache::acquire(int layer,
                                 const int* expert_ids,
                                 int count,
                                 int* out_slots) {
    if (!initialized_ || count <= 0 || expert_ids == nullptr || out_slots == nullptr) {
        return;
    }

    struct MissInfo {
        int batch_index;
        int layer;
        int expert_id;
        int slot_id;
    };

    std::vector<MissInfo> misses;
    misses.reserve(count);

    std::vector<int> batch_reserved_slots;
    batch_reserved_slots.reserve(count);

    // Phase 1: check hits and deduplicate within this batch
    for (int i = 0; i < count; ++i) {
        int expert = expert_ids[i];
        uint64_t key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            int slot = it->second;
            out_slots[i] = slot;
            if (!slot_reserved_[slot]) {
                slot_reserved_[slot] = true;
                batch_reserved_slots.push_back(slot);
                lru_list_.erase(slot_iters_[slot]);
                lru_list_.push_front(slot);
                slot_iters_[slot] = lru_list_.begin();
            }
            ++stats_.hit_count;
        } else {
            out_slots[i] = -1;
        }
    }

    // Phase 2: assign slots for misses, protecting all batch_reserved_slots from eviction
    for (int i = 0; i < count; ++i) {
        if (out_slots[i] >= 0) continue;
        int expert = expert_ids[i];
        uint64_t key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);

        // Check if another miss earlier in this same batch already bound this key
        auto it = key_to_slot_.find(key);
        if (it != key_to_slot_.end()) {
            out_slots[i] = it->second;
            continue;
        }

        // Search for least recently used unreserved slot from back
        int victim_slot = -1;
        for (auto r_it = lru_list_.rbegin(); r_it != lru_list_.rend(); ++r_it) {
            int s = *r_it;
            if (!slot_reserved_[s]) {
                victim_slot = s;
                break;
            }
        }

        if (victim_slot < 0) {
            for (int s : batch_reserved_slots) slot_reserved_[s] = false;
            throw std::runtime_error("Qwen4ExpExpertCache: all slots reserved in batch, count exceeds num_slots");
        }

        // Evict old key if bound
        uint64_t old_key = slot_to_key_[victim_slot];
        if (old_key != UINT64_MAX) {
            key_to_slot_.erase(old_key);
        }
        key_to_slot_[key] = victim_slot;
        slot_to_key_[victim_slot] = key;

        // Move victim to MRU (front) and reserve for this batch
        lru_list_.erase(slot_iters_[victim_slot]);
        lru_list_.push_front(victim_slot);
        slot_iters_[victim_slot] = lru_list_.begin();

        slot_reserved_[victim_slot] = true;
        batch_reserved_slots.push_back(victim_slot);

        out_slots[i] = victim_slot;
        misses.push_back({i, layer, expert, victim_slot});
        ++stats_.miss_count;
    }

    // Phase 3: populate pinned staging buffer & initiate DMA copies
    if (!misses.empty()) {
        const int staging_capacity = std::max(1, static_cast<int>(staging_buffer_bytes_ / slot_bytes_));
        for (size_t m = 0; m < misses.size(); m += staging_capacity) {
            int chunk_size = std::min<int>(staging_capacity, static_cast<int>(misses.size() - m));
            for (int c = 0; c < chunk_size; ++c) {
                const auto& miss = misses[m + c];
                uint8_t* staging_slot = pinned_staging_ptr_ + static_cast<size_t>(c) * slot_bytes_;
                load_expert_to_staging(miss.layer, miss.expert_id, staging_slot);

                uint8_t* dev_slot = dev_pool_ptr_ + static_cast<size_t>(miss.slot_id) * slot_bytes_;
                ctx_->memcpy_h2d_async(dev_slot, staging_slot, slot_bytes_);
                stats_.total_transferred_bytes += slot_bytes_;
            }
            ctx_->synchronize();
        }
    }

    // Phase 4: release batch reservations
    for (int s : batch_reserved_slots) {
        slot_reserved_[s] = false;
    }
}

void Qwen4ExpExpertCache::load_expert_to_staging(int layer, int expert_id, uint8_t* staging_slot) {
    if (layer < 0 || layer >= static_cast<int>(layer_weights_.size()) || !layer_weights_[layer].valid) {
        std::memset(staging_slot, 0, slot_bytes_);
        return;
    }

    const auto& gu_ref = layer_weights_[layer].gate_up;
    const auto& dn_ref = layer_weights_[layer].down;

    // 1. Packed gate_up
    if (gu_ref.packed_weight.valid() && gu_ref.packed_weight.ptr) {
        const uint8_t* src = gu_ref.packed_weight.ptr + static_cast<size_t>(expert_id) * gu_packed_bytes_;
        std::memcpy(staging_slot + gu_packed_offset_, src, gu_packed_bytes_);
    } else {
        std::memset(staging_slot + gu_packed_offset_, 0, gu_packed_bytes_);
    }

    // 2. Absmax gate_up
    float* gu_absmax_dst = reinterpret_cast<float*>(staging_slot + gu_absmax_offset_);
    if (gu_ref.absmax.valid() && gu_ref.absmax.ptr) {
        int64_t expert_absmax_offset = static_cast<int64_t>(expert_id) * gu_absmax_count_;
        if (gu_ref.quant_state.nested) {
            const uint8_t* qabsmax = gu_ref.absmax_codes_ptr() + expert_absmax_offset;
            const float* nested_absmax = gu_ref.nested_absmax_ptr();
            const float* nested_quant_map = gu_ref.nested_quant_map_ptr();
            const int nested_blocksize = gu_ref.quant_state.nested_blocksize > 0 ? gu_ref.quant_state.nested_blocksize : 256;
            const float nested_offset = gu_ref.quant_state.nested_offset;

            for (size_t i = 0; i < gu_absmax_count_; ++i) {
                int64_t global_idx = expert_absmax_offset + static_cast<int64_t>(i);
                int64_t n_blk = global_idx / nested_blocksize;
                uint8_t code = qabsmax[i];
                gu_absmax_dst[i] = nested_quant_map[code] * nested_absmax[n_blk] + nested_offset;
            }
        } else {
            const float* absmax_f32 = gu_ref.absmax_f32_ptr() + expert_absmax_offset;
            std::memcpy(gu_absmax_dst, absmax_f32, gu_absmax_count_ * sizeof(float));
        }
    } else {
        std::memset(gu_absmax_dst, 0, gu_absmax_bytes_);
    }

    // 3. Packed down
    if (dn_ref.packed_weight.valid() && dn_ref.packed_weight.ptr) {
        const uint8_t* src = dn_ref.packed_weight.ptr + static_cast<size_t>(expert_id) * dn_packed_bytes_;
        std::memcpy(staging_slot + dn_packed_offset_, src, dn_packed_bytes_);
    } else {
        std::memset(staging_slot + dn_packed_offset_, 0, dn_packed_bytes_);
    }

    // 4. Absmax down
    float* dn_absmax_dst = reinterpret_cast<float*>(staging_slot + dn_absmax_offset_);
    if (dn_ref.absmax.valid() && dn_ref.absmax.ptr) {
        int64_t expert_absmax_offset = static_cast<int64_t>(expert_id) * dn_absmax_count_;
        if (dn_ref.quant_state.nested) {
            const uint8_t* qabsmax = dn_ref.absmax_codes_ptr() + expert_absmax_offset;
            const float* nested_absmax = dn_ref.nested_absmax_ptr();
            const float* nested_quant_map = dn_ref.nested_quant_map_ptr();
            const int nested_blocksize = dn_ref.quant_state.nested_blocksize > 0 ? dn_ref.quant_state.nested_blocksize : 256;
            const float nested_offset = dn_ref.quant_state.nested_offset;

            for (size_t i = 0; i < dn_absmax_count_; ++i) {
                int64_t global_idx = expert_absmax_offset + static_cast<int64_t>(i);
                int64_t n_blk = global_idx / nested_blocksize;
                uint8_t code = qabsmax[i];
                dn_absmax_dst[i] = nested_quant_map[code] * nested_absmax[n_blk] + nested_offset;
            }
        } else {
            const float* absmax_f32 = dn_ref.absmax_f32_ptr() + expert_absmax_offset;
            std::memcpy(dn_absmax_dst, absmax_f32, dn_absmax_count_ * sizeof(float));
        }
    } else {
        std::memset(dn_absmax_dst, 0, dn_absmax_bytes_);
    }
}

void Qwen4ExpExpertCache::log_stats(int layer) const {
    uint64_t total = stats_.hit_count + stats_.miss_count;
    double rate = total > 0 ? 100.0 * static_cast<double>(stats_.hit_count) / static_cast<double>(total) : 0.0;
    if (layer >= 0) {
        std::cout << "[Q4E MoE] layer=" << layer
                  << " hit=" << stats_.hit_count << "/" << total
                  << " (" << rate << "%)"
                  << " bytes=" << stats_.total_transferred_bytes << "\n";
    } else {
        std::cout << "[Q4E MoE] total"
                  << " hit=" << stats_.hit_count << "/" << total
                  << " (" << rate << "%)"
                  << " bytes=" << stats_.total_transferred_bytes << "\n";
    }
}
