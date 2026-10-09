#pragma once

#include "core/Context.hpp"
#include "models/Qwen4ExpHostArena.hpp"
#include "models/Qwen4ExpHotExperts.hpp"
#include "ops/MoeExpertOps.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

namespace aila {

/// Marker value for a non-resident expert
inline constexpr int32_t kAdaptiveNotResident = -1;

/// Configuration for Qwen4ExpAdaptiveCache
struct AdaptiveCacheConfig {
    int num_layers = 48;
    int num_experts_per_layer = 512;
    int num_gpu_slots = 2800;            // Fixed resident VRAM slots (2,800 experts ≈ 6.8GB VRAM)
    int staging_slots_per_buffer = 16;   // Ping-Pong A/B buffer capacity per buffer (16 cold experts)
    int hidden_size = 2560;
    int moe_intermediate_size = 640;
    int blocksize = 64;
    bool compact_nested = true;          // uint8_t codes for absmax (compact format, slot = 2,534,400 bytes)
    float hysteresis_threshold = 1.5f;   // Δ > 1.5 threshold to prevent cache thrashing
    float candidate_min_usage = 2.0f;    // Minimum usage count before candidate is eligible for swap
    float decay_factor = 0.7f;           // Periodic usage decay factor
};

/// Record of which expert is currently assigned to a resident GPU slot
struct ResidentSlotEntry {
    int32_t layer = -1;
    int32_t expert_id = -1;
};

/// A planned swap of a cold resident victim for a hotter candidate
struct AdaptiveSwapPlan {
    int layer = -1;
    int candidate_expert = -1; // Incoming high-frequency expert
    int victim_expert = -1;    // Outgoing low-frequency expert
    int victim_slot = -1;      // Resident GPU slot to be replaced
    float gain = 0.0f;         // usage(candidate) - usage(victim)
};

/// Result of staging cold experts via ping-pong buffer
struct StagingBatch {
    int ping_pong_index = 0;              // 0 for Buffer A, 1 for Buffer B
    int layer = -1;
    int count = 0;                        // Number of cold experts staged
    std::array<int, 16> expert_ids{};      // Expert IDs staged
    std::array<int, 16> staging_slots{};   // Staging slot indices (0 .. count-1)
    sycl::event ready_event{};            // Hardware DMA completion event
};

/// Breakdown of route resolution for a layer
struct RouteResolution {
    int num_hits = 0;
    int num_misses = 0;
    std::vector<int> hit_expert_ids;
    std::vector<int> hit_slots;
    std::vector<int> miss_expert_ids;
};

/// Telemetry and statistics
struct AdaptiveCacheStats {
    uint64_t route_queries = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t warm_bytes_transferred = 0;
    uint64_t cold_bytes_transferred = 0;
    uint64_t swap_bytes_transferred = 0;
    uint64_t cold_experts_staged = 0;
    uint64_t swaps_executed = 0;

    double hit_rate() const noexcept {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<double>(cache_hits) / static_cast<double>(total) : 0.0;
    }
};

// Fixed GPU slots with an indexed hotset and two bounded transfer banks.
// The arena owns each nonresident expert in native RAM. Replacements preserve
// the outgoing GPU expert before transferring RAM ownership to it. Cross-queue
// copy and consumer events remain live across forward calls and cancellation.
class Qwen4ExpAdaptiveCache {
public:
    Qwen4ExpAdaptiveCache() = default;
    ~Qwen4ExpAdaptiveCache();

    Qwen4ExpAdaptiveCache(const Qwen4ExpAdaptiveCache&) = delete;
    Qwen4ExpAdaptiveCache& operator=(const Qwen4ExpAdaptiveCache&) = delete;

    Qwen4ExpAdaptiveCache(Qwen4ExpAdaptiveCache&& other) noexcept;
    Qwen4ExpAdaptiveCache& operator=(Qwen4ExpAdaptiveCache&& other) noexcept;

    // =========================================================================
    // Allocation & Lifecycle
    // =========================================================================

    /// Allocate resident slots, ping-pong staging buffers, and mapping tables via Context
    bool Allocate(
        Context& ctx,
        const AdaptiveCacheConfig& cfg = AdaptiveCacheConfig{},
        std::string* error_message = nullptr);

    /// Allocate resident slots, ping-pong staging buffers, and mapping tables directly via SYCL queue (unit tests)
    bool Allocate(
        sycl::queue& q,
        const AdaptiveCacheConfig& cfg = AdaptiveCacheConfig{},
        std::string* error_message = nullptr);

    /// Release all allocated GPU device memory and reset tables
    void Release() noexcept;

    bool IsAllocated() const noexcept { return !slot_dev_base_.empty() && num_slots_ > 0; }

    int NumLayers() const noexcept { return config_.num_layers; }
    int NumExpertsPerLayer() const noexcept { return config_.num_experts_per_layer; }
    int NumGpuSlots() const noexcept { return num_slots_; }
    int StagingSlotsPerBuffer() const noexcept { return config_.staging_slots_per_buffer; }
    size_t SlotBytes() const noexcept { return geom_.slot_bytes; }
    const ExpertSlotGeometry& Geometry() const noexcept { return geom_; }

    size_t TotalResidentBytes() const noexcept { return static_cast<size_t>(num_slots_) * geom_.slot_bytes; }
    size_t TotalStagingBytes() const noexcept { return static_cast<size_t>(config_.staging_slots_per_buffer * 2) * geom_.slot_bytes; }
    size_t TotalVramAllocatedBytes() const noexcept { return TotalResidentBytes() + TotalStagingBytes(); }

    // =========================================================================
    // Lookup & Device Access
    // =========================================================================

    /// Returns resident slot ID (0 <= slot < num_slots) if cached, or kAdaptiveNotResident (-1)
    inline int32_t GetSlot(int layer, int expert_id) const noexcept {
        if (layer < 0 || layer >= config_.num_layers ||
            expert_id < 0 || expert_id >= config_.num_experts_per_layer) {
            return kAdaptiveNotResident;
        }
        return slot_table_[static_cast<size_t>(layer) * static_cast<size_t>(config_.num_experts_per_layer) + static_cast<size_t>(expert_id)];
    }

    /// True if (layer, expert_id) is currently resident in VRAM
    inline bool IsResident(int layer, int expert_id) const noexcept {
        return GetSlot(layer, expert_id) >= 0;
    }

    /// Reverse mapping: inspect which expert is resident in slot_id
    inline ResidentSlotEntry GetResidentEntry(int slot_id) const noexcept {
        if (slot_id < 0 || slot_id >= num_slots_) return ResidentSlotEntry{-1, -1};
        return slot_to_expert_[slot_id];
    }

    /// Direct host pointer to flat slot table [num_layers * num_experts_per_layer]
    const int32_t* HostSlotTable() const noexcept { return slot_table_.data(); }

    /// Direct GPU device pointer to flat slot table [num_layers * num_experts_per_layer]
    const int32_t* DeviceSlotTable() const noexcept { return slot_table_dev_; }

    /// Base GPU device pointer for resident slot
    inline uint8_t* GetSlotDevicePtr(int slot_id) const noexcept {
        if (slot_id < 0 || slot_id >= static_cast<int>(slot_dev_base_.size())) return nullptr;
        return slot_dev_base_[slot_id];
    }

    /// Base GPU device pointer for ping-pong staging slot
    inline uint8_t* GetStagingSlotDevicePtr(int ping_pong_index, int staging_slot_idx) const noexcept {
        const int pp = ping_pong_index & 1;
        if (staging_slot_idx < 0 || staging_slot_idx >= config_.staging_slots_per_buffer) return nullptr;
        return staging_dev_[pp] + static_cast<size_t>(staging_slot_idx) * geom_.slot_bytes;
    }

    // =========================================================================
    // Kernel Descriptors (MoeExpertSlotDev)
    // =========================================================================

    /// Build device descriptor for a resident slot
    ops::MoeExpertSlotDev GetResidentSlotDev(
        int slot_id,
        const LayerNestedMetadata* nested = nullptr) const;

    /// Build device descriptor for a staging slot
    ops::MoeExpertSlotDev GetStagingSlotDev(
        int ping_pong_index,
        int staging_slot_idx,
        const LayerNestedMetadata* nested = nullptr) const;

    // Establish a unique initial hotset before the arena allocates its RAM
    // complement. Returned pointers are startup-only upload destinations.
    std::vector<uint8_t*> InitializeResidency(
        const std::vector<std::pair<int, int>>* ranked_profile = nullptr);
    void FinishInitialLoad(sycl::queue& q);
    const sycl::event* ResidentReady() const noexcept { return resident_pending_ ? &resident_ready_ : nullptr; }
    void RecordResidentConsumer(const sycl::event& e) { resident_reader_=e; resident_read_=true; }
    void RecordStagingConsumer(int index, const sycl::event& e) {
        staging_reader_[index & 1]=e; staging_read_[index & 1]=true;
    }

    // =========================================================================
    // Warmup & Device Table Sync
    // =========================================================================

    /**
     * @brief Pre-warm VRAM slots at startup from Host Arena via PCIe DMA.
     * @param arena Populated native RAM arena
     * @param copy_q Dedicated copy queue
     * @param ranked_profile Optional ranked list of {layer, expert_id} pairs; if null, uses default static hot experts
     * @param error_message Optional error message
     * @return true on success
     */
    bool Warmup(
        const Qwen4ExpHostArena& arena,
        sycl::queue& copy_q,
        const std::vector<std::pair<int, int>>* ranked_profile = nullptr,
        std::string* error_message = nullptr);

    /// Asynchronously synchronizes the host slot_table_ to slot_table_dev_ on GPU
    sycl::event SyncSlotTableToDevice(sycl::queue& q);

    // =========================================================================
    // Routing Resolution & Online Usage Tracking
    // =========================================================================

    /// Resolves which experts in route are cache hits and which are misses
    RouteResolution ResolveRoute(int layer, const int* expert_ids, int count) const;

    /// Record routing observation and update hit/miss counters and frequency usage
    void ObserveRouting(int layer, const int* expert_ids, int count, float weight = 1.0f);

    /// Record individual expert access
    void RecordAccess(int layer, int expert_id, float weight = 1.0f);

    /// Decay historical frequency usage by decay_factor (e.g. 0.7f)
    void Decay(float factor = -1.0f);

    /// Query current usage score of an expert
    float GetUsage(int layer, int expert_id) const;

    // =========================================================================
    // Adaptive Eviction & Dynamic Replacement
    // =========================================================================

    /**
     * @brief Generates per-layer swap plans with hysteresis threshold (Δ > 1.5).
     * @param max_swaps_per_layer Maximum swaps allowed per layer (0 = unlimited, default 1)
     * @param max_total_swaps Maximum swaps in total across all layers (0 = unlimited, default 16)
     * @return List of AdaptiveSwapPlan sorted by gain descending
     */
    std::vector<AdaptiveSwapPlan> PlanSwaps(
        int max_swaps_per_layer = 1,
        int max_total_swaps = 16) const;

    /**
     * @brief Preserves sole GPU copies in RAM and installs candidates using bounded staging.
     * The CPU ownership exchange waits for D2H; the returned event publishes the device table.
     * @param arena Populated native RAM arena
     * @param copy_q Dedicated copy queue
     * @param plans Swap plans to execute
     * @return sycl::event representing the completion of DMA transfers and device table updates
     */
    sycl::event ExecuteSwapsAsync(
        Qwen4ExpHostArena& arena,
        sycl::queue& copy_q,
        const std::vector<AdaptiveSwapPlan>& plans);

    // =========================================================================
    // Cold Expert Dual-Buffering Staging (Ping-Pong)
    // =========================================================================

    /**
     * @brief Asynchronously stages cold experts into the specified Ping-Pong buffer (A or B)
     *        through a bounded Host USM bank; device reuse depends on its last reader.
     * @param arena Populated native RAM arena
     * @param copy_q Dedicated copy queue
     * @param layer Layer index
     * @param cold_expert_ids Array of cold expert IDs
     * @param count Number of cold expert IDs
     * @param ping_pong_index 0 for Buffer A, 1 for Buffer B
     * @return StagingBatch containing staged slot indices and DMA completion event
     */
    StagingBatch StageColdExpertsAsync(
        const Qwen4ExpHostArena& arena,
        sycl::queue& copy_q,
        int layer,
        const int* cold_expert_ids,
        int count,
        int ping_pong_index);

    // =========================================================================
    // Telemetry & Statistics
    // =========================================================================

    const AdaptiveCacheStats& Stats() const noexcept { return stats_; }
    void ResetStats() noexcept { stats_ = AdaptiveCacheStats{}; }

private:
    void InitGeometry();

    AdaptiveCacheConfig config_{};
    ExpertSlotGeometry geom_{};
    int num_slots_ = 0;
    size_t total_experts_ = 0;

    Context* ctx_ = nullptr;
    std::unique_ptr<sycl::queue> standalone_queue_;

    // Resident VRAM pool (chunked to stay within GPU max single allocation limits)
    std::vector<uint8_t*> dev_pool_chunks_;
    std::vector<uint8_t*> slot_dev_base_;

    // Ping-Pong Staging Buffers [Buffer A (0), Buffer B (1)]
    uint8_t* staging_dev_[2] = {nullptr, nullptr};
    uint8_t* staging_host_[2] = {nullptr, nullptr};
    std::array<std::array<int, 16>, 2> staging_experts_{};
    sycl::event staging_copy_[2]{}, staging_reader_[2]{};
    bool staging_pending_[2]{}, staging_read_[2]{};
    sycl::event resident_ready_{}, resident_reader_{};
    bool resident_pending_=false, resident_read_=false;

    // Fast mapping tables
    std::vector<int32_t> slot_table_;           // Host table [num_layers * num_experts_per_layer]
    int32_t* slot_table_dev_ = nullptr;         // Device table [num_layers * num_experts_per_layer]
    std::vector<ResidentSlotEntry> slot_to_expert_; // Reverse table [num_slots_]

    // Frequency / Usage tracking
    std::vector<float> usage_;                  // [num_layers * num_experts_per_layer]
    std::vector<uint64_t> access_count_;        // [num_layers * num_experts_per_layer]

    // Telemetry
    mutable AdaptiveCacheStats stats_{};
};

} // namespace aila
