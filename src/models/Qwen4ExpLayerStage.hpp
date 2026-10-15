#pragma once

// Qwen4Exp layer staging (role E7, 2026-10-15).
//
// V3 speculative whole-layer staging: while layer L's attention kernels run
// on the compute queue, this module streams the layer's nonresident experts
// (weights only - expert weights are routing independent) into one contiguous
// device arena through a dedicated copy queue. When the router returns,
// experts the arena holds are served from the arena descriptors and the
// remainder falls back to the production staging banks. This is pure data
// movement: slot payloads are copied verbatim, so consumers see bit-identical
// bytes to the bank path.
//
// Wired into the prefill layer loop behind AILA_Q4E_LAYER_STAGE (default 0;
// see docs/plans/qwen38-r1-layer-stage-2026-10-15.md sections 2 and 5). The
// host fill runs on a small persistent pool (config.fill_threads, env
// AILA_Q4E_LAYER_STAGE_THREADS) because the single-threaded memcpy was the
// measured stage bottleneck (178 ms of a 184 ms pass, 2026-10-15). The pool
// copies disjoint slots, so the staged bytes stay bit-identical.

#include "core/Context.hpp"
#include "models/Qwen4ExpAdaptiveCache.hpp"
#include "models/Qwen4ExpExpertReader.hpp"
#include "models/Qwen4ExpHostArena.hpp"
#include "ops/MoeExpertOps.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sycl/sycl.hpp>

namespace aila {

struct LayerStageConfig {
    int num_experts_per_layer = 512;
    int max_slots = 512;      // arena capacity in expert slots
    int chunk_slots = 16;     // slots per host fill chunk / H2D batch
    int fill_banks = 2;       // host fill banks (2..8)
    int fill_threads = 0;     // 0 = env auto (AILA_Q4E_LAYER_STAGE_THREADS, 6);
                              // 1 = inline serial fill (the pre-pool behavior)
};

struct LayerStageStats {
    uint64_t epochs = 0;
    uint64_t staged_experts = 0;
    uint64_t staged_bytes = 0;
    uint64_t skipped_no_source = 0;
    // Per-epoch host intervals of the most recent job (ms).
    double last_fill_ms = 0;
    double last_bank_wait_ms = 0;
    double last_submit_ms = 0;   // Begin -> last H2D submitted
    double last_ready_ms = 0;    // Begin -> last H2D completed
};

// One contiguous device arena (max_slots * slot_bytes) plus small pinned host
// fill banks. One epoch is in flight at a time; the background fill thread
// owns every submission on the caller-supplied copy queue until the epoch's
// last DMA completes.
class Qwen4ExpLayerStage {
public:
    struct RouteDelta {
        int staged_hits = 0;    // routed entries found in the arena (occurrences)
        std::vector<int> missed; // unique routed experts the arena does not hold
    };

    Qwen4ExpLayerStage() = default;
    ~Qwen4ExpLayerStage();
    Qwen4ExpLayerStage(const Qwen4ExpLayerStage&) = delete;
    Qwen4ExpLayerStage& operator=(const Qwen4ExpLayerStage&) = delete;

    // The arena reuses the host arena's ExpertSlotGeometry (compact nested
    // layout) so GetSlotDev descriptors are the same shape as cache slots.
    bool Allocate(Context& ctx, const ExpertSlotGeometry& geom,
                  const LayerStageConfig& cfg = {}, std::string* error = nullptr);
    void Release() noexcept;
    bool IsAllocated() const noexcept { return arena_dev_ != nullptr; }

    int MaxSlots() const noexcept { return config_.max_slots; }
    int ChunkSlots() const noexcept { return config_.chunk_slots; }
    int FillBanks() const noexcept { return config_.fill_banks; }
    int FillThreads() const noexcept { return config_.fill_threads; }
    size_t ArenaBytes() const noexcept { return arena_bytes_; }
    size_t HostBytes() const noexcept { return host_bytes_; }

    // ---------------------------------------------------------------- epochs
    // Speculative staging of `expert_ids` (same order as `sources`) into the
    // arena. `copy_q` must be in order; the module is its only writer for the
    // duration of the epoch. `arena_reader` (optional) gates the first H2D
    // behind the previous epoch's last arena consumer so the arena can be
    // overwritten safely while that consumer still runs.
    // Returns an epoch id; the id stays valid until the next Begin.
    uint64_t Begin(int layer, const std::vector<int>& expert_ids,
                   const std::vector<const uint8_t*>& sources,
                   sycl::queue& copy_q, sycl::event arena_reader = {});
    uint64_t BeginFromArena(int layer, const std::vector<int>& expert_ids,
                            const Qwen4ExpHostArena& arena, sycl::queue& copy_q,
                            sycl::event arena_reader = {});

    // Convenience: every expert of `layer` that the resident table does not
    // hold (sorted ascending). Snapshot only; swaps after Begin are benign
    // because the arena map is keyed by expert id.
    static std::vector<int> ColdExperts(const Qwen4ExpAdaptiveCache& cache, int layer);

    // True once the epoch's last H2D completed (or the job failed).
    bool IsReady(uint64_t epoch) const noexcept;
    // Blocks until the last H2D completed; rethrows a worker failure.
    void WaitReady(uint64_t epoch);
    // Last H2D event of the epoch (empty when none/failed). Kernels may gate
    // on it instead of waiting the host.
    sycl::event ReadyEvent(uint64_t epoch) const;
    // Per-chunk H2D event; valid after WaitReady returned (chunk < ChunkCount).
    sycl::event ChunkEvent(uint64_t epoch, int chunk) const;
    int ChunkCount(uint64_t epoch) const;

    // The last kernel that reads the arena for this epoch; the next Begin
    // without an explicit gate uses it.
    void RecordArenaConsumer(uint64_t epoch, const sycl::event& consumer);

    // ------------------------------------------------------------- consumers
    // Routed experts split into arena hits and unique missed ids (bank
    // fallback). Reads the epoch map; call after WaitReady or, when gating
    // kernels on ReadyEvent, after the map was built (i.e. any time after
    // Begin on the calling thread).
    RouteDelta MissedExperts(uint64_t epoch, const int* route_ids, int count) const;
    int SlotOf(uint64_t epoch, int expert_id) const;
    ops::MoeExpertSlotDev GetSlotDev(uint64_t epoch, int expert_id,
                                     const LayerNestedMetadata* nested) const;

    // ------------------------------------------------------------- lifecycle
    bool Busy() const noexcept { return job_active_.load(); }
    void WaitIdle();
    // Drop epoch bookkeeping. Requires no consumer still reading the arena.
    void Reset();
    LayerStageStats Stats() const;

private:
    void WorkerLoop();
    void RunJob(uint64_t epoch, int layer, bool has_reader, const sycl::event& reader,
                sycl::queue& q);
    void FailJob(uint64_t epoch, const std::string& message);
    ops::MoeExpertSlotDev MakeSlot(const uint8_t* base, int expert,
                                   const LayerNestedMetadata* nested) const;
    // Parallel host fill pool (fill_threads > 1). Helpers block on cv_fill_;
    // the staging worker publishes one chunk at a time under mu_fill_ and
    // copies share 0 itself, then waits cv_fill_done_ until every share is
    // memcpy'd. Shares are disjoint slot ranges, so the result is the same
    // bytes as the serial loop regardless of the thread count.
    void FillLoop(int index);
    static void FillChunkShare(int index, int threads, uint8_t* bank,
                               const uint8_t* const* sources, int count,
                               size_t slot_bytes);

    LayerStageConfig config_{};
    Context* ctx_ = nullptr;
    ExpertSlotGeometry geom_{};
    size_t arena_bytes_ = 0;
    size_t host_bytes_ = 0;
    uint8_t* arena_dev_ = nullptr;
    uint8_t* host_banks_ = nullptr;

    // Worker thread and job handoff.
    std::thread worker_;
    mutable std::mutex mu_;
    std::condition_variable cv_job_;
    std::condition_variable cv_done_;
    bool quit_ = false;
    bool job_present_ = false;
    std::vector<int> in_ids_;                  // caller thread -> worker
    std::vector<const uint8_t*> in_sources_;
    std::vector<int> work_ids_;                // worker owned
    std::vector<const uint8_t*> work_sources_;
    int in_layer_ = -1;
    int work_layer_ = -1;
    bool in_has_reader_ = false;
    sycl::event in_reader_{};
    sycl::queue* copy_q_ = nullptr;
    uint64_t in_epoch_ = 0;

    std::atomic<bool> job_active_{false};
    std::atomic<uint64_t> done_epoch_{0};
    bool failed_ = false;
    std::string error_;
    LayerStageStats stats_{};

    // Epoch map (valid for the current epoch only).
    uint64_t epoch_serial_ = 0;
    uint64_t current_epoch_ = 0;
    int current_layer_ = -1;
    std::vector<int32_t> slot_of_;             // [num_experts_per_layer]
    std::vector<int> epoch_ids_;
    std::vector<sycl::event> chunk_events_;    // worker writes, readers after wait
    sycl::event last_event_{};
    bool last_event_valid_ = false;
    std::vector<sycl::event> bank_events_;     // worker private
    std::vector<bool> bank_event_valid_;
    bool arena_read_ = false;                  // consumer recorded for current
    sycl::event arena_reader_{};

    // Parallel fill pool state (idle when fill_workers_ is empty).
    std::vector<std::thread> fill_workers_;
    mutable std::mutex mu_fill_;
    std::condition_variable cv_fill_;
    std::condition_variable cv_fill_done_;
    std::atomic<uint64_t> fill_gen_{0};
    std::atomic<int> fill_remaining_{0};
    std::atomic<bool> fill_quit_{false};
    uint8_t* fill_bank_ = nullptr;
    const uint8_t* const* fill_sources_ = nullptr;
    int fill_count_ = 0;
};

} // namespace aila
