#pragma once

#include "core/Context.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/Qwen4ExpCpuExpertOps.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace aila {
class Qwen4ExpHostArena;

struct RouteRankEntry {
    int token_idx = 0;
    int rank = 0;
    int expert_id = -1;
    float route_weight = 0.0f;
    bool is_gpu = true;
    int gpu_slot = -1;
    uint64_t gpu_generation = 0;
    int result_idx = -1;
    sycl::event dma_event{};
};

struct HeteroExecutionStats {
    uint64_t cpu_tasks_dispatched = 0;
    uint64_t cpu_tasks_completed = 0;
    uint64_t cpu_mapped_tasks_dispatched = 0, cpu_mapped_tasks_completed = 0;
    uint64_t cpu_cached_tasks_dispatched = 0, cpu_cached_tasks_completed = 0;
    uint64_t cpu_straggler_count = 0;
    uint64_t cpu_overlapped_count = 0;
    uint64_t cpu_compute_ns = 0;
    uint64_t cpu_activation_wait_ns = 0;
    uint64_t cpu_route_wait_ns = 0, cpu_route_async_layers = 0;
    uint64_t cpu_mmap_prefetch_ns = 0, cpu_mmap_prefetch_calls = 0;
    uint64_t cpu_mmap_prefetch_ranges = 0, cpu_mmap_prefetch_bytes = 0, cpu_mmap_prefetch_failures = 0;
    uint64_t cpu_mmap_touch_ns = 0, cpu_mmap_touch_pages = 0;
    uint64_t cpu_scale_cache_bytes = 0, cpu_scale_cache_hits = 0, cpu_scale_cache_misses = 0;
    uint64_t cpu_scale_cache_fills = 0, cpu_scale_cache_evictions = 0, cpu_scale_cache_fallbacks = 0;
    uint64_t cpu_scale_cache_prepare_ns = 0;
    uint64_t cpu_diagnostic_batches = 0, cpu_diagnostic_failures = 0;
    uint64_t cpu_verify_batches = 0;
    uint64_t cpu_a64_verify_batches = 0;
    uint64_t cpu_process_faults = 0, cpu_process_user_ns = 0, cpu_process_kernel_ns = 0;
    uint64_t cpu_gate_up_ns = 0, cpu_quantize_ns = 0, cpu_down_ns = 0;
    uint64_t cpu_gate_chunks = 0, cpu_down_chunks = 0;
    uint64_t cpu_straggler_interval_ns = 0;
    uint64_t gpu_tasks_dispatched = 0;
    uint64_t layers_hetero_executed = 0;
    uint64_t ready_first_layers = 0;
    uint64_t gpu_acquire_ns = 0;
    uint64_t cpu_async_layers = 0;
    uint64_t cpu_join_wait_ns = 0;
    uint64_t cpu_acquire_overlap_ns = 0;
    uint64_t cpu_max_batch_tasks = 0;
    uint64_t cpu_host_prepare_ns = 0;
    uint64_t cpu_host_prepare_batches = 0;
    uint64_t cpu_host_prepared_experts = 0;
    uint64_t cpu_result_copies = 0;
    uint64_t cpu_result_bytes = 0;
    uint64_t cpu_result_submit_ns = 0;
    uint64_t cold_eligible = 0, cold_budget_rejected = 0, cold_queue_ns = 0;

    void reset() {
        *this = HeteroExecutionStats{};
    }
};

class Qwen4ExpExpertExecutor {
public:
    using CpuObserver = void (*)(void*, Context&, int, const RouteRankEntry&,
                                 const ops::MoeExpertSlotHost&, const bf16*, const bf16*, const float*);
    void set_cpu_observer_for_testing(CpuObserver observer, void* state) {
        cpu_observer_ = observer;
        cpu_observer_state_ = state;
    }
    Qwen4ExpExpertExecutor();
    ~Qwen4ExpExpertExecutor();

    Qwen4ExpExpertExecutor(const Qwen4ExpExpertExecutor&) = delete;
    Qwen4ExpExpertExecutor& operator=(const Qwen4ExpExpertExecutor&) = delete;

    void init(Context& ctx, int max_seq_len, int top_k = 10,
              int hidden_size = 2560, int intermediate_size = 640);

    void set_copy_queue(sycl::queue* copy_q) { copy_q_override_ = copy_q; }

    bool is_initialized() const { return initialized_; }
    bool cpu_overlap_enabled() const { return cpu_worker_ != nullptr; }
    bool cpu_work_complete_for_testing() const;
    bool CopyColdStagingHostSlots(const uint8_t* const* sources, uint8_t* destination,
                                  size_t slot_bytes, int count, bool allow_parallel) const;
    const HeteroExecutionStats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }

    int StartArenaCold(const Qwen4ExpHostArena& arena, int layer, const int* ids,
                       const int* ranks, int count, const bf16* x_dev,
                       int row_token = 0, int tokens = 1);
    int RunGraphColdPacket(const Qwen4ExpHostArena& arena, int layer,
                           const int* ids, const int* ranks, int count,
                           const bf16* x_host, int* rank_to_row,
                           float* packet_rows);
    bool CanExecuteArenaColdRows(int rows) const;
    int StartArenaColdRows(const Qwen4ExpHostArena& arena, int layer, int rows,
                           const int* ids, const int* ranks, const int* row_task_begin,
                           const int* row_task_count, int task_count, const bf16* x_dev);
    sycl::event JoinArenaColdRowsRow(float* rank_output, int row,
                                     const sycl::event* order_dep = nullptr);
    void StageResidentInput(const bf16* x_dev, int rows = 1,
                            const sycl::event* producer = nullptr);
    bf16* x_host_ptr() const { return x_host_; }
    void MarkComputeQueueSyncedAfterRoute() noexcept {
        resident_input_event_ = sycl::event{};
        cpu_output_reader_event_ = sycl::event{};
    }
    sycl::event JoinResidentCold(float* rank_output, const sycl::event* order_dep = nullptr);
    const void* ResidentCpuResultPacket() const { return cpu_result_packet_active_ ? cpu_result_packet_dev_ : nullptr; }
    const ops::MoeCpuHostDownRows* ResidentCpuHostRows() const { return cpu_host_result_active_ ? &cpu_host_rows_ : nullptr; }
    bool PrepareDeferredHostReadRows(ops::MoeCpuHostDownRows& out_rows) const noexcept {
        if (!cpu_result_host_read_ || cold_count_ <= 0 || cpu_observer_ != nullptr) return false;
        out_rows.values = cpu_out_host_;
        out_rows.rows = cold_count_;
        out_rows.rank_to_row.fill(-1);
        for (int c = 0; c < cold_count_; ++c) {
            out_rows.rank_to_row[cold_ranks_[c]] = c;
        }
        return true;
    }
    void FinishResidentCpuRead(const sycl::event& merge);
    void AbortResidentCold() noexcept;

    bool CanExecuteArenaVerifyRows(const Qwen4ExpHostArena& arena, int rows, int top_k) const;
    int StartArenaVerifyRows(const Qwen4ExpHostArena& arena, int layer, int rows,
                             const int* ids, const sycl::event& route_event);
    sycl::event JoinArenaVerifyRows(float* rank_output);

    int max_cpu_credits() const { return max_cpu_credits_; }
    void set_max_cpu_credits(int credits) { max_cpu_credits_ = std::clamp(credits, 0, ops::cpu::kMaxExpertBatchTasks); }
    int max_cpu_tokens() const { return max_cpu_tokens_; }
    void set_max_cpu_tokens(int tokens) { max_cpu_tokens_ = std::clamp(tokens, 1, 4); }

    bool is_straggler_penalty_active() const { return straggler_penalty_active_; }
    void set_straggler_penalty(bool active) {
        straggler_penalty_active_ = active;
        cpu_retry_countdown_ = active ? 8 : 0;
    }

private:
    struct CpuWorker;
    std::unique_ptr<CpuWorker> cpu_worker_;
    void cleanup();
    sycl::queue* copy_queue() const;

    Context* ctx_ = nullptr;
    int max_seq_len_ = 0;
    int top_k_ = 10;
    int hidden_size_ = 2560;
    int intermediate_size_ = 640;
    bool initialized_ = false;

    int max_cpu_credits_ = 2;
    int max_cpu_tokens_ = 4;
    bool straggler_penalty_active_ = false;
    int cpu_retry_countdown_ = 0;
    HeteroExecutionStats stats_{};
    CpuObserver cpu_observer_ = nullptr;
    void* cpu_observer_state_ = nullptr;

    bf16* x_host_ = nullptr;
    float* cpu_out_host_ = nullptr;
    void* cpu_result_packet_host_ = nullptr;
    void* cpu_result_packet_dev_ = nullptr;
    bool cpu_result_packet_active_ = false;
    bool cpu_result_host_read_ = false;
    bool cpu_host_result_active_ = false;
    ops::MoeCpuHostDownRows cpu_host_rows_{};
    bf16* scratch_h_ = nullptr;

    sycl::queue* copy_q_override_ = nullptr;
    sycl::event prev_cpu_h2d_ev_{};
    sycl::event cpu_output_reader_event_{};
    sycl::event resident_input_event_{};

    std::array<int, ops::cpu::kMaxExpertBatchTasks> cold_ranks_{}, cold_ids_{};
    std::array<ops::MoeExpertSlotHost, ops::cpu::kMaxExpertBatchTasks> cold_source_slots_{};
    int cold_count_ = 0, cold_layer_ = -1;
    int cold_mapped_count_ = 0;
    int cold_cached_count_ = 0;
    uint64_t cold_submit_ns_ = 0;

    bool arena_verify_pending_ = false;
    bool arena_verify_logged_ = false;
    const Qwen4ExpHostArena* arena_verify_arena_ = nullptr;
    std::vector<ops::MoeExpertSlotHost> arena_verify_sources_;

    bool arena_cold_rows_pending_ = false;
    bool arena_cold_rows_joined_ = false;
    bool arena_cold_rows_logged_ = false;
    int arena_cold_rows_ = 0;
    int arena_cold_rows_next_row_ = 0;
    std::array<int, 4> arena_cold_row_begin_{}, arena_cold_row_count_{};

    uint64_t merge_stat_pairs_ = 0, merge_stat_groups_ = 0, merge_stat_builds_ = 0;
    bool merge_inactive_logged_ = false;
};

} // namespace aila
