#pragma once

#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/Qwen4ExpCpuExpertOps.hpp"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <memory>

namespace aila {

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

struct LayerMoePlan {
    int layer = -1;
    int token_count = 0;
    int top_k = 10;
    int total_tasks = 0;
    int gpu_tasks = 0;
    int cpu_tasks = 0;
    int ready_gpu_tasks = 0;
    int pending_gpu_tasks = 0;

    // Fixed capacity preallocated in executor to ensure zero dynamic allocation on hot path
    std::vector<RouteRankEntry> entries;
    std::vector<int> ready_gpu_indices;
    std::vector<int> pending_gpu_indices;
    std::vector<int> cpu_indices;

    sycl::event ready_subset_event{};
    sycl::event final_dma_event{};

    void clear() {
        layer = -1;
        token_count = 0;
        total_tasks = 0;
        gpu_tasks = 0;
        cpu_tasks = 0;
        ready_gpu_tasks = 0;
        pending_gpu_tasks = 0;
        entries.clear();
        ready_gpu_indices.clear();
        pending_gpu_indices.clear();
        cpu_indices.clear();
        ready_subset_event = sycl::event{};
        final_dma_event = sycl::event{};
    }
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
    // Advisory call duration and requested ranges/bytes, not physical I/O or residency.
    uint64_t cpu_mmap_prefetch_ns = 0, cpu_mmap_prefetch_calls = 0;
    uint64_t cpu_mmap_prefetch_ranges = 0, cpu_mmap_prefetch_bytes = 0, cpu_mmap_prefetch_failures = 0;
    // Single-thread in-span page visits; not unique resident pages or OS faults.
    uint64_t cpu_mmap_touch_ns = 0, cpu_mmap_touch_pages = 0;
    // Opt-in process-wide diagnostic counters sampled around CPU compute.
    // Faults include soft and hard faults from all process threads.
    uint64_t cpu_diagnostic_batches = 0, cpu_diagnostic_failures = 0;
    uint64_t cpu_verify_batches = 0; // diagnostic H8 shadow comparisons, mismatch throws
    uint64_t cpu_process_faults = 0, cpu_process_user_ns = 0, cpu_process_kernel_ns = 0;
    uint64_t cpu_gate_up_ns = 0, cpu_quantize_ns = 0, cpu_down_ns = 0;
    uint64_t cpu_gate_chunks = 0, cpu_down_chunks = 0;
    uint64_t cpu_straggler_interval_ns = 0; // observed host interval, not exposed GPU tail
    uint64_t gpu_tasks_dispatched = 0;
    uint64_t layers_hetero_executed = 0;
    uint64_t ready_first_layers = 0;
    uint64_t gpu_acquire_ns = 0;
    uint64_t cpu_async_layers = 0;
    uint64_t cpu_join_wait_ns = 0;
    uint64_t cpu_acquire_overlap_ns = 0; // intersection of host compute and acquire intervals
    uint64_t cpu_max_batch_tasks = 0;
    uint64_t cpu_host_prepare_ns = 0;
    uint64_t cpu_host_prepare_batches = 0;
    uint64_t cpu_host_prepared_experts = 0;
    uint64_t cpu_result_copies = 0;
    uint64_t cpu_result_bytes = 0;
    uint64_t cpu_result_submit_ns = 0;
    // Submit-to-compute interval minus activation wait; includes source prefetch/touch.
    uint64_t cold_eligible = 0, cold_budget_rejected = 0, cold_queue_ns = 0;

    void reset() {
        *this = HeteroExecutionStats{};
    }
};

struct ExpertPrefetchRequest {
    int layer = -1;
    const int* expert_ids = nullptr;
    int count = 0;
    int quota = 0;
};

class Qwen4ExpExpertExecutor {
public:
    // Synchronous test hook while the host lease or immutable mapped source is
    // still valid. Disabled in normal execution.
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

    void init(Context& ctx, Qwen4ExpExpertCache& cache, int max_seq_len, int top_k = 10,
              int hidden_size = 2560, int intermediate_size = 640);

    // Plan layer experts from route output.
    // Pure CPU control logic; zero allocations on hot path (reuses internal plan storage).
    LayerMoePlan& plan_layer(
        int layer,
        const int* router_host_ids,
        const float* router_host_weights,
        int token_count,
        int top_k,
        bool enable_hetero = false);

    // Acquire the entire GPU subset together, protecting ready entries from eviction.
    void ensure_gpu_pending(LayerMoePlan& plan);

    // Execute heterogeneous decode with ordered rank merge:
    // Computes CPU tasks via HostExpertLease + cpu_expert_forward,
    // computes GPU tasks via moe_decode_gate_up_swiglu_grouped + moe_decode_down_to_rank_f32,
    // and merges all ranks in strict rank order into y_dev via moe_ordered_accumulate_down_rows.
    sycl::event execute_hetero_decode(
        Context& ctx,
        LayerMoePlan& plan,
        const bf16* x_dev,
        bf16* moe_h_dev,
        const float* router_weights_host,
        bf16* y_dev,
        int token_count,
        int top_k,
        const bf16* shared_out = nullptr,
        const bf16* shared_gate = nullptr,
        bool uses_nested = false,
        bool defer_gpu_acquire = false,
        const ExpertPrefetchRequest* next_prefetch = nullptr);

    // Record execution completion
    void record_execution_complete(LayerMoePlan& plan, sycl::event finish_event);

    bool is_initialized() const { return initialized_; }
    bool cpu_overlap_enabled() const { return cpu_worker_ != nullptr; }
    bool cpu_work_complete_for_testing() const;
    const HeteroExecutionStats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }
    // C1 single-row cold service. Start removes only successfully leased CPU
    // entries (or opt-in immutable mapped sources) from the cold subset; caller
    // submits remaining GPU work before Join. ModelWeights must outlive executor.
    int StartResidentCold(GpuReadyLease& route, const bf16* x_dev);
    void StageResidentInput(const bf16* x_dev);
    sycl::event JoinResidentCold(float* rank_output);
    void AbortResidentCold() noexcept;

    int max_cpu_credits() const { return max_cpu_credits_; }
    void set_max_cpu_credits(int credits) { max_cpu_credits_ = std::clamp(credits, 0, ops::cpu::kMaxExpertBatchTasks); }
    int max_cpu_tokens() const { return max_cpu_tokens_; }
    void set_max_cpu_tokens(int tokens) { max_cpu_tokens_ = std::clamp(tokens, 1, 4); }

    // Check if straggler protection penalty is currently active
    bool is_straggler_penalty_active() const { return straggler_penalty_active_; }
    void set_straggler_penalty(bool active) {
        straggler_penalty_active_ = active;
        cpu_retry_countdown_ = active ? 8 : 0;
    }

private:
    struct CpuWorker;
    std::unique_ptr<CpuWorker> cpu_worker_;
    void cleanup();
    uint64_t acquire_begin_ns_ = 0;
    uint64_t acquire_end_ns_ = 0;

    Context* ctx_ = nullptr;
    Qwen4ExpExpertCache* cache_ = nullptr;
    int max_seq_len_ = 0;
    int top_k_ = 10;
    int hidden_size_ = 2560;
    int intermediate_size_ = 640;
    bool initialized_ = false;

    // H3: Bounded admission credit & Straggler parameters
    int max_cpu_credits_ = 2; // maximum expert/token tasks per plan, up to 40
    int cpu_placement_ = 0; // 0: host-ready GPU misses, 1: any host-ready, 2: prepare host on demand
    bool cpu_fixed_credits_ = false;
    bool cpu_batch_prepare_ = false;
    bool cpu_wait_prefetch_ = false;
    bool cpu_coalesce_results_ = false;
    bool cpu_mmap_cold_ = false; // opt-in immutable source for resident host misses
    int max_cpu_tokens_ = 4;   // strictly m <= 4
    bool straggler_penalty_active_ = false;
    int consecutive_stragglers_ = 0;
    int cpu_retry_countdown_ = 0;
    HeteroExecutionStats stats_{};
    CpuObserver cpu_observer_ = nullptr;
    void* cpu_observer_state_ = nullptr;

    LayerMoePlan active_plan_;
    std::vector<HostExpertLease> active_cpu_leases_;

    // Preallocated pinned host and device buffers for zero dynamic allocation on hot path
    bf16* x_host_ = nullptr;         // [max_cpu_tokens * hidden_size] pinned host bf16
    float* cpu_out_host_ = nullptr;  // [max_cpu_credits * hidden_size] pinned host float
    bf16* scratch_h_ = nullptr;      // [max_cpu_tokens * intermediate_size] pinned host bf16

    ops::MoeExpertSlotDev* gpu_slots_host_ = nullptr;
    ops::MoeExpertSlotDev* gpu_slots_dev_ = nullptr;
    int* gpu_ranks_host_ = nullptr;
    int* gpu_ranks_dev_ = nullptr;
    int* gpu_tokens_host_ = nullptr;
    int* gpu_tokens_dev_ = nullptr;
    float* weights_dev_ = nullptr;
    float* down_acc_dev_ = nullptr;
    std::vector<bool> slot_seen_;

    // Zero-allocation preallocated staging for ensure_gpu_pending
    std::vector<int> pending_experts_buf_;
    std::vector<int> assigned_slots_buf_;
    sycl::event prev_cpu_h2d_ev_{};
    sycl::event resident_input_event_{};
    std::array<int, 16> cold_ranks_{}, cold_ids_{};
    std::array<ops::MoeExpertSlotHost, 16> cold_source_slots_{};
    std::array<ops::cpu::CpuExpertTask,16> cold_reference_tasks_{};
    std::array<double, 17> cold_cost_ns_{};
    int cold_count_ = 0, cold_layer_ = -1;
    int cold_mapped_count_ = 0;
    int cold_cached_count_ = 0;
    uint64_t cold_submit_ns_ = 0;
    double cold_budget_ns_ = 1000000.0;
    int cold_policy_ = 0; // 0 C1 budget, 1 all host-Ready cold, 2 recoverable budget, 3 Strata PCIe fraction
    int cold_pcie_num_ = 141; // 0.55 rounded to 256ths, explicitly selected candidate
    uint64_t cold_batches_seen_ = 0;
};

} // namespace aila
