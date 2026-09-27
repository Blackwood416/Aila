#pragma once

#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "ops/MoeExpertOps.hpp"
#include "ops/Qwen4ExpCpuExpertOps.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace aila {

struct RouteRankEntry {
    int token_idx = 0;
    int rank = 0;
    int expert_id = -1;
    float route_weight = 0.0f;
    bool is_gpu = true;
    int gpu_slot = -1;
    int result_idx = -1;
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
    uint64_t cpu_straggler_count = 0;
    uint64_t cpu_overlapped_count = 0;
    uint64_t cpu_compute_ns = 0;
    uint64_t cpu_exposed_tail_ns = 0;
    uint64_t gpu_tasks_dispatched = 0;
    uint64_t layers_hetero_executed = 0;

    void reset() {
        *this = HeteroExecutionStats{};
    }
};

class Qwen4ExpExpertExecutor {
public:
    Qwen4ExpExpertExecutor() = default;
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

    // Ensure all pending GPU tasks are acquired/staged into GPU slots and record final_dma_event
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
        bool uses_nested = false);

    // Record execution completion
    void record_execution_complete(LayerMoePlan& plan, sycl::event finish_event);

    bool is_initialized() const { return initialized_; }
    const HeteroExecutionStats& stats() const { return stats_; }
    void reset_stats() { stats_.reset(); }

    int max_cpu_credits() const { return max_cpu_credits_; }
    void set_max_cpu_credits(int credits) { max_cpu_credits_ = std::max(0, std::min(4, credits)); }
    int max_cpu_tokens() const { return max_cpu_tokens_; }
    void set_max_cpu_tokens(int tokens) { max_cpu_tokens_ = std::max(1, tokens); }

    // Check if straggler protection penalty is currently active
    bool is_straggler_penalty_active() const { return straggler_penalty_active_; }
    void set_straggler_penalty(bool active) { straggler_penalty_active_ = active; }

private:
    void cleanup();

    Context* ctx_ = nullptr;
    Qwen4ExpExpertCache* cache_ = nullptr;
    int max_seq_len_ = 0;
    int top_k_ = 10;
    int hidden_size_ = 2560;
    int intermediate_size_ = 640;
    bool initialized_ = false;

    // H3: Bounded admission credit & Straggler parameters
    int max_cpu_credits_ = 2; // bounded in [1, 4]
    int max_cpu_tokens_ = 4;   // strictly m <= 4
    bool straggler_penalty_active_ = false;
    int consecutive_stragglers_ = 0;
    HeteroExecutionStats stats_{};

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
};

} // namespace aila
