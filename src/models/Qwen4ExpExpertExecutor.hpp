#pragma once

#include "core/Context.hpp"
#include "models/Qwen4ExpExpertCache.hpp"
#include "ops/MoeExpertOps.hpp"

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

class Qwen4ExpExpertExecutor {
public:
    Qwen4ExpExpertExecutor() = default;
    ~Qwen4ExpExpertExecutor() = default;

    void init(Context& ctx, Qwen4ExpExpertCache& cache, int max_seq_len, int top_k = 10);

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

    // Record execution completion
    void record_execution_complete(LayerMoePlan& plan, sycl::event finish_event);

    bool is_initialized() const { return initialized_; }

private:
    Context* ctx_ = nullptr;
    Qwen4ExpExpertCache* cache_ = nullptr;
    int max_seq_len_ = 0;
    int top_k_ = 10;
    bool initialized_ = false;
    LayerMoePlan active_plan_;
};

} // namespace aila
