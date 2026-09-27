#include "models/Qwen4ExpExpertExecutor.hpp"
#include <algorithm>
#include <stdexcept>

namespace aila {

void Qwen4ExpExpertExecutor::init(Context& ctx, Qwen4ExpExpertCache& cache, int max_seq_len, int top_k) {
    ctx_ = &ctx;
    cache_ = &cache;
    max_seq_len_ = std::max(1, max_seq_len);
    top_k_ = std::max(1, top_k);

    const size_t max_entries = static_cast<size_t>(max_seq_len_) * top_k_;
    active_plan_.entries.reserve(max_entries);
    active_plan_.ready_gpu_indices.reserve(max_entries);
    active_plan_.pending_gpu_indices.reserve(max_entries);
    active_plan_.cpu_indices.reserve(max_entries);
    active_plan_.top_k = top_k_;
    initialized_ = true;
}

LayerMoePlan& Qwen4ExpExpertExecutor::plan_layer(
    int layer,
    const int* router_host_ids,
    const float* router_host_weights,
    int token_count,
    int top_k,
    bool enable_hetero) {
    if (!initialized_ || !cache_) {
        throw std::logic_error("Qwen4ExpExpertExecutor: not initialized");
    }

    active_plan_.clear();
    active_plan_.layer = layer;
    active_plan_.token_count = token_count;
    active_plan_.top_k = top_k;
    if (token_count <= 0 || !router_host_ids) {
        return active_plan_;
    }

    const int total = token_count * top_k;
    active_plan_.total_tasks = total;

    if (total > static_cast<int>(active_plan_.entries.capacity())) {
        active_plan_.entries.reserve(total);
        active_plan_.ready_gpu_indices.reserve(total);
        active_plan_.pending_gpu_indices.reserve(total);
        active_plan_.cpu_indices.reserve(total);
    }

    for (int t = 0; t < token_count; ++t) {
        for (int r = 0; r < top_k; ++r) {
            const int idx = t * top_k + r;
            const int expert_id = router_host_ids[idx];
            const float weight = router_host_weights ? router_host_weights[idx] : 1.0f;

            RouteRankEntry entry;
            entry.token_idx = t;
            entry.rank = r;
            entry.expert_id = expert_id;
            entry.route_weight = weight;
            entry.result_idx = idx;

            const ExpertResidency res = cache_->query_residency(layer, expert_id);
            if (res.gpu_ready) {
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                active_plan_.entries.push_back(entry);
                active_plan_.ready_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.ready_gpu_tasks;
            } else if (enable_hetero && res.host_ready && !res.gpu_ready) {
                // Host-ready candidate for CPU execution
                entry.is_gpu = false;
                entry.gpu_slot = -1;
                active_plan_.entries.push_back(entry);
                active_plan_.cpu_indices.push_back(idx);
                ++active_plan_.cpu_tasks;
            } else {
                // GPU pending (copying or needs DMA)
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                active_plan_.entries.push_back(entry);
                active_plan_.pending_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.pending_gpu_tasks;
            }
        }
    }

    return active_plan_;
}

void Qwen4ExpExpertExecutor::ensure_gpu_pending(LayerMoePlan& plan) {
    if (!cache_ || plan.pending_gpu_indices.empty()) {
        return;
    }

    // Collect unique pending expert IDs
    std::vector<int> pending_experts;
    pending_experts.reserve(plan.pending_gpu_indices.size());
    for (int idx : plan.pending_gpu_indices) {
        int eid = plan.entries[idx].expert_id;
        if (std::find(pending_experts.begin(), pending_experts.end(), eid) == pending_experts.end()) {
            pending_experts.push_back(eid);
        }
    }

    std::vector<int> assigned_slots(pending_experts.size(), -1);
    sycl::event dma_ev = cache_->acquire(plan.layer, pending_experts.data(),
                                         static_cast<int>(pending_experts.size()),
                                         assigned_slots.data());
    plan.final_dma_event = dma_ev;

    // Update entries with assigned slots
    for (int idx : plan.pending_gpu_indices) {
        int eid = plan.entries[idx].expert_id;
        for (size_t i = 0; i < pending_experts.size(); ++i) {
            if (pending_experts[i] == eid) {
                plan.entries[idx].gpu_slot = assigned_slots[i];
                break;
            }
        }
    }
}

void Qwen4ExpExpertExecutor::record_execution_complete(LayerMoePlan& plan, sycl::event finish_event) {
    if (!cache_) return;
    const int max_slot = cache_->num_slots();
    std::vector<bool> seen(max_slot, false);

    for (int idx : plan.ready_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !seen[slot]) {
            seen[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
    for (int idx : plan.pending_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !seen[slot]) {
            seen[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
}

} // namespace aila
