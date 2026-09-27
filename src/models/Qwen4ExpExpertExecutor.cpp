#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace aila {

Qwen4ExpExpertExecutor::~Qwen4ExpExpertExecutor() {
    cleanup();
}

void Qwen4ExpExpertExecutor::cleanup() {
    active_cpu_leases_.clear();
    if (ctx_) {
        if (x_host_) { ctx_->free_host(x_host_); x_host_ = nullptr; }
        if (cpu_out_host_) { ctx_->free_host(cpu_out_host_); cpu_out_host_ = nullptr; }
        if (scratch_h_) { ctx_->free_host(scratch_h_); scratch_h_ = nullptr; }
        if (gpu_slots_host_) { ctx_->free_host(gpu_slots_host_); gpu_slots_host_ = nullptr; }
        if (gpu_slots_dev_) { ctx_->free_device(gpu_slots_dev_); gpu_slots_dev_ = nullptr; }
        if (gpu_ranks_host_) { ctx_->free_host(gpu_ranks_host_); gpu_ranks_host_ = nullptr; }
        if (gpu_ranks_dev_) { ctx_->free_device(gpu_ranks_dev_); gpu_ranks_dev_ = nullptr; }
        if (gpu_tokens_host_) { ctx_->free_host(gpu_tokens_host_); gpu_tokens_host_ = nullptr; }
        if (gpu_tokens_dev_) { ctx_->free_device(gpu_tokens_dev_); gpu_tokens_dev_ = nullptr; }
        if (weights_dev_) { ctx_->free_device(weights_dev_); weights_dev_ = nullptr; }
        if (down_acc_dev_) { ctx_->free_device(down_acc_dev_); down_acc_dev_ = nullptr; }
    }
    initialized_ = false;
}

void Qwen4ExpExpertExecutor::init(
    Context& ctx,
    Qwen4ExpExpertCache& cache,
    int max_seq_len,
    int top_k,
    int hidden_size,
    int intermediate_size) {
    cleanup();

    ctx_ = &ctx;
    cache_ = &cache;
    max_seq_len_ = std::max(1, max_seq_len);
    top_k_ = std::max(1, top_k);
    hidden_size_ = hidden_size > 0 ? hidden_size : 2560;
    intermediate_size_ = intermediate_size > 0 ? intermediate_size : 640;

    const int env_max_cpu = aila::env::read_int("AILA_Q4E_CPU_MAX_EXPERTS", 2);
    max_cpu_credits_ = std::max(1, std::min(4, env_max_cpu));
    const int env_max_tokens = aila::env::read_int("AILA_Q4E_CPU_MAX_TOKENS", 4);
    max_cpu_tokens_ = std::max(1, std::min(4, env_max_tokens));

    const size_t max_entries = static_cast<size_t>(max_seq_len_) * top_k_;
    active_plan_.entries.reserve(max_entries);
    active_plan_.ready_gpu_indices.reserve(max_entries);
    active_plan_.pending_gpu_indices.reserve(max_entries);
    active_plan_.cpu_indices.reserve(max_entries);
    active_plan_.top_k = top_k_;

    active_cpu_leases_.reserve(static_cast<size_t>(max_cpu_credits_));

    // Preallocate pinned host and device buffers
    x_host_ = static_cast<bf16*>(ctx_->alloc_host(static_cast<size_t>(max_cpu_tokens_) * hidden_size_ * sizeof(bf16)));
    cpu_out_host_ = static_cast<float*>(ctx_->alloc_host(static_cast<size_t>(max_cpu_credits_) * hidden_size_ * sizeof(float)));
    scratch_h_ = static_cast<bf16*>(ctx_->alloc_host(static_cast<size_t>(max_cpu_tokens_) * intermediate_size_ * sizeof(bf16)));

    const size_t max_gpu_tasks = static_cast<size_t>(max_seq_len_) * top_k_;
    gpu_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_host(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_device(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_ranks_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_ranks_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    gpu_tokens_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_tokens_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    weights_dev_ = static_cast<float*>(ctx_->alloc_device(max_gpu_tasks * sizeof(float)));

    const size_t max_down_acc = static_cast<size_t>(max_cpu_tokens_) * top_k_ * hidden_size_;
    down_acc_dev_ = static_cast<float*>(ctx_->alloc_device(max_down_acc * sizeof(float)));

    stats_.reset();
    slot_seen_.assign(cache_->num_slots(), false);
    straggler_penalty_active_ = false;
    consecutive_stragglers_ = 0;
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
    active_cpu_leases_.clear();
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

    // H3: Determine effective CPU credits for this layer
    const int effective_credits = straggler_penalty_active_ ? std::max(0, max_cpu_credits_ - 1) : max_cpu_credits_;
    const bool allow_cpu_dispatch = enable_hetero && (token_count <= max_cpu_tokens_) && (total <= 16) && (effective_credits > 0);

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
                // Hit in GPU cache: always keep on GPU (zero PCIe transfer needed)
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                active_plan_.entries.push_back(entry);
                active_plan_.ready_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.ready_gpu_tasks;
            } else if (allow_cpu_dispatch && res.host_ready && !res.gpu_copying &&
                       active_plan_.cpu_tasks < effective_credits) {
                // GPU miss, but Host-Ready and within CPU admission credits!
                // Attempt to acquire thread-safe host lease
                auto lease = cache_->acquire_host_lease(layer, expert_id);
                if (lease.is_valid()) {
                    entry.is_gpu = false;
                    entry.gpu_slot = -1;
                    active_plan_.entries.push_back(entry);
                    active_plan_.cpu_indices.push_back(idx);
                    active_cpu_leases_.push_back(std::move(lease));
                    ++active_plan_.cpu_tasks;
                } else {
                    // Lease failed (e.g. reader race or eviction): fall back to GPU pending
                    entry.is_gpu = true;
                    entry.gpu_slot = -1;
                    active_plan_.entries.push_back(entry);
                    active_plan_.pending_gpu_indices.push_back(idx);
                    ++active_plan_.gpu_tasks;
                    ++active_plan_.pending_gpu_tasks;
                }
            } else {
                // GPU pending (copying or needs DMA from host/disk)
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

    constexpr int kMaxStackPending = 16;
    int pending_experts_buf[kMaxStackPending];
    int assigned_slots_buf[kMaxStackPending];
    int* pending_experts = pending_experts_buf;
    int* assigned_slots = assigned_slots_buf;
    std::vector<int> pending_vec;
    std::vector<int> assigned_vec;

    if (plan.pending_gpu_indices.size() > kMaxStackPending) {
        pending_vec.reserve(plan.pending_gpu_indices.size());
        assigned_vec.resize(plan.pending_gpu_indices.size(), -1);
        pending_experts = pending_vec.data();
        assigned_slots = assigned_vec.data();
    }

    int num_pending = 0;
    for (int idx : plan.pending_gpu_indices) {
        int eid = plan.entries[idx].expert_id;
        bool found = false;
        for (int i = 0; i < num_pending; ++i) {
            if (pending_experts[i] == eid) {
                found = true;
                break;
            }
        }
        if (!found) {
            pending_experts[num_pending++] = eid;
        }
    }

    for (int i = 0; i < num_pending; ++i) assigned_slots[i] = -1;

    sycl::event dma_ev = cache_->acquire(plan.layer, pending_experts,
                                         num_pending,
                                         assigned_slots);
    plan.final_dma_event = dma_ev;

    // Update entries with assigned slots
    for (int idx : plan.pending_gpu_indices) {
        int eid = plan.entries[idx].expert_id;
        for (int i = 0; i < num_pending; ++i) {
            if (pending_experts[i] == eid) {
                plan.entries[idx].gpu_slot = assigned_slots[i];
                break;
            }
        }
    }
}

sycl::event Qwen4ExpExpertExecutor::execute_hetero_decode(
    Context& ctx,
    LayerMoePlan& plan,
    const bf16* x_dev,
    bf16* moe_h_dev,
    const float* router_weights_host,
    bf16* y_dev,
    int token_count,
    int top_k,
    const bf16* shared_out,
    const bf16* shared_gate,
    bool uses_nested) {
    if (!initialized_ || token_count <= 0) return {};

    const size_t total_acts = static_cast<size_t>(token_count) * top_k;
    // Copy router weights to device for ordered rank accumulation
    ctx.queue().memcpy(weights_dev_, router_weights_host, total_acts * sizeof(float));

    // 1. If CPU tasks exist, fetch input activations x_dev to pinned host memory (5 KB for decode)
    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        ctx.queue().memcpy(x_host_, x_dev, static_cast<size_t>(token_count) * hidden_size_ * sizeof(bf16)).wait();
    }

    // 2. Launch GPU execution FIRST so GPU hardware runs asynchronously while CPU computes
    sycl::event down_ev{};
    if (plan.gpu_tasks > 0) {
        int g_idx = 0;
        for (int idx : plan.ready_gpu_indices) {
            gpu_slots_host_[g_idx] = cache_->get_slot_dev(plan.entries[idx].gpu_slot);
            gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
            gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
            ++g_idx;
        }
        for (int idx : plan.pending_gpu_indices) {
            gpu_slots_host_[g_idx] = cache_->get_slot_dev(plan.entries[idx].gpu_slot);
            gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
            gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
            ++g_idx;
        }
        stats_.gpu_tasks_dispatched += plan.gpu_tasks;

        // Launch GPU gate_up (reading host pointers gpu_slots_host_ and gpu_tokens_host_)
        sycl::event gu_ev = ops::moe_decode_gate_up_swiglu_grouped(
            ctx, gpu_slots_host_, nullptr, x_dev, moe_h_dev, plan.gpu_tasks,
            hidden_size_, intermediate_size_, 64,
            plan.final_dma_event != sycl::event{} ? &plan.final_dma_event : nullptr,
            gpu_tokens_host_);

        // Launch GPU down_to_rank_f32 writing directly into rank slots of down_acc_dev_
        down_ev = ops::moe_decode_down_to_rank_f32(
            ctx, gpu_slots_host_, gpu_ranks_host_, moe_h_dev, down_acc_dev_,
            plan.gpu_tasks, top_k, intermediate_size_, hidden_size_, 64,
            gpu_tokens_host_, nullptr, &gu_ev, uses_nested);

        // Record consumer reader fences for the GPU slots used
        for (int idx : plan.ready_gpu_indices) {
            cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
        }
        for (int idx : plan.pending_gpu_indices) {
            cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
        }
    }

    // 3. Concurrent CPU execution for Host-ready miss experts (runs in parallel with GPU!)
    sycl::event last_cpu_h2d_ev{};
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    bool cpu_did_work = false;

    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        cpu_did_work = true;

        for (int c = 0; c < plan.cpu_tasks; ++c) {
            const int plan_idx = plan.cpu_indices[c];
            const auto& entry = plan.entries[plan_idx];
            const int t = entry.token_idx;
            const int r = entry.rank;
            const bf16* token_x = x_host_ + static_cast<size_t>(t) * hidden_size_;
            float* out_slot = cpu_out_host_ + static_cast<size_t>(c) * hidden_size_;

            // Execute H2a CPU forward kernel (unweighted FP32 output)
            ops::cpu::cpu_expert_forward(
                active_cpu_leases_[c].slot_host(),
                token_x,
                out_slot,
                1,
                scratch_h_,
                nullptr,
                1.0f,
                false,
                hidden_size_,
                intermediate_size_,
                hidden_size_,
                64);

            // Release lease immediately after reading weights
            active_cpu_leases_[c].release();

            // Transfer unweighted result to device down_acc_dev_ slot: [(t * top_k + r) * hidden_size]
            float* dst_acc = down_acc_dev_ + (static_cast<size_t>(t) * top_k + r) * hidden_size_;
            last_cpu_h2d_ev = ctx.queue().memcpy(dst_acc, out_slot, static_cast<size_t>(hidden_size_) * sizeof(float));
        }

        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        const uint64_t cpu_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_t1 - cpu_t0).count();
        stats_.cpu_compute_ns += cpu_elapsed_ns;
        stats_.cpu_tasks_dispatched += plan.cpu_tasks;
        stats_.cpu_tasks_completed += plan.cpu_tasks;
        active_cpu_leases_.clear();
    }

    // 4. H3: Straggler Monitoring & Protection
    if (cpu_did_work && plan.gpu_tasks > 0) {
        if (last_cpu_h2d_ev != sycl::event{}) {
            last_cpu_h2d_ev.wait();
        }
        const auto gpu_status = down_ev.get_info<sycl::info::event::command_execution_status>();
        if (gpu_status == sycl::info::event_command_status::complete) {
            // GPU finished before CPU work/copy completed -> CPU was a straggler!
            stats_.cpu_straggler_count++;
            consecutive_stragglers_++;
            if (consecutive_stragglers_ >= 2) {
                straggler_penalty_active_ = true;
            }
        } else {
            // CPU finished while GPU was still executing -> successful pipeline overlap!
            stats_.cpu_overlapped_count++;
            consecutive_stragglers_ = 0;
            straggler_penalty_active_ = false;
        }
    }

    // 5. Ordered Rank Merge: merges all ranks in strict rank order (0..top_k-1)
    sycl::event merge_ev = ops::moe_ordered_accumulate_down_rows(
        ctx, down_acc_dev_, weights_dev_, y_dev, token_count, top_k,
        hidden_size_, shared_out, shared_gate,
        plan.gpu_tasks > 0 ? &down_ev : (last_cpu_h2d_ev != sycl::event{} ? &last_cpu_h2d_ev : nullptr));

    stats_.layers_hetero_executed++;
    record_execution_complete(plan, merge_ev);
    return merge_ev;
}

void Qwen4ExpExpertExecutor::record_execution_complete(LayerMoePlan& plan, sycl::event finish_event) {
    if (!cache_) return;
    const int max_slot = cache_->num_slots();
    if (static_cast<int>(slot_seen_.size()) < max_slot) {
        slot_seen_.resize(max_slot, false);
    }
    std::fill(slot_seen_.begin(), slot_seen_.end(), false);

    for (int idx : plan.ready_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !slot_seen_[slot]) {
            slot_seen_[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
    for (int idx : plan.pending_gpu_indices) {
        int slot = plan.entries[idx].gpu_slot;
        if (slot >= 0 && slot < max_slot && !slot_seen_[slot]) {
            slot_seen_[slot] = true;
            cache_->record_consumer_reader_fence(slot, finish_event);
        }
    }
}

} // namespace aila
