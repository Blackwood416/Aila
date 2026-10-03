#include "Qwen4ExpExpertCache.hpp"
#include "models/Qwen4ExpHotExperts.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstring>
#include <functional>
#include <exception>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <immintrin.h>

namespace {
void trace_demand_route(int layer, const int* expert_ids, int count) {
    // AILA_DEBUG_Q4E_ROUTE_TRACE: append int32 {layer, count, ids[count]} per call, for offline
    // replacement-policy simulation (perf/qwen4exp/q4e_cache_sim.py).
    static FILE* s_trace = [] {
        const std::string path = aila::env::read_string("AILA_DEBUG_Q4E_ROUTE_TRACE", "");
        if (path.empty()) return static_cast<FILE*>(nullptr);
        FILE* fp = nullptr;
#ifdef _WIN32
        fopen_s(&fp, path.c_str(), "wb");
#else
        fp = std::fopen(path.c_str(), "wb");
#endif
        return fp;
    }();
    if (s_trace) {
        const int header[2] = {layer, count};
        std::fwrite(header, sizeof(int), 2, s_trace);
        std::fwrite(expert_ids, sizeof(int), static_cast<size_t>(count), s_trace);
        std::fflush(s_trace);
    }
}
bool is_event_complete(const sycl::event& ev) {
    try {
        return ev.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (...) {
        return false; // an unavailable status is not evidence that readers/DMA finished
    }
}
}

// Persistent worker pool for host-side expert staging fills.
// Supports high-priority demand fills (executed immediately by workers + caller)
// and background speculative prefetch tasks that yield to demand fills.
class ExpertFillPool {
public:
    explicit ExpertFillPool(int threads) {
        for (int i = 1; i < threads; ++i) {
            workers_.emplace_back([this] { WorkerLoop(); });
        }
    }

    ~ExpertFillPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            demand_job_.admitting = false;
            demand_job_.in_flight = false;
            prefetch_job_.admitting = false;
            prefetch_job_.in_flight = false;
        }
        cv_work_.notify_all();
        cv_done_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    int thread_count() const { return static_cast<int>(workers_.size()) + 1; }
    bool busy() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        return demand_job_.in_flight || prefetch_job_.in_flight;
    }

    // Run a high-priority demand job on workers + calling thread, waiting until complete.
    void run_demand(int total, std::function<void(int)> fn) {
        if (total <= 0) return;
        if (workers_.empty() || total == 1) {
            for (int i = 0; i < total; ++i) fn(i);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            demand_job_.fn = std::move(fn);
            demand_job_.total = total;
            demand_job_.next = 0;
            demand_job_.done = 0;
            demand_job_.admitting = true;
            demand_job_.in_flight = true;
        }
        cv_work_.notify_all();

        // Main thread helps execute demand tasks
        for (;;) {
            std::function<void(int)> task_fn;
            int task_id = -1;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (demand_job_.admitting && demand_job_.next < demand_job_.total) {
                    task_id = demand_job_.next++;
                    task_fn = demand_job_.fn;
                }
            }
            if (task_id < 0) break;
            task_fn(task_id);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                demand_job_.done++;
                if (demand_job_.done >= demand_job_.total) {
                    cv_done_.notify_all();
                }
            }
        }

        // Wait until all tasks are completed and all active workers have finished
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [&] {
                return demand_job_.done >= demand_job_.total && demand_job_.active_workers == 0;
            });
            demand_job_.admitting = false;
            demand_job_.in_flight = false;
            demand_job_.fn = nullptr;
        }
    }

    // Fallback wrapper for run()
    void run(int total, std::function<void(int)> fn) {
        run_demand(total, std::move(fn));
    }

    // Start background prefetch job. Returns immediately.
    void start_prefetch(int total, std::function<void(int)> fn) {
        if (total <= 0) return;
        if (workers_.empty()) {
            for (int i = 0; i < total; ++i) fn(i);
            return;
        }
        cancel_prefetch_job();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            prefetch_job_.fn = std::move(fn);
            prefetch_job_.total = total;
            prefetch_job_.next = 0;
            prefetch_job_.done = 0;
            prefetch_job_.admitting = true;
            prefetch_job_.in_flight = true;
        }
        cv_work_.notify_all();
    }

    void start(int total, std::function<void(int)> fn) {
        start_prefetch(total, std::move(fn));
    }

    void cancel_prefetch_job() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!prefetch_job_.in_flight && !prefetch_job_.admitting) return;

        prefetch_job_.admitting = false;
        prefetch_job_.in_flight = false;
        prefetch_job_.next = prefetch_job_.total;

        cv_done_.wait(lock, [&] {
            return prefetch_job_.active_workers == 0;
        });

        prefetch_job_.done = prefetch_job_.total;
        prefetch_job_.fn = nullptr;
    }

    void wait_prefetch_job() {
        // Main thread helps execute prefetch tasks
        for (;;) {
            std::function<void(int)> task_fn;
            int task_id = -1;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (prefetch_job_.admitting && prefetch_job_.next < prefetch_job_.total) {
                    task_id = prefetch_job_.next++;
                    task_fn = prefetch_job_.fn;
                }
            }
            if (task_id < 0) break;
            task_fn(task_id);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                prefetch_job_.done++;
                if (prefetch_job_.done >= prefetch_job_.total) {
                    cv_done_.notify_all();
                }
            }
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [&] {
            return prefetch_job_.done >= prefetch_job_.total && prefetch_job_.active_workers == 0;
        });

        prefetch_job_.admitting = false;
        prefetch_job_.in_flight = false;
        prefetch_job_.fn = nullptr;
    }

    void cancel() {
        cancel_prefetch_job();
    }

    void wait() {
        wait_prefetch_job();
    }

private:
    struct TaskJob {
        std::function<void(int)> fn;
        int total = 0;
        int next = 0;
        int done = 0;
        int active_workers = 0;
        bool in_flight = false;
        bool admitting = false;
    };

    void WorkerLoop() {
        for (;;) {
            std::function<void(int)> task_fn;
            int task_id = -1;
            bool is_demand = false;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_work_.wait(lock, [&] {
                    return stop_ ||
                           (demand_job_.admitting && demand_job_.next < demand_job_.total) ||
                           (prefetch_job_.admitting && prefetch_job_.next < prefetch_job_.total);
                });
                if (stop_) return;

                if (demand_job_.admitting && demand_job_.next < demand_job_.total) {
                    task_id = demand_job_.next++;
                    task_fn = demand_job_.fn;
                    demand_job_.active_workers++;
                    is_demand = true;
                } else if (prefetch_job_.admitting && prefetch_job_.next < prefetch_job_.total) {
                    task_id = prefetch_job_.next++;
                    task_fn = prefetch_job_.fn;
                    prefetch_job_.active_workers++;
                    is_demand = false;
                }
            }

            if (task_id >= 0 && task_fn) {
                task_fn(task_id);

                std::lock_guard<std::mutex> lock(mutex_);
                if (is_demand) {
                    demand_job_.active_workers--;
                    demand_job_.done++;
                    if (demand_job_.done >= demand_job_.total || demand_job_.active_workers == 0) {
                        cv_done_.notify_all();
                    }
                } else {
                    prefetch_job_.active_workers--;
                    prefetch_job_.done++;
                    if (prefetch_job_.done >= prefetch_job_.total || prefetch_job_.active_workers == 0) {
                        cv_done_.notify_all();
                    }
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    bool stop_ = false;
    TaskJob demand_job_;
    TaskJob prefetch_job_;
};

Qwen4ExpExpertCache::Qwen4ExpExpertCache() : host_list_mutex_(std::make_unique<std::mutex>()) {}

Qwen4ExpExpertCache::~Qwen4ExpExpertCache() {
    cleanup();
}

Qwen4ExpExpertCache::Qwen4ExpExpertCache(Qwen4ExpExpertCache&& other) noexcept {
    *this = std::move(other);
}

Qwen4ExpExpertCache& Qwen4ExpExpertCache::operator=(Qwen4ExpExpertCache&& other) noexcept {
    if (this == &other) return *this;

    try {
        other.PublishHotsetPending(true);
        other.cancel_prefetch();
        if (other.copy_q_) {
            other.copy_q_->wait_and_throw();
        }
        if (other.fill_pool_) {
            other.fill_pool_->wait();
        }
    } catch (...) {}

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
    compact_nested_ = other.compact_nested_;

    num_slots_ = other.num_slots_;
    dev_pool_bytes_ = other.dev_pool_bytes_;
    dev_pool_chunks_ = std::move(other.dev_pool_chunks_);
    slot_dev_base_ = std::move(other.slot_dev_base_);
    layer_meta_allocs_ = std::move(other.layer_meta_allocs_);

    staging_buffer_bytes_ = other.staging_buffer_bytes_;
    pinned_staging_ptr_ = other.pinned_staging_ptr_;
    prefetch_staging_ptr_ = other.prefetch_staging_ptr_;
    prefetch_staging_slots_ = other.prefetch_staging_slots_;
    prefetch_items_ = std::move(other.prefetch_items_);
    prefetch_in_flight_ = other.prefetch_in_flight_;
    copy_q_ = std::move(other.copy_q_);
    last_dma_event_ = std::move(other.last_dma_event_);

    layer_weights_ = std::move(other.layer_weights_);
    lru_list_ = std::move(other.lru_list_);
    slot_iters_ = std::move(other.slot_iters_);
    layer_expert_to_slot_ = std::move(other.layer_expert_to_slot_);
    slot_to_key_ = std::move(other.slot_to_key_);
    slot_reserved_ = std::move(other.slot_reserved_);
    slot_pinned_ = std::move(other.slot_pinned_);
    num_pinned_slots_ = other.num_pinned_slots_;
    other.num_pinned_slots_ = 0;
    slots_dev_ = std::move(other.slots_dev_);
    gpu_slot_records_ = std::move(other.gpu_slot_records_);
    frozen_ = other.frozen_; other.frozen_ = false;
    frozen_count_ = other.frozen_count_; other.frozen_count_ = 0;
    frozen_slots_ = std::move(other.frozen_slots_);
    frozen_descriptors_published_ = other.frozen_descriptors_published_;
    frozen_indices_published_ = std::move(other.frozen_indices_published_);
    hotset_policy_=std::move(other.hotset_policy_);
    adaptive_hotset_=other.adaptive_hotset_; other.adaptive_hotset_=false;
    hotset_borrow_=other.hotset_borrow_; hotset_borrowed_=other.hotset_borrowed_;
    hotset_interval_=other.hotset_interval_; hotset_swaps_=other.hotset_swaps_; hotset_tokens_=other.hotset_tokens_;
    hotset_prompt_=other.hotset_prompt_;hotset_prompt_swaps_=other.hotset_prompt_swaps_;
    hotset_prompt_prefill_=other.hotset_prompt_prefill_;hotset_prefill_keys_=std::move(other.hotset_prefill_keys_);
    hotset_adopt_ready_=other.hotset_adopt_ready_;hotset_adopt_limit_=other.hotset_adopt_limit_;
    hotset_adopt_count_=other.hotset_adopt_count_;
    hotset_explicit_prefill_=other.hotset_explicit_prefill_;
    hotset_identity_=std::move(other.hotset_identity_); hotset_export_=std::move(other.hotset_export_);
    other.hotset_export_.clear(); hotset_keys_=std::move(other.hotset_keys_);
    hotset_cold_slots_=std::move(other.hotset_cold_slots_);hotset_cold_cursor_=other.hotset_cold_cursor_;hotset_ring_=other.hotset_ring_;
    hotset_cold_policy_=other.hotset_cold_policy_;hotset_cold_lru_=std::move(other.hotset_cold_lru_);
    hotset_cold_iters_=std::move(other.hotset_cold_iters_);hotset_cold_member_=std::move(other.hotset_cold_member_);
    resident_descriptors_dev_ = std::move(other.resident_descriptors_dev_);
    resident_indices_dev_ = std::move(other.resident_indices_dev_);
    resident_descriptors_host_ = std::move(other.resident_descriptors_host_);
    resident_descriptor_generations_ = std::move(other.resident_descriptor_generations_);
    resident_indices_host_ = std::move(other.resident_indices_host_);
    resident_descriptor_upload_ = std::move(other.resident_descriptor_upload_);
    resident_index_uploads_ = std::move(other.resident_index_uploads_);
    stats_ = other.stats_;
    cost_aware_eviction_ = other.cost_aware_eviction_;
    cache_trace_ = std::move(other.cache_trace_);
    trace_forward_id_ = other.trace_forward_id_;
    trace_acquire_id_ = other.trace_acquire_id_;
    initialized_ = other.initialized_;
    faulted_ = other.faulted_;
    fill_pool_ = std::move(other.fill_pool_);
    fill_parts_ = other.fill_parts_;

    host_pool_enabled_ = other.host_pool_enabled_;
    host_pool_bytes_ = other.host_pool_bytes_;
    num_host_slots_ = other.num_host_slots_;
    host_pool_chunks_ = std::move(other.host_pool_chunks_);
    host_slot_ptrs_ = std::move(other.host_slot_ptrs_);
    host_layer_expert_to_slot_ = std::move(other.host_layer_expert_to_slot_);
    host_slot_to_key_ = std::move(other.host_slot_to_key_);
    host_slot_records_ = std::move(other.host_slot_records_);
    host_empty_slots_ = std::move(other.host_empty_slots_);
    host_probationary_list_ = std::move(other.host_probationary_list_);
    host_protected_list_ = std::move(other.host_protected_list_);
    host_slot_iters_ = std::move(other.host_slot_iters_);
    host_slot_is_protected_ = std::move(other.host_slot_is_protected_);
    host_slot_state_ = std::move(other.host_slot_state_);
    host_slot_reserved_ = std::move(other.host_slot_reserved_);
    host_slot_in_batch_readers_ = std::move(other.host_slot_in_batch_readers_);
    batch_reserved_hslots_scratch_ = std::move(other.batch_reserved_hslots_scratch_);
    host_slot_dma_events_ = std::move(other.host_slot_dma_events_);
    protected_ratio_ = other.protected_ratio_;

    misses_scratch_ = std::move(other.misses_scratch_);
    batch_reserved_slots_scratch_ = std::move(other.batch_reserved_slots_scratch_);
    to_fill_scratch_ = std::move(other.to_fill_scratch_);
    assigned_hslots_scratch_ = std::move(other.assigned_hslots_scratch_);
    assigned_hslot_gens_scratch_ = std::move(other.assigned_hslot_gens_scratch_);
    parts_left_scratch_ = std::move(other.parts_left_scratch_);
    parts_left_capacity_ = other.parts_left_capacity_;
    dma_timing_stats_ = other.dma_timing_stats_;
    last_batch_submit_end_time_ = other.last_batch_submit_end_time_;
    last_batch_dma_event_ = std::move(other.last_batch_dma_event_);
    batch_dma_events_ = std::move(other.batch_dma_events_);
    new_dma_batch_pending_ = other.new_dma_batch_pending_;
    other.dma_timing_stats_ = {};
    other.new_dma_batch_pending_ = false;

    other.ctx_ = nullptr;
    other.weights_ = nullptr;
    other.pinned_staging_ptr_ = nullptr;
    other.prefetch_staging_ptr_ = nullptr;
    other.prefetch_staging_slots_ = 0;
    other.prefetch_in_flight_ = false;
    other.last_dma_event_ = sycl::event{};
    other.initialized_ = false;
    other.faulted_ = false;
    other.num_slots_ = 0;
    other.num_host_slots_ = 0;
    other.host_pool_enabled_ = false;
    other.parts_left_capacity_ = 0;
    other.protected_ratio_ = 0.75f;
    other.stats_ = {};

    host_list_mutex_ = other.host_list_mutex_ ? std::move(other.host_list_mutex_) : std::make_unique<std::mutex>();
    if (!host_list_mutex_) host_list_mutex_ = std::make_unique<std::mutex>();

    return *this;
}

void Qwen4ExpExpertCache::cleanup() {
    try { PublishHotsetPending(true); } catch (...) {
        if(copy_q_)try {copy_q_->wait();} catch (...) {}
    }
    hotset_pending_.clear();
    if(!hotset_export_.empty()) {
        try { hotset_policy_.Save(hotset_export_,hotset_identity_); }
        catch(const std::exception& e) { AILA_LOG_ERROR("[Q4E Hotset] export failed: %s",e.what()); }
        hotset_export_.clear();
    }
    adaptive_hotset_=false; hotset_borrowed_=false; hotset_keys_.clear();
    hotset_cold_slots_.clear();hotset_cold_cursor_=0;
    hotset_cold_lru_.clear();hotset_cold_iters_.clear();hotset_cold_member_.clear();
    cache_trace_.reset();
    trace_forward_id_ = trace_acquire_id_ = 0;
    try {
        cancel_prefetch();
    } catch (...) {}
    if (ctx_) {
        try {
            ctx_->synchronize();
        } catch (...) {}
        // The copy queue may still use pool/staging pointers after the main
        // queue has drained. Wait before releasing either allocation.
        if (copy_q_) {
            try { copy_q_->wait_and_throw(); } catch (...) {}
        }
        for (uint8_t* meta : layer_meta_allocs_) {
            if (meta) ctx_->free_device(meta);
        }
        if (prefetch_staging_ptr_) {
            ctx_->free_host(prefetch_staging_ptr_);
            prefetch_staging_ptr_ = nullptr;
        }
        for (uint8_t* chunk : dev_pool_chunks_) {
            ctx_->free_device(chunk);
        }
        if (pinned_staging_ptr_) {
            ctx_->free_host(pinned_staging_ptr_);
            pinned_staging_ptr_ = nullptr;
        }
        for (void* chunk : host_pool_chunks_) {
            ctx_->free_host(chunk);
        }
    }
    dev_pool_chunks_.clear();
    resident_descriptors_dev_ = Tensor{};
    resident_indices_dev_ = Tensor{};
    resident_descriptors_host_.clear();
    resident_descriptor_generations_.clear();
    resident_indices_host_.clear();
    resident_index_uploads_.clear();
    resident_descriptor_upload_ = {};
    layer_meta_allocs_.clear();
    slot_dev_base_.clear();
    ctx_ = nullptr;
    weights_ = nullptr;
    dev_pool_bytes_ = 0;
    staging_buffer_bytes_ = 0;
    num_slots_ = 0;
    compact_nested_ = false;
    initialized_ = false;
    faulted_ = false;
    layer_weights_.clear();
    lru_list_.clear();
    slot_iters_.clear();
    layer_expert_to_slot_.clear();
    slot_to_key_.clear();
    slot_reserved_.clear();
    slot_pinned_.clear();
    frozen_slots_.clear(); frozen_ = false; frozen_count_ = 0;
    frozen_indices_published_.clear(); frozen_descriptors_published_=false;
    num_pinned_slots_ = 0;
    slots_dev_.clear();
    gpu_slot_records_.clear();
    host_pool_chunks_.clear();
    host_slot_ptrs_.clear();
    host_layer_expert_to_slot_.clear();
    host_slot_to_key_.clear();
    host_slot_records_.clear();
    host_empty_slots_.clear();
    host_probationary_list_.clear();
    host_protected_list_.clear();
    host_slot_iters_.clear();
    host_slot_is_protected_.clear();
    host_slot_reserved_.clear();
    host_slot_in_batch_readers_.clear();
    batch_reserved_hslots_scratch_.clear();
    host_slot_dma_events_.clear();
    host_slot_state_.reset();
    host_pool_bytes_ = 0;
    num_host_slots_ = 0;
    host_pool_enabled_ = false;
    protected_ratio_ = 0.75f;
    misses_scratch_.clear();
    batch_reserved_slots_scratch_.clear();
    to_fill_scratch_.clear();
    assigned_hslots_scratch_.clear();
    parts_left_scratch_.reset();
    parts_left_capacity_ = 0;
    fill_pool_.reset();
    prefetch_items_.clear();
    last_dma_event_ = sycl::event{};
    last_batch_dma_event_ = sycl::event{};
    batch_dma_events_.clear();
    dma_timing_stats_ = {};
    new_dma_batch_pending_ = false;
    if (copy_q_) {
        try {
            copy_q_->wait();
        } catch (...) {}
        copy_q_.reset();
    }
}

bool Qwen4ExpExpertCache::init(Context& ctx,
                              ModelWeights& weights,
                              int num_layers,
                              size_t cache_bytes,
                              int num_experts_per_layer,
                              int hidden_size,
                              int intermediate_size,
                              int blocksize,
                              std::string* error_message,
                              bool force_nested_codes,
                              const std::string& trace_suffix) {
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

    // down: hidden_size rows, intermediate_size cols
    const int dn_rows = hidden_size_;
    dn_packed_bytes_ = static_cast<size_t>(dn_rows) * (intermediate_size_ / 2);
    dn_absmax_count_ = static_cast<size_t>(dn_rows) * (intermediate_size_ / blocksize_);

    // Load host refs before sizing the pool: nested u8 codes use one byte per
    // absmax instead of four, so the same VRAM budget can hold more experts.
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
    const auto eligible = [&](const LayerWeights& lw) {
        return lw.valid && lw.gate_up.quant_state.nested && lw.down.quant_state.nested &&
               lw.gate_up.absmax.dtype == DT_U8 && lw.down.absmax.dtype == DT_U8 &&
               lw.gate_up.nested_absmax.valid() && lw.gate_up.nested_absmax.dtype == DT_F32 &&
               lw.gate_up.nested_quant_map.valid() && lw.gate_up.nested_quant_map.dtype == DT_F32 &&
               lw.gate_up.nested_quant_map.bytes >= 256 * sizeof(float) &&
               lw.down.nested_absmax.valid() && lw.down.nested_absmax.dtype == DT_F32 &&
               lw.down.nested_quant_map.valid() && lw.down.nested_quant_map.dtype == DT_F32 &&
               lw.down.nested_quant_map.bytes >= 256 * sizeof(float) &&
               lw.gate_up.quant_state.nested_blocksize > 0 &&
               lw.down.quant_state.nested_blocksize > 0 &&
               gu_absmax_count_ % static_cast<size_t>(lw.gate_up.quant_state.nested_blocksize) == 0 &&
               dn_absmax_count_ % static_cast<size_t>(lw.down.quant_state.nested_blocksize) == 0;
    };
    const bool all_nested = !layer_weights_.empty() &&
                            std::all_of(layer_weights_.begin(), layer_weights_.end(), eligible);
    compact_nested_ = force_nested_codes ||
                      (aila::env::read_flag("AILA_Q4E_ABSMAX_CODES", true) && all_nested);
    gu_absmax_bytes_ = gu_absmax_count_ * (compact_nested_ ? sizeof(uint8_t) : sizeof(float));
    dn_absmax_bytes_ = dn_absmax_count_ * (compact_nested_ ? sizeof(uint8_t) : sizeof(float));

    gu_packed_offset_ = 0;
    gu_absmax_offset_ = gu_packed_bytes_;
    dn_packed_offset_ = gu_absmax_offset_ + gu_absmax_bytes_;
    dn_absmax_offset_ = dn_packed_offset_ + dn_packed_bytes_;
    slot_bytes_ = dn_absmax_offset_ + dn_absmax_bytes_;

    if (slot_bytes_ == 0 || (compact_nested_ && dn_packed_offset_ % alignof(uint32_t) != 0)) {
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
        const int chunk_mb = std::max(64, aila::env::read_int("AILA_Q4E_EXPERT_CHUNK_MB", 1024));
        const int slots_per_chunk = std::max(1, static_cast<int>(static_cast<size_t>(chunk_mb) * 1024 * 1024 / slot_bytes_));
        slot_dev_base_.resize(num_slots_);
        for (int s0 = 0; s0 < num_slots_; s0 += slots_per_chunk) {
            const int n = std::min(slots_per_chunk, num_slots_ - s0);
            uint8_t* chunk = static_cast<uint8_t*>(ctx_->alloc_device(static_cast<size_t>(n) * slot_bytes_));
            dev_pool_chunks_.push_back(chunk);
            for (int i = 0; i < n; ++i) {
                slot_dev_base_[s0 + i] = chunk + static_cast<size_t>(i) * slot_bytes_;
            }
        }
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Qwen4ExpExpertCache device alloc failed: ") + e.what();
        cleanup();
        return false;
    }

    // Pinned host staging buffer for DDR3 -> PCIe 3.0 DMA (double-buffered in acquire()).
    const size_t staging_mb = static_cast<size_t>(std::max(16, aila::env::read_int("AILA_Q4E_STAGING_MB", 128)));
    staging_buffer_bytes_ = std::max<size_t>(slot_bytes_ * 16, staging_mb * 1024 * 1024);
    if (cache_bytes < staging_buffer_bytes_) {
        staging_buffer_bytes_ = std::max<size_t>(slot_bytes_ * 2, cache_bytes);
    }

    try {
        try {
            pinned_staging_ptr_ = static_cast<uint8_t*>(ctx_->alloc_host(staging_buffer_bytes_));
        } catch (const std::exception& e) {
            if (error_message) *error_message = std::string("pinned_staging alloc failed (") + std::to_string(staging_buffer_bytes_) + " bytes): " + e.what();
            cleanup();
            return false;
        }

        try {
            prefetch_staging_slots_ = 16;
            prefetch_staging_ptr_ = static_cast<uint8_t*>(ctx_->alloc_host(static_cast<size_t>(prefetch_staging_slots_) * slot_bytes_));
        } catch (const std::exception& e) {
            if (error_message) *error_message = std::string("prefetch_staging alloc failed: ") + e.what();
            cleanup();
            return false;
        }

        try {
            try {
                copy_q_ = std::make_unique<sycl::queue>(
                    ctx_->queue().get_context(), ctx_->queue().get_device(),
                    sycl::property_list{sycl::property::queue::in_order(),
                                        sycl::ext::intel::property::queue::immediate_command_list(),
                                        sycl::property::queue::enable_profiling()});
            } catch (...) {
                copy_q_ = std::make_unique<sycl::queue>(
                    ctx_->queue().get_context(), ctx_->queue().get_device(),
                    sycl::property_list{sycl::property::queue::in_order(),
                                        sycl::ext::intel::property::queue::immediate_command_list()});
            }
        } catch (const std::exception& e) {
            if (error_message) *error_message = std::string("copy_q creation failed: ") + e.what();
            cleanup();
            return false;
        }
    } catch (...) {
        cleanup();
        return false;
    }

    // Initialize Hot Host RAM Pool (pinned USM host memory allocated in 1 GB chunks).
    // Default 16384 MB (16 GiB). If <= 0, disabled. Only enabled for full model runs (cache_bytes >= 1GB) or unit test flag.
    const int host_ram_mb = aila::env::read_int("AILA_Q4E_HOT_RAM_MB", 16384);
    const bool test_host_pool = aila::env::read_flag("AILA_TEST_Q4E_HOST_POOL", false);
    const int test_host_slots = aila::env::read_int("AILA_TEST_Q4E_HOST_SLOTS", 0);
    if ((host_ram_mb > 0 || test_host_slots > 0) && (cache_bytes >= 1024ULL * 1024ULL * 1024ULL || test_host_pool)) {
        size_t target_host_bytes = (test_host_slots > 0 && test_host_pool)
            ? static_cast<size_t>(test_host_slots) * slot_bytes_
            : static_cast<size_t>(host_ram_mb) * 1024ULL * 1024ULL;
        const size_t total_model_experts = static_cast<size_t>(num_layers_) * num_experts_per_layer_;
        const size_t max_needed_bytes = total_model_experts * slot_bytes_;
        target_host_bytes = std::min(target_host_bytes, max_needed_bytes);
        const size_t chunk_bytes = test_host_pool ? target_host_bytes : (1024ULL * 1024ULL * 1024ULL);
        size_t allocated = 0;
        while (allocated < target_host_bytes) {
            size_t this_chunk = std::min(chunk_bytes, target_host_bytes - allocated);
            size_t slots_in_chunk = this_chunk / slot_bytes_;
            if (slots_in_chunk == 0) break;
            size_t alloc_bytes = slots_in_chunk * slot_bytes_;
            void* chunk_ptr = nullptr;
            try {
                chunk_ptr = ctx_->alloc_host(alloc_bytes);
            } catch (const std::exception& e) {
                AILA_LOG_WARN("[Q4E ExpertCache] Host pool allocation stopped at %zu MB: %s",
                              allocated / (1024 * 1024), e.what());
                break;
            }
            host_pool_chunks_.push_back(chunk_ptr);
            for (size_t s = 0; s < slots_in_chunk; ++s) {
                host_slot_ptrs_.push_back(static_cast<uint8_t*>(chunk_ptr) + s * slot_bytes_);
            }
            allocated += alloc_bytes;
        }
        host_pool_bytes_ = allocated;
        num_host_slots_ = static_cast<int>(host_slot_ptrs_.size());
        if (num_host_slots_ > 0) {
            host_pool_enabled_ = true;
            protected_ratio_ = aila::env::read_float("AILA_Q4E_HOST_PROTECTED_RATIO", 0.75f);
            host_slot_to_key_.assign(num_host_slots_, UINT64_MAX);
            host_slot_records_.assign(num_host_slots_, HostSlotRecord{});
            host_slot_iters_.resize(num_host_slots_);
            host_slot_is_protected_.assign(num_host_slots_, false);
            host_slot_reserved_.assign(num_host_slots_, false);
            host_slot_in_batch_readers_.assign(num_host_slots_, false);
            host_slot_dma_events_.assign(num_host_slots_, sycl::event{});
            batch_reserved_hslots_scratch_.clear();
            batch_reserved_hslots_scratch_.reserve(num_host_slots_);
            host_empty_slots_.clear();
            host_empty_slots_.reserve(num_host_slots_);
            for (int h = num_host_slots_ - 1; h >= 0; --h) {
                host_empty_slots_.push_back(h);
            }
            host_probationary_list_.clear();
            host_protected_list_.clear();
            host_layer_expert_to_slot_.assign(static_cast<size_t>(num_layers_) * num_experts_per_layer_, -1);
            host_slot_state_ = std::make_unique<std::atomic<uint8_t>[]>(num_host_slots_);
            for (int h = 0; h < num_host_slots_; ++h) {
                host_slot_state_[h].store(static_cast<uint8_t>(ExpertSlotState::Empty), std::memory_order_relaxed);
            }
            AILA_LOG_INFO("[Q4E ExpertCache] Hot Host RAM Pool initialized: slots=%d pool=%.1fMB",
                          num_host_slots_, host_pool_bytes_ / (1024.0 * 1024.0));
        }
    }

    // Setup slot tracking & descriptors
    lru_list_.clear();
    slot_iters_.resize(num_slots_);
    layer_expert_to_slot_.assign(static_cast<size_t>(num_layers_) * num_experts_per_layer_, -1);
    slot_to_key_.assign(num_slots_, UINT64_MAX);
    slot_reserved_.assign(num_slots_, false);
    slot_pinned_.assign(num_slots_, false);
    frozen_slots_.assign(num_slots_, false);
    frozen_indices_published_.assign(num_layers_,false);
    num_pinned_slots_ = 0;
    gpu_slot_records_.assign(num_slots_, GpuSlotRecord{});

    slots_dev_.resize(num_slots_);

    for (int s = 0; s < num_slots_; ++s) {
        lru_list_.push_back(s);
        auto it = lru_list_.end();
        --it;
        slot_iters_[s] = it;

        uint8_t* base = slot_dev_base_[s];
        slots_dev_[s].gate_up_packed = base + gu_packed_offset_;
        slots_dev_[s].gate_up_absmax = compact_nested_ ? nullptr : reinterpret_cast<const float*>(base + gu_absmax_offset_);
        slots_dev_[s].down_packed = base + dn_packed_offset_;
        slots_dev_[s].down_absmax = compact_nested_ ? nullptr : reinterpret_cast<const float*>(base + dn_absmax_offset_);
    }

    layer_meta_allocs_.resize(num_layers_, nullptr);
    try {
        if (compact_nested_) {
            for (int L = 0; L < num_layers_; ++L) {
                if (layer_weights_[L].valid) configure_layer_nested_metadata(L);
            }
        }
    } catch (const std::exception& e) {
        if (error_message) *error_message = std::string("Qwen4ExpExpertCache nested metadata failed: ") + e.what();
        cleanup();
        return false;
    }

    stats_ = {};

    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int fill_threads = aila::env::read_int("AILA_Q4E_FILL_THREADS", std::min(8, hw));
    fill_threads = std::clamp(fill_threads, 1, hw);
    fill_parts_ = std::clamp(aila::env::read_int("AILA_Q4E_FILL_PARTS", 8), 1, 64);
    cost_aware_eviction_ = aila::env::read_flag("AILA_Q4E_COST_AWARE_EVICTION", true);

    fill_pool_ = std::make_unique<ExpertFillPool>(fill_threads);

    // Read directly into the bounded host pool instead of also faulting the expert
    // payload into this process's mmap working set. Long prompts can otherwise touch
    // most of the model on top of the 16 GiB pinned pool. Preserve explicit mmap opt-in
    // and the reader's existing fallback for sources without valid file spans.
#ifdef _WIN32
    constexpr const char* default_reader = "readfile";
#else
    constexpr const char* default_reader = "mmap";
#endif
    const std::string reader_env = aila::env::read_string("AILA_Q4E_EXPERT_READER", default_reader);
    aila::ExpertReaderType reader_type = aila::ExpertReaderType::Mmap;
    if (reader_env == "readfile" || reader_env == "overlapped" || reader_env == "buffered_readfile") {
        reader_type = aila::ExpertReaderType::OverlappedReadFile;
    }

    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = gu_packed_bytes_;
    geom.gu_absmax_count = gu_absmax_count_;
    geom.gu_absmax_bytes = gu_absmax_bytes_;
    geom.dn_packed_bytes = dn_packed_bytes_;
    geom.dn_absmax_count = dn_absmax_count_;
    geom.dn_absmax_bytes = dn_absmax_bytes_;
    geom.gu_packed_offset = gu_packed_offset_;
    geom.gu_absmax_offset = gu_absmax_offset_;
    geom.dn_packed_offset = dn_packed_offset_;
    geom.dn_absmax_offset = dn_absmax_offset_;
    geom.slot_bytes = slot_bytes_;
    geom.compact_nested = compact_nested_;

    std::vector<aila::ExpertLayerSource> sources(num_layers_);
    for (int L = 0; L < num_layers_; ++L) {
        if (L < static_cast<int>(layer_weights_.size()) && layer_weights_[L].valid) {
            sources[L].gate_up = layer_weights_[L].gate_up;
            sources[L].down = layer_weights_[L].down;
            sources[L].valid = true;
        }
    }

    reader_ = aila::CreateExpertReader(reader_type, geom, weights.shard_paths(), sources);

    misses_scratch_.clear();
    misses_scratch_.reserve(num_experts_per_layer_);
    batch_dma_events_.clear();
    batch_dma_events_.reserve(num_experts_per_layer_);
    dma_timing_stats_ = {};
    new_dma_batch_pending_ = false;
    batch_reserved_slots_scratch_.clear();
    batch_reserved_slots_scratch_.reserve(num_experts_per_layer_);
    to_fill_scratch_.clear();
    to_fill_scratch_.reserve(num_experts_per_layer_);
    assigned_hslots_scratch_.clear();
    assigned_hslots_scratch_.reserve(std::max(1, num_host_slots_));
    parts_left_capacity_ = std::max(static_cast<size_t>(num_experts_per_layer_), static_cast<size_t>(std::max(1, num_host_slots_)));
    parts_left_scratch_ = std::make_unique<std::atomic<int>[]>(parts_left_capacity_);
    prefetch_items_.reserve(std::max(prefetch_staging_slots_, num_host_slots_));

    AILA_LOG_INFO("[Q4E ExpertCache] slots=%d slot_bytes=%zu pool=%.1fMB nested_codes=%d fill_threads=%d fill_parts=%d reader=%s",
                  num_slots_, slot_bytes_, dev_pool_bytes_ / (1024.0 * 1024.0),
                  compact_nested_ ? 1 : 0, fill_threads, fill_parts_, reader_->name());

    initialized_ = true;
    const auto trace_path = aila::env::read_string("AILA_Q4E_CACHE_TRACE", "");
    if (!trace_path.empty()) {
        cache_trace_ = std::make_unique<Qwen4ExpCacheTrace>(trace_path + trace_suffix);
        cache_trace_->Row("config") << ",\"schema\":1,\"slots\":" << num_slots_
            << ",\"slot_bytes\":" << slot_bytes_ << ",\"layers\":" << num_layers_
            << ",\"experts\":" << num_experts_per_layer_ << ",\"host_slots\":" << num_host_slots_
            << ",\"cost_aware\":" << cost_aware_eviction_;
        cache_trace_->End();
        cache_trace_->Flush();
    }
    return true;
}

void Qwen4ExpExpertCache::trace_forward(int position, int tokens, bool begin, bool need_logits) {
    if (!cache_trace_) return;
    if (begin) ++trace_forward_id_;
    cache_trace_->Row(begin ? "forward_begin" : "forward_end")
        << ",\"forward\":" << trace_forward_id_ << ",\"position\":" << position
        << ",\"tokens\":" << tokens << ",\"logits\":" << need_logits
        << ",\"prefetch_issued_bytes\":" << stats_.prefetch_issued_bytes
        << ",\"prefetch_useful_bytes\":" << stats_.prefetch_useful_bytes
        << ",\"prefetch_cancelled_bytes\":" << stats_.prefetch_cancelled_bytes
        << ",\"prefetch_wasted_bytes\":" << stats_.prefetch_wasted_bytes;
    cache_trace_->End();
    if (!begin) cache_trace_->Flush();
}

void Qwen4ExpExpertCache::trace_host_interval(const char* name, int layer, int64_t begin_ns, int64_t end_ns) {
    if (!cache_trace_) return;
    cache_trace_->Row("host") << ",\"forward\":" << trace_forward_id_
        << ",\"name\":\"" << name << "\",\"layer\":" << layer
        << ",\"begin_ns\":" << begin_ns << ",\"end_ns\":" << end_ns;
    cache_trace_->End();
}

void Qwen4ExpExpertCache::trace_router(int layer, const int* ids, int rows, int top_k) {
    if (!cache_trace_) return;
    auto& out = cache_trace_->Row("route");
    out << ",\"forward\":" << trace_forward_id_ << ",\"layer\":" << layer
        << ",\"rows\":" << rows << ",\"top_k\":" << top_k << ",\"ids\":";
    Qwen4ExpCacheTrace::Ints(out, ids, rows * top_k);
    cache_trace_->End();
}

void Qwen4ExpExpertCache::trace_acquire_end(const ExpertCacheStats& before, int64_t begin_ns) {
    if (!cache_trace_) return;
    cache_trace_->Row("acquire_end") << ",\"acquire\":" << trace_acquire_id_
        << ",\"begin_ns\":" << begin_ns
        << ",\"hits\":" << stats_.hit_count - before.hit_count
        << ",\"misses\":" << stats_.miss_count - before.miss_count
        << ",\"bytes\":" << stats_.total_transferred_bytes - before.total_transferred_bytes
        << ",\"host_hits\":" << stats_.host_pool_hits - before.host_pool_hits
        << ",\"file_fills\":" << stats_.host_pool_misses - before.host_pool_misses
        << ",\"fill_ns\":" << stats_.fill_ns - before.fill_ns
        << ",\"prefetch_wait_ns\":" << stats_.prefetch_wait_ns - before.prefetch_wait_ns
        << ",\"prefetch_used\":" << stats_.prefetch_used - before.prefetch_used
        << ",\"prefetch_cancelled_bytes\":" << stats_.prefetch_cancelled_bytes - before.prefetch_cancelled_bytes;
    cache_trace_->End();
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
    if (compact_nested_) configure_layer_nested_metadata(layer);
    if (reader_) {
        aila::ExpertLayerSource src;
        src.gate_up = gate_up;
        src.down = down;
        src.valid = true;
        reader_->set_layer_source(layer, src);
    }
}

void Qwen4ExpExpertCache::set_reader_type(aila::ExpertReaderType type) {
    if (!initialized_) return;
    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = gu_packed_bytes_;
    geom.gu_absmax_count = gu_absmax_count_;
    geom.gu_absmax_bytes = gu_absmax_bytes_;
    geom.dn_packed_bytes = dn_packed_bytes_;
    geom.dn_absmax_count = dn_absmax_count_;
    geom.dn_absmax_bytes = dn_absmax_bytes_;
    geom.gu_packed_offset = gu_packed_offset_;
    geom.gu_absmax_offset = gu_absmax_offset_;
    geom.dn_packed_offset = dn_packed_offset_;
    geom.dn_absmax_offset = dn_absmax_offset_;
    geom.slot_bytes = slot_bytes_;
    geom.compact_nested = compact_nested_;

    std::vector<aila::ExpertLayerSource> sources(num_layers_);
    for (int L = 0; L < num_layers_; ++L) {
        if (L < static_cast<int>(layer_weights_.size()) && layer_weights_[L].valid) {
            sources[L].gate_up = layer_weights_[L].gate_up;
            sources[L].down = layer_weights_[L].down;
            sources[L].valid = true;
        }
    }
    std::vector<std::string> paths;
    if (weights_) paths = weights_->shard_paths();
    reader_ = aila::CreateExpertReader(type, geom, paths, sources);
}

void Qwen4ExpExpertCache::configure_layer_nested_metadata(int layer) {
    if (!compact_nested_) return;
    if (layer < 0 || layer >= num_layers_ || !layer_weights_[layer].valid) {
        throw std::invalid_argument("Qwen4ExpExpertCache: invalid layer for nested metadata");
    }
    auto& lw = layer_weights_[layer];
    const auto valid_matrix = [](const Bnb4BitHostWeightRef& ref, size_t per_expert_count) {
        return ref.quant_state.nested && ref.absmax.dtype == DT_U8 &&
               ref.nested_absmax.valid() && ref.nested_absmax.dtype == DT_F32 &&
               ref.nested_quant_map.valid() && ref.nested_quant_map.dtype == DT_F32 &&
               ref.nested_quant_map.bytes >= 256 * sizeof(float) &&
               ref.quant_state.nested_blocksize > 0 &&
               per_expert_count % static_cast<size_t>(ref.quant_state.nested_blocksize) == 0;
    };
    if (!valid_matrix(lw.gate_up, gu_absmax_count_) ||
        !valid_matrix(lw.down, dn_absmax_count_)) {
        throw std::invalid_argument("Qwen4ExpExpertCache: nested codes require aligned u8 absmax and f32 metadata");
    }
    for (uint64_t key : slot_to_key_) {
        if (key != UINT64_MAX && static_cast<int>(key >> 32) == layer) {
            throw std::logic_error("Qwen4ExpExpertCache: cannot replace metadata for resident experts");
        }
    }
    const size_t gu_scales_bytes = lw.gate_up.nested_absmax.bytes;
    const size_t gu_map_bytes = lw.gate_up.nested_quant_map.bytes;
    const size_t dn_scales_bytes = lw.down.nested_absmax.bytes;
    const size_t dn_map_bytes = lw.down.nested_quant_map.bytes;
    const size_t total_bytes = gu_scales_bytes + gu_map_bytes + dn_scales_bytes + dn_map_bytes;
    std::vector<uint8_t> host(total_bytes);
    size_t offset = 0;
    auto append = [&](const HostTensorView& view) {
        std::memcpy(host.data() + offset, view.ptr, view.bytes);
        offset += view.bytes;
    };
    append(lw.gate_up.nested_absmax);
    append(lw.gate_up.nested_quant_map);
    append(lw.down.nested_absmax);
    append(lw.down.nested_quant_map);
    uint8_t* dev = static_cast<uint8_t*>(ctx_->alloc_device(total_bytes));
    try {
        ctx_->memcpy_h2d(dev, host.data(), total_bytes);
    } catch (...) {
        ctx_->free_device(dev);
        throw;
    }
    if (layer_meta_allocs_.size() <= static_cast<size_t>(layer)) {
        layer_meta_allocs_.resize(static_cast<size_t>(layer) + 1, nullptr);
    }
    if (layer_meta_allocs_[layer]) ctx_->free_device(layer_meta_allocs_[layer]);
    layer_meta_allocs_[layer] = dev;
    offset = 0;
    lw.gu_nested_absmax_dev = reinterpret_cast<const float*>(dev + offset);
    offset += gu_scales_bytes;
    lw.gu_nested_qmap_dev = reinterpret_cast<const float*>(dev + offset);
    offset += gu_map_bytes;
    lw.dn_nested_absmax_dev = reinterpret_cast<const float*>(dev + offset);
    offset += dn_scales_bytes;
    lw.dn_nested_qmap_dev = reinterpret_cast<const float*>(dev + offset);
}

void Qwen4ExpExpertCache::bind_slot_nested_metadata(int slot, int layer, int expert_id) {
    if (!compact_nested_) return;
    const auto& lw = layer_weights_.at(layer);
    if (!lw.gu_nested_absmax_dev || !lw.dn_nested_absmax_dev ||
        expert_id < 0 || expert_id >= num_experts_per_layer_) {
        throw std::runtime_error("Qwen4ExpExpertCache: missing nested metadata for expert");
    }
    auto& dev = slots_dev_.at(slot);
    const int gu_block = lw.gate_up.quant_state.nested_blocksize;
    const int dn_block = lw.down.quant_state.nested_blocksize;
    dev.gate_up_nested.codes = slot_dev_base_[slot] + gu_absmax_offset_;
    dev.gate_up_nested.nested_absmax = lw.gu_nested_absmax_dev +
        static_cast<size_t>(expert_id) * (gu_absmax_count_ / static_cast<size_t>(gu_block));
    dev.gate_up_nested.nested_quant_map = lw.gu_nested_qmap_dev;
    dev.gate_up_nested.offset = lw.gate_up.quant_state.nested_offset;
    dev.gate_up_nested.blocksize = gu_block;
    dev.down_nested.codes = slot_dev_base_[slot] + dn_absmax_offset_;
    dev.down_nested.nested_absmax = lw.dn_nested_absmax_dev +
        static_cast<size_t>(expert_id) * (dn_absmax_count_ / static_cast<size_t>(dn_block));
    dev.down_nested.nested_quant_map = lw.dn_nested_qmap_dev;
    dev.down_nested.offset = lw.down.quant_state.nested_offset;
    dev.down_nested.blocksize = dn_block;
}

int Qwen4ExpExpertCache::pick_victim_hslot() {
    std::unique_lock<std::mutex> lock;
    if (host_list_mutex_) lock = std::unique_lock<std::mutex>(*host_list_mutex_);
    auto is_available = [&](int h) {
        return !host_slot_reserved_[h] &&
               host_slot_records_[h].cpu_readers.load(std::memory_order_acquire) == 0;
    };
    // 1. Try unreserved empty slots first
    for (auto it = host_empty_slots_.rbegin(); it != host_empty_slots_.rend(); ++it) {
        int h = *it;
        if (h >= 0 && h < num_host_slots_ && is_available(h)) {
            host_empty_slots_.erase(std::next(it).base());
            return h;
        }
    }
    // 2. Try unreserved probationary slots whose DMA is already complete
    for (auto r_it = host_probationary_list_.rbegin(); r_it != host_probationary_list_.rend(); ++r_it) {
        int h = *r_it;
        if (is_available(h) && is_event_complete(host_slot_dma_events_[h])) {
            return h;
        }
    }
    // 3. Try unreserved protected slots whose DMA is already complete
    for (auto r_it = host_protected_list_.rbegin(); r_it != host_protected_list_.rend(); ++r_it) {
        int h = *r_it;
        if (is_available(h) && is_event_complete(host_slot_dma_events_[h])) {
            return h;
        }
    }
    // 4. If all unreserved candidates have in-flight DMAs, pick LRU unreserved slot and wait on its DMA
    for (auto r_it = host_probationary_list_.rbegin(); r_it != host_probationary_list_.rend(); ++r_it) {
        int h = *r_it;
        if (is_available(h)) {
            if (host_slot_dma_events_[h] != sycl::event{}) {
                host_slot_dma_events_[h].wait_and_throw();
                host_slot_dma_events_[h] = sycl::event{};
            }
            return h;
        }
    }
    for (auto r_it = host_protected_list_.rbegin(); r_it != host_protected_list_.rend(); ++r_it) {
        int h = *r_it;
        if (is_available(h)) {
            if (host_slot_dma_events_[h] != sycl::event{}) {
                host_slot_dma_events_[h].wait_and_throw();
                host_slot_dma_events_[h] = sycl::event{};
            }
            return h;
        }
    }
    return -1;
}

bool Qwen4ExpExpertCache::host_has_dma_in_flight(int layer, int expert_id) const {
    if (!initialized_ || !host_pool_enabled_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
    int hslot = host_layer_expert_to_slot_[expert_flat_index(layer, expert_id)];
    if (hslot < 0 || hslot >= num_host_slots_) return false;
    return !is_event_complete(host_slot_dma_events_[hslot]);
}

void Qwen4ExpExpertCache::touch_host_slot(int hslot) {
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return;
    std::unique_lock<std::mutex> lock;
    if (host_list_mutex_) lock = std::unique_lock<std::mutex>(*host_list_mutex_);
    const size_t max_protected = static_cast<size_t>(num_host_slots_ * protected_ratio_);
    if (!host_slot_is_protected_[hslot]) {
        if (max_protected > 0) {
            host_slot_is_protected_[hslot] = true;
            host_protected_list_.splice(host_protected_list_.begin(),
                                        host_probationary_list_, host_slot_iters_[hslot]);
            if (host_protected_list_.size() > max_protected && !host_protected_list_.empty()) {
                int demote_hslot = host_protected_list_.back();
                host_slot_is_protected_[demote_hslot] = false;
                host_probationary_list_.splice(host_probationary_list_.begin(),
                                               host_protected_list_, host_slot_iters_[demote_hslot]);
            }
        } else {
            host_probationary_list_.splice(host_probationary_list_.begin(),
                                           host_probationary_list_, host_slot_iters_[hslot]);
        }
    } else {
        host_protected_list_.splice(host_protected_list_.begin(),
                                    host_protected_list_, host_slot_iters_[hslot]);
    }
}

void Qwen4ExpExpertCache::add_batch_reader_hslot(int hslot) {
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return;
    if (!host_slot_in_batch_readers_[hslot]) {
        host_slot_in_batch_readers_[hslot] = true;
        batch_reserved_hslots_scratch_.push_back(hslot);
    }
    host_slot_reserved_[hslot] = true;
}

void Qwen4ExpExpertCache::ensure_prefetch_item_ready(PrefetchItem& item, int c) {
    if (item.ready.load(std::memory_order_acquire)) return;

    const auto t_start = std::chrono::steady_clock::now();
    const int parts = fill_parts_;
    uint8_t* dst = (item.hslot >= 0) ? host_slot_ptrs_[item.hslot]
                                     : prefetch_staging_ptr_ + static_cast<size_t>(c) * slot_bytes_;

    for (int p = 0; p < parts; ++p) {
        const uint64_t bit = uint64_t{1} << p;
        if (!(item.claimed_parts.load(std::memory_order_relaxed) & bit)) {
            const uint64_t old = item.claimed_parts.fetch_or(bit, std::memory_order_acq_rel);
            if (!(old & bit)) {
                load_expert_to_staging(item.layer, item.expert_id, dst, p, parts, item.expected_generation);
                if (item.parts_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    item.ready.store(true, std::memory_order_release);
                    if (item.hslot >= 0) {
                        if (host_slot_records_[item.hslot].generation.load(std::memory_order_acquire) == item.expected_generation) {
                            host_slot_state_[item.hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
                        }
                    }
                }
            }
        }
    }

    for (int spin = 0; spin < 500 && !item.ready.load(std::memory_order_acquire); ++spin) {
        _mm_pause();
    }
    while (!item.ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t_start).count());
    stats_.prefetch_wait_ns += wait_ns;
    stats_.demand_exposed_wait_ns += wait_ns;
}

void Qwen4ExpExpertCache::cancel_prefetch() {
    if (!prefetch_in_flight_ && prefetch_items_.empty()) return;

    if (prefetch_in_flight_ && fill_pool_) {
        fill_pool_->cancel();
    }
    if (reader_) {
        reader_->cancel();
    }

    if (host_pool_enabled_) {
        for (auto& item : prefetch_items_) {
            if (item.used) continue; // Already acquired and counted in prefetch_useful_bytes
            if (item.hslot >= 0 && item.hslot < num_host_slots_) {
                const int h = item.hslot;
                const bool in_batch = host_slot_in_batch_readers_[h];
                if (!in_batch) {
                    host_slot_reserved_[h] = false;
                }
                if (!item.ready.load(std::memory_order_acquire)) {
                    const int rem = item.parts_remaining.load(std::memory_order_relaxed);
                    if (fill_parts_ > 0) {
                        const int completed = fill_parts_ - rem;
                        if (completed > 0) {
                            stats_.prefetch_wasted_bytes += (static_cast<uint64_t>(completed) * slot_bytes_) / fill_parts_;
                        }
                        if (rem > 0) {
                            stats_.prefetch_cancelled_bytes += (static_cast<uint64_t>(rem) * slot_bytes_) / fill_parts_;
                        }
                    }
                    const int flat = expert_flat_index(item.layer, item.expert_id);
                    if (host_layer_expert_to_slot_[flat] == h) {
                        host_layer_expert_to_slot_[flat] = -1;
                    }
                    host_slot_to_key_[h] = UINT64_MAX;
                    host_slot_records_[h].key = UINT64_MAX;
                    host_slot_state_[h].store(static_cast<uint8_t>(ExpertSlotState::Empty), std::memory_order_release);
                    if (host_slot_is_protected_[h]) {
                        host_protected_list_.erase(host_slot_iters_[h]);
                    } else {
                        host_probationary_list_.erase(host_slot_iters_[h]);
                    }
                    host_slot_is_protected_[h] = false;
                    host_empty_slots_.push_back(h);
                } else {
                    stats_.prefetch_wasted_bytes += slot_bytes_;
                    if (host_slot_records_[h].generation.load(std::memory_order_acquire) == item.expected_generation) {
                        host_slot_state_[h].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
                    }
                }
            } else if (!item.ready.load(std::memory_order_acquire)) {
                const int rem = item.parts_remaining.load(std::memory_order_relaxed);
                if (fill_parts_ > 0) {
                    const int completed = fill_parts_ - rem;
                    if (completed > 0) {
                        stats_.prefetch_wasted_bytes += (static_cast<uint64_t>(completed) * slot_bytes_) / fill_parts_;
                    }
                    if (rem > 0) {
                        stats_.prefetch_cancelled_bytes += (static_cast<uint64_t>(rem) * slot_bytes_) / fill_parts_;
                    }
                }
            } else {
                stats_.prefetch_wasted_bytes += slot_bytes_;
            }
        }
    } else {
        for (auto& item : prefetch_items_) {
            if (item.used) continue;
            if (!item.ready.load(std::memory_order_acquire)) {
                const int rem = item.parts_remaining.load(std::memory_order_relaxed);
                if (fill_parts_ > 0) {
                    const int completed = fill_parts_ - rem;
                    if (completed > 0) {
                        stats_.prefetch_wasted_bytes += (static_cast<uint64_t>(completed) * slot_bytes_) / fill_parts_;
                    }
                    if (rem > 0) {
                        stats_.prefetch_cancelled_bytes += (static_cast<uint64_t>(rem) * slot_bytes_) / fill_parts_;
                    }
                }
            } else {
                stats_.prefetch_wasted_bytes += slot_bytes_;
            }
        }
    }

    prefetch_items_.clear();
    prefetch_in_flight_ = false;
    if (reader_) {
        reader_->reset_cancelled();
    }
}

sycl::event Qwen4ExpExpertCache::acquire(int layer,
                                         const int* expert_ids,
                                         int count,
                                         int* out_slots) {
    return acquire_impl(layer, expert_ids, count, out_slots, true);
}

GpuReadyLease::~GpuReadyLease() {
    if (cache_) cache_->ReleaseReady(*this, {}, true);
}

GpuReadyLease::GpuReadyLease(GpuReadyLease&& other) noexcept
    : layer(other.layer), ready_count(other.ready_count), cold_count(other.cold_count),
      ready_slots(other.ready_slots), ready_ranks(other.ready_ranks), ready_ids(other.ready_ids),
      ready_generations(other.ready_generations), cold_ids(other.cold_ids), cold_ranks(other.cold_ranks),
      cache_(other.cache_) {
    other.cache_ = nullptr;
}

void GpuReadyLease::Finish(const sycl::event& last_reader) {
    if (cache_) cache_->ReleaseReady(*this, last_reader, last_reader == sycl::event{});
}

bool Qwen4ExpExpertCache::ValidateReadyBinding(int layer, int expert, int slot, uint64_t generation) const {
    if (!is_initialized() || layer < 0 || layer >= num_layers_ || expert < 0 ||
        expert >= num_experts_per_layer_ || slot < 0 || slot >= num_slots_) return false;
    const auto& record = gpu_slot_records_[slot];
    return layer_expert_to_slot_[expert_flat_index(layer, expert)] == slot &&
        record.key == ((uint64_t(layer) << 32) | uint32_t(expert)) &&
        record.generation == generation && get_slot_state(slot) == ExpertSlotState::Ready;
}

GpuReadyLease Qwen4ExpExpertCache::BindReady(int layer, const int* ids, int count) {
    return BindReadyImpl(layer, ids, count, nullptr);
}

void Qwen4ExpExpertCache::ConfigureHotset() {
    const auto profile=aila::env::read_string("AILA_Q4E_HOTSET_PROFILE","");
    hotset_export_=aila::env::read_string("AILA_Q4E_HOTSET_EXPORT","");
    if(profile.empty() && hotset_export_.empty())return;
    hotset_policy_.Init(num_layers_,num_experts_per_layer_);
    // Layout/quantization plus sampled source bytes, computed once at load. This
    // is an expert-identity guard, not a cryptographic hash of the entire model.
    uint64_t hash=1469598103934665603ULL;
    auto bytes=[&](const uint8_t* p,size_t n){for(size_t i=0;i<n;++i){hash^=p[i];hash*=1099511628211ULL;}};
    for(const auto& lw:layer_weights_) for(const auto* w:{&lw.gate_up,&lw.down}) {
        for(const auto* view:{&w->packed_weight,&w->absmax,&w->quant_map,&w->nested_absmax,&w->nested_quant_map}) {
            const uint64_t n=view->bytes;bytes(reinterpret_cast<const uint8_t*>(&n),sizeof(n));
            if(view->valid()) {const size_t sample=std::min<size_t>(4096,n);
                bytes(view->ptr,sample); if(n>sample)bytes(view->ptr+n-sample,sample);}
        }
        bytes(reinterpret_cast<const uint8_t*>(&w->quant_state.nested_offset),sizeof(w->quant_state.nested_offset));
        bytes(reinterpret_cast<const uint8_t*>(&w->quant_state.nested_blocksize),sizeof(w->quant_state.nested_blocksize));
    }
    hotset_identity_="nf4-original-expert-order-"+std::to_string(num_layers_)+"x"+std::to_string(num_experts_per_layer_)+
        "-"+std::to_string(hidden_size_)+"x"+std::to_string(intermediate_size_)+"-b"+std::to_string(blocksize_)+
        "-nested"+std::to_string(int(compact_nested_))+"-sample-fnv64-"+std::to_string(hash);
    if(profile.empty())return;
    hotset_policy_.Load(profile,hotset_identity_); // Fail explicit experiments on a wrong profile.
    hotset_interval_=std::clamp(aila::env::read_int_raw("AILA_Q4E_HOTSET_INTERVAL",4),0,1024);
    hotset_swaps_=std::clamp(aila::env::read_int_raw("AILA_Q4E_HOTSET_SWAPS",96),0,512);
    hotset_borrow_=aila::env::read_flag("AILA_Q4E_HOTSET_BORROW",false);
    hotset_prompt_=aila::env::read_flag("AILA_Q4E_HOTSET_PROMPT",false);
    hotset_prompt_prefill_=aila::env::read_flag("AILA_Q4E_HOTSET_PROMPT_PREFILL",false);
    hotset_adopt_ready_=aila::env::read_flag("AILA_Q4E_HOTSET_ADOPT_READY",false);
    hotset_adopt_limit_=std::clamp(aila::env::read_int_raw("AILA_Q4E_HOTSET_ADOPT_LIMIT",96),0,4096);
    hotset_prompt_=hotset_prompt_ || hotset_prompt_prefill_ || hotset_adopt_ready_;
    hotset_prompt_swaps_=std::clamp(aila::env::read_int_raw("AILA_Q4E_HOTSET_PROMPT_SWAPS",1024),0,num_slots_-1);
    hotset_ring_=aila::env::read_flag("AILA_Q4E_HOTSET_RING",true);
    hotset_cold_policy_=std::clamp(aila::env::read_int_raw("AILA_Q4E_HOTSET_COLD_QUEUE",hotset_ring_?2:0),0,2);
    const int staging=std::min(hotset_borrow_?16:num_experts_per_layer_,num_slots_-1);
    const int capacity=std::clamp(aila::env::read_int("AILA_Q4E_HOTSET_SLOTS",num_slots_-staging),1,num_slots_-staging);
    hotset_keys_.assign(hotset_policy_.ranking.begin(),hotset_policy_.ranking.begin()+capacity);
    hotset_pending_.reserve(hotset_swaps_); adaptive_hotset_=true;
    InstallHotset(hotset_keys_);
    AILA_LOG_INFO("[Q4E Hotset] profile=%s identity=%s slots=%d interval=%d swaps=%d decay=0.7 min_usage=2 gain=1.5 borrow=%d",
        profile.c_str(),hotset_identity_.c_str(),capacity,hotset_interval_,hotset_swaps_,int(hotset_borrow_));
}

void Qwen4ExpExpertCache::InstallHotset(const std::vector<int>& keys) {
    // Validate before any mutation. Acquire can never evict an already installed
    // member, even if loading a later layer wraps the dynamic LRU.
    std::vector<bool> seen(size_t(num_layers_)*num_experts_per_layer_,false);
    if(num_pinned_slots_)throw std::invalid_argument("Profile hotset cannot mix with legacy pinning");
    if(keys.empty() || keys.size()>=size_t(num_slots_))throw std::invalid_argument("Hotset leaves no staging");
    for(int k:keys) {if(k<0 || k>=int(seen.size()) || seen[k])throw std::invalid_argument("Invalid hotset key");seen[k]=true;}
    PublishHotsetPending(true);ThawReady();ctx_->synchronize();copy_q_->wait_and_throw();
    const auto start=std::chrono::steady_clock::now();const auto bytes0=stats_.total_transferred_bytes;
    const auto hits0=stats_.hit_count, misses0=stats_.miss_count, calls0=stats_.acquire_calls, batches0=stats_.demand_microbatches;
    if(cache_trace_){cache_trace_->Row("epoch_load_begin")<<",\"forward\":"<<trace_forward_id_;cache_trace_->End();}
    frozen_=true; frozen_count_=0;
    // Preserve target weights that survived the borrowed prefill region. Protect
    // all of them before filling any layer, so refill cannot evict later targets.
    std::vector<int> preserved;
    for(int k:keys) {const int s=layer_expert_to_slot_[k];if(s>=0) {
        frozen_slots_[s]=true;++frozen_count_;preserved.push_back(s);gpu_slot_records_[s].state=ExpertSlotState::Ready;}}
    if(cache_trace_ && !preserved.empty()){auto& out=cache_trace_->Row("epoch_protect");out<<",\"slots\":";Qwen4ExpCacheTrace::Ints(out,preserved.data(),int(preserved.size()));cache_trace_->End();}
    for(int l=0;l<num_layers_;++l) {
        std::vector<int> ids,slots;for(int k:keys)if(k/num_experts_per_layer_==l && layer_expert_to_slot_[k]<0)ids.push_back(k%num_experts_per_layer_);
        // Chunk by available staging while progressively protecting the target set.
        for(size_t offset=0;offset<ids.size();) {
            const int n=std::min<int>(int(ids.size()-offset),num_slots_-frozen_count_);slots.resize(n);
            acquire_impl(l,ids.data()+offset,n,slots.data(),false).wait_and_throw();
            std::vector<int> added;
            for(int s:slots){if(!frozen_slots_[s]){frozen_slots_[s]=true;++frozen_count_;added.push_back(s);}gpu_slot_records_[s].state=ExpertSlotState::Ready;}
            if(cache_trace_ && !added.empty()){auto& out=cache_trace_->Row("epoch_protect");out<<",\"slots\":";Qwen4ExpCacheTrace::Ints(out,added.data(),int(added.size()));cache_trace_->End();}
            offset+=n;
        }
    }
    for(int s=0;s<num_slots_;++s)if(!frozen_slots_[s])DiscardTransient(&s,1);
    frozen_descriptors_published_=false;std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);
    stats_.epoch_refill_bytes+=stats_.total_transferred_bytes-bytes0;
    stats_.total_transferred_bytes=bytes0; // Epoch traffic is reported separately from demand.
    // Maintenance is not a routed demand. In particular, profile loading must
    // not depress the hit-rate that controls speculative prefetch gating.
    stats_.hit_count=hits0;stats_.miss_count=misses0;stats_.acquire_calls=calls0;stats_.demand_microbatches=batches0;
    stats_.epoch_refill_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    ++stats_.epoch_publishes;
    RefreshHotsetColdSlots();
    if(cache_trace_){cache_trace_->Row("epoch_load_end");cache_trace_->End();}
}

void Qwen4ExpExpertCache::RefreshHotsetColdSlots() {
    hotset_cold_slots_.clear();hotset_cold_cursor_=0;
    hotset_cold_lru_.clear();hotset_cold_iters_.resize(num_slots_);hotset_cold_member_.assign(num_slots_,false);
    if(adaptive_hotset_ && frozen_ && hotset_cold_policy_==1)
        for(int s=0;s<num_slots_;++s)if(!frozen_slots_[s])hotset_cold_slots_.push_back(s);
    if(adaptive_hotset_ && frozen_ && hotset_cold_policy_==2)
        for(int s:lru_list_)if(!frozen_slots_[s]) {
            hotset_cold_lru_.push_back(s);hotset_cold_iters_[s]=std::prev(hotset_cold_lru_.end());hotset_cold_member_[s]=true;
        }
}
void Qwen4ExpExpertCache::TouchHotsetColdSlot(int slot,bool front) {
    if(!hotset_cold_lru_.empty() && hotset_cold_member_[slot])
        hotset_cold_lru_.splice(front?hotset_cold_lru_.begin():hotset_cold_lru_.end(),hotset_cold_lru_,hotset_cold_iters_[slot]);
}

void Qwen4ExpExpertCache::PublishHotsetPending(bool wait) {
    if(hotset_pending_.empty())return;
    for(const auto& p:hotset_pending_)if(!wait && !is_event_complete(p.ready))return;
    for(auto& p:hotset_pending_) {
        p.ready.wait_and_throw();
        const int l=p.key/num_experts_per_layer_,e=p.key%num_experts_per_layer_;
        const int transient=layer_expert_to_slot_[p.key];
        if(transient>=0 && transient!=p.slot)DiscardTransient(&transient,1);
        bind_slot_nested_metadata(p.slot,l,e);
        const uint64_t key=(uint64_t(l)<<32)|uint32_t(e);
        layer_expert_to_slot_[p.key]=p.slot;slot_to_key_[p.slot]=key;
        auto& r=gpu_slot_records_[p.slot];r.key=key;r.state=ExpertSlotState::Ready;
        if(cache_trace_){cache_trace_->Row("epoch_publish")<<",\"slot\":"<<p.slot<<",\"key\":"<<p.key;cache_trace_->End();}
    }
    hotset_pending_.clear();frozen_descriptors_published_=false;
    std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);++stats_.epoch_publishes;
}

void Qwen4ExpExpertCache::RebalanceHotset() {
    if(!frozen_ || !hotset_pending_.empty() || hotset_swaps_==0)return;
    std::vector<int> resident(layer_expert_to_slot_.size(),-1);
    for(size_t k=0;k<resident.size();++k) {const int s=layer_expert_to_slot_[k];if(s>=0 && frozen_slots_[s])resident[k]=s;}
    const auto swaps=hotset_policy_.Select(resident,hotset_swaps_);hotset_policy_.Decay();
    if(swaps.empty())return;
    for(const auto& r:gpu_slot_records_)if(r.direct_leases)throw std::logic_error("Epoch switch with active readers");
    // Slot-specific consumer fences protect payloads; no global compute queue
    // drain is needed. Retire before DMA, publish only at a later forward boundary.
    const auto begin=std::chrono::steady_clock::now();
    cancel_prefetch();
    frozen_descriptors_published_=false;std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);
    try {
    for(const auto& sw:swaps) {
        const int l=sw.incoming/num_experts_per_layer_,e=sw.incoming%num_experts_per_layer_,s=resident[sw.outgoing];
        prepare_host(l,e);auto lease=acquire_host_lease(l,e);
        if(!lease.is_valid())continue;
        auto& r=gpu_slot_records_[s];
        if(r.consumer_fence!=sycl::event{})r.consumer_fence.wait_and_throw();
        if(r.dma_event!=sycl::event{})r.dma_event.wait_and_throw();
        layer_expert_to_slot_[sw.outgoing]=-1;slot_to_key_[s]=UINT64_MAX;r.key=UINT64_MAX;++r.generation;
        const auto ready=copy_q_->memcpy(slot_dev_base_[s],host_slot_ptrs_[lease.hslot()],slot_bytes_);
        r.dma_event=ready;r.state=ExpertSlotState::Copying;
        hotset_pending_.push_back({sw.incoming,s,std::move(lease),ready});
        auto old=std::find(hotset_keys_.begin(),hotset_keys_.end(),sw.outgoing);if(old!=hotset_keys_.end())*old=sw.incoming;
        stats_.epoch_refill_bytes+=slot_bytes_;++stats_.epoch_swaps;
        if(cache_trace_){cache_trace_->Row("epoch_retire")<<",\"slot\":"<<s<<",\"key\":"<<sw.incoming<<",\"old_key\":"<<sw.outgoing;cache_trace_->End();}
    }
    } catch(...) {
        faulted_=true;
        try {copy_q_->wait();} catch(...) {}
        hotset_pending_.clear();
        throw;
    }
    stats_.epoch_refill_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
}

void Qwen4ExpExpertCache::BeginHotsetForward(int position,int rows,int prefill_remaining) {
    hotset_explicit_prefill_=prefill_remaining>=0;
    hotset_adopt_count_=0;
    if(!adaptive_hotset_)return;
    PublishHotsetPending(false);
    if(hotset_prompt_ && (position==0 || ((rows>1 || hotset_explicit_prefill_) && !hotset_policy_.collecting_prompt))) {
        hotset_policy_.BeginPrompt();hotset_tokens_=0;
        hotset_prefill_keys_=hotset_keys_;
    }
    if(hotset_prompt_ && rows==1 && position>0 && !hotset_explicit_prefill_ && hotset_policy_.collecting_prompt)
        HandoffPromptHotset();
    if(hotset_borrow_ && (rows>1 || hotset_explicit_prefill_) && !hotset_borrowed_) {
        PublishHotsetPending(true);ctx_->synchronize();
        const int retain=std::min<int>(int(hotset_keys_.size()),std::max(0,num_slots_-num_experts_per_layer_));
        std::vector<int> lent;
        for(size_t i=retain;i<hotset_keys_.size();++i) {const int s=layer_expert_to_slot_[hotset_keys_[i]];
            if(s>=0 && frozen_slots_[s]){frozen_slots_[s]=false;--frozen_count_;lent.push_back(s);}}
        if(cache_trace_){auto& out=cache_trace_->Row("epoch_borrow");out<<",\"slots\":";Qwen4ExpCacheTrace::Ints(out,lent.data(),int(lent.size()));cache_trace_->End();}
        frozen_descriptors_published_=false;std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);
        hotset_borrowed_=true;
        RefreshHotsetColdSlots();
    } else if(rows==1 && !hotset_explicit_prefill_ && hotset_borrowed_) {InstallHotset(hotset_keys_);hotset_borrowed_=false;}
    if(position==0)hotset_tokens_=0;
}
void Qwen4ExpExpertCache::ObserveHotsetRoute(int layer,const int* ids,int rows,int topk) {
    if(hotset_policy_.layers)hotset_policy_.Observe(layer,ids,rows,topk,hotset_explicit_prefill_);
    if(hotset_prompt_ && hotset_policy_.collecting_prompt)hotset_policy_.ObservePrompt(layer,ids,rows,topk);
}

void Qwen4ExpExpertCache::HandoffPromptHotset() {
    PublishHotsetPending(true);
    const float similarity=hotset_policy_.SeedFromPrompt();
    std::vector<int> target(hotset_policy_.usage.size(),-1);
    // Logical tier includes lent/missing slots. Select before restore so old
    // targets that lose the prompt election are never needlessly reloaded.
    for(int k:hotset_keys_)target[k]=0;
    // A repeated prompt already has decode evidence. Do not churn low-score
    // tail entries merely because a no-hysteresis handoff can rank them again.
    const int budget=int(hotset_prompt_swaps_*std::clamp(1.0f-similarity,0.0f,1.0f));
    const auto swaps=hotset_policy_.Select(target,budget,true);
    for(const auto& sw:swaps) {
        auto old=std::find(hotset_keys_.begin(),hotset_keys_.end(),sw.outgoing);
        *old=sw.incoming;
    }
    // Borrow the least valuable tail on the next prefill, rather than retaining
    // inherited positions from the original offline profile indefinitely.
    std::stable_sort(hotset_keys_.begin(),hotset_keys_.end(),[&](int a,int b){
        return hotset_policy_.usage[a]>hotset_policy_.usage[b];
    });
    const auto bytes0=stats_.epoch_refill_bytes,ns0=stats_.epoch_refill_ns;
    if(hotset_borrowed_ || !swaps.empty())InstallHotset(hotset_keys_);
    hotset_borrowed_=false;
    AILA_LOG_INFO("[Q4E PromptHotset] rows=%d similarity=%.6f budget=%d swaps=%d refill_bytes=%llu refill_ns=%llu",
        hotset_policy_.prompt_rows,similarity,budget,int(swaps.size()),
        (unsigned long long)(stats_.epoch_refill_bytes-bytes0),(unsigned long long)(stats_.epoch_refill_ns-ns0));
    if(cache_trace_) {auto& out=cache_trace_->Row("prompt_handoff");
        out<<",\"rows\":"<<hotset_policy_.prompt_rows<<",\"swaps\":"<<swaps.size()<<",\"keys\":";
        Qwen4ExpCacheTrace::Ints(out,hotset_keys_.data(),int(hotset_keys_.size()));cache_trace_->End();}
}
void Qwen4ExpExpertCache::RebalancePromptPrefill() {
    const int retain=hotset_borrow_?std::min<int>(int(hotset_keys_.size()),num_slots_-num_experts_per_layer_):int(hotset_keys_.size());
    if(retain<=0 || hotset_policy_.prompt_rows==0 || hotset_prompt_swaps_==0)return;
    // Separate temporary prefill residency from the learned decode tier. A long
    // prompt may need different experts in different chunks, without destroying
    // the decode prior we want to reuse for a repeated request.
    std::vector<float> scores(hotset_policy_.usage.size(),0);
    for(int l=0;l<num_layers_;++l)if(hotset_policy_.prompt_mass[l]>0)
        for(int e=0;e<num_experts_per_layer_;++e) {const int k=l*num_experts_per_layer_+e;
            scores[k]=hotset_policy_.prompt_usage[k]/hotset_policy_.prompt_mass[l];}
    std::vector<int> target(scores.size(),-1);for(int k:hotset_prefill_keys_)target[k]=0;
    const auto swaps=hotset_policy_.Select(target,hotset_prompt_swaps_,true,&scores);
    for(const auto& sw:swaps)*std::find(hotset_prefill_keys_.begin(),hotset_prefill_keys_.end(),sw.outgoing)=sw.incoming;
    std::stable_sort(hotset_prefill_keys_.begin(),hotset_prefill_keys_.end(),[&](int a,int b){return scores[a]>scores[b];});
    std::vector<int> keys(hotset_prefill_keys_.begin(),hotset_prefill_keys_.begin()+retain);
    const auto bytes0=stats_.epoch_refill_bytes,ns0=stats_.epoch_refill_ns;
    InstallHotset(keys);hotset_borrowed_=true;
    AILA_LOG_INFO("[Q4E PrefillHotset] rows=%d swaps=%d slots=%d refill_bytes=%llu refill_ns=%llu",
        hotset_policy_.prompt_rows,int(swaps.size()),retain,
        (unsigned long long)(stats_.epoch_refill_bytes-bytes0),(unsigned long long)(stats_.epoch_refill_ns-ns0));
    if(cache_trace_) {auto& out=cache_trace_->Row("prefill_handoff");
        out<<",\"rows\":"<<hotset_policy_.prompt_rows<<",\"swaps\":"<<swaps.size()<<",\"keys\":";
        Qwen4ExpCacheTrace::Ints(out,keys.data(),int(keys.size()));cache_trace_->End();}
}
void Qwen4ExpExpertCache::EndHotsetForward(int rows,int prefill_remaining) {
    // Avoid rebuilding a multi-GB tier for the small checkpoint/logits tail.
    // 32 remaining rows is an experimental amortization guard, not a kernel dispatch rule.
    if(adaptive_hotset_ && hotset_prompt_prefill_ && hotset_policy_.collecting_prompt && rows>1 && prefill_remaining>=32)
        RebalancePromptPrefill();
    if(adaptive_hotset_ && rows==1 && prefill_remaining<0 && hotset_interval_ && ++hotset_tokens_%hotset_interval_==0)RebalanceHotset();
}

bool Qwen4ExpExpertCache::IsDirectReady(int layer, int expert) const {
    if (!is_initialized() || layer < 0 || layer >= num_layers_ || expert < 0 || expert >= num_experts_per_layer_) return false;
    const int slot = layer_expert_to_slot_[expert_flat_index(layer, expert)];
    return slot >= 0 && (!frozen_ || frozen_slots_[slot]) &&
        ValidateReadyBinding(layer, expert, slot, gpu_slot_records_[slot].generation);
}

uint64_t Qwen4ExpExpertCache::FrozenIdentity() const {
    uint64_t hash = 1469598103934665603ULL;
    // Hash sorted logical keys, independent of physical slot placement.
    for (int l = 0; l < num_layers_; ++l) for (int e = 0; e < num_experts_per_layer_; ++e) {
        const int s = layer_expert_to_slot_[expert_flat_index(l,e)];
        if (s >= 0 && frozen_slots_[s]) { hash ^= uint64_t(l) * num_experts_per_layer_ + e; hash *= 1099511628211ULL; }
    }
    return hash;
}

void Qwen4ExpExpertCache::FreezeReady(int staging_slots) {
    if (!is_initialized() || staging_slots < 1 || staging_slots >= num_slots_ || num_pinned_slots_)
        throw std::invalid_argument("Frozen residency requires transient capacity and no legacy pinning");
    if (frozen_) return;
    for (const auto& r : gpu_slot_records_) if (r.direct_leases)
        throw std::logic_error("Cannot freeze with active direct readers");
    ctx_->synchronize();
    copy_q_->wait_and_throw();
    frozen_count_ = 0;
    // The original MRU order contains only previously observed demand.
    for (int s : lru_list_) {
        if (frozen_count_ >= num_slots_ - staging_slots) break;
        if (slot_to_key_[s] != UINT64_MAX && get_slot_state(s) == ExpertSlotState::Ready) {
            frozen_slots_[s] = true; ++frozen_count_;
        }
    }
    frozen_ = true;
    if(cache_trace_) {
        auto& out=cache_trace_->Row("freeze");out<<",\"slots\":[";
        bool first=true;for(int s=0;s<num_slots_;++s)if(frozen_slots_[s]){if(!first)out<<',';out<<s;first=false;}
        out<<']';cache_trace_->End();
    }
    frozen_descriptors_published_=false;
    std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);
    for (int s = 0; s < num_slots_; ++s) if (!frozen_slots_[s]) DiscardTransient(&s,1);
    AILA_LOG_INFO("[Q4E Frozen] slots=%d staging=%d identity=%llu", frozen_count_, num_slots_-frozen_count_,
        static_cast<unsigned long long>(FrozenIdentity()));
}

void Qwen4ExpExpertCache::ThawReady() {
    hotset_cold_slots_.clear();hotset_cold_cursor_=0;
    hotset_cold_lru_.clear();
    if (!frozen_) return;
    for (const auto& r : gpu_slot_records_) if (r.direct_leases)
        throw std::logic_error("Cannot thaw with active direct readers");
    ctx_->synchronize(); copy_q_->wait_and_throw();
    std::fill(frozen_slots_.begin(), frozen_slots_.end(), false);
    frozen_ = false; frozen_count_ = 0;
    if(cache_trace_){cache_trace_->Row("thaw");cache_trace_->End();}
}

void Qwen4ExpExpertCache::RetirePrefillSlots(const int* slots, int count) {
    if (hotset_adopt_ready_ && adaptive_hotset_ && frozen_ &&
        hotset_policy_.collecting_prompt && hotset_pending_.empty()) {
        const auto begin=std::chrono::steady_clock::now();
        for (int i=0;i<count && hotset_adopt_count_<hotset_adopt_limit_;++i) {
            const int s=slots[i];
            if(s<0 || s>=num_slots_)throw std::invalid_argument("Invalid prefill retirement slot");
            const auto key=slot_to_key_[s];
            auto& incoming=gpu_slot_records_[s];
            if(frozen_slots_[s] || key==UINT64_MAX || incoming.direct_leases || slot_reserved_[s])continue;
            const int layer=int(key>>32),expert=int(uint32_t(key));
            const int flat=expert_flat_index(layer,expert);
            if(!ValidateReadyBinding(layer,expert,s,incoming.generation))continue;
            // [HYPOTHESIS] Same-layer prompt frequency/hysteresis, restricted to
            // payloads already paid for by demand. No speculative refill or wait.
            const auto& scores=hotset_policy_.prompt_usage;
            if(scores[flat]<2.0f)continue;
            int victim=-1,old_key=-1;
            for(int e=0;e<num_experts_per_layer_;++e) {
                const int k=expert_flat_index(layer,e),v=layer_expert_to_slot_[k];
                if(v<0 || !frozen_slots_[v] || gpu_slot_records_[v].direct_leases || slot_reserved_[v])continue;
                if(victim<0 || scores[k]<scores[old_key] || (scores[k]==scores[old_key] && k<old_key)) {
                    victim=v;old_key=k;
                }
            }
            if(victim<0 || scores[flat]-scores[old_key]<1.5f)continue;
            auto target=std::find(hotset_keys_.begin(),hotset_keys_.end(),old_key);
            if(target==hotset_keys_.end() || std::find(hotset_keys_.begin(),hotset_keys_.end(),flat)!=hotset_keys_.end())continue;
            if(std::find(hotset_prefill_keys_.begin(),hotset_prefill_keys_.end(),flat)!=hotset_prefill_keys_.end())continue;
            // Swap tier ownership only. Both payloads, generations and outstanding
            // reader/DMA fences survive; demoted reuse follows acquire's fences.
            *target=flat;
            auto temporary=std::find(hotset_prefill_keys_.begin(),hotset_prefill_keys_.end(),old_key);
            if(temporary!=hotset_prefill_keys_.end())*temporary=flat;
            frozen_slots_[s]=true;frozen_slots_[victim]=false;
            incoming.state=ExpertSlotState::Ready;
            for(int& cold:hotset_cold_slots_)if(cold==s){cold=victim;break;}
            if(!hotset_cold_lru_.empty() && hotset_cold_member_[s]) {
                const auto it=hotset_cold_iters_[s];*it=victim;
                hotset_cold_iters_[victim]=it;hotset_cold_member_[s]=false;hotset_cold_member_[victim]=true;
            }
            frozen_descriptors_published_=false;frozen_indices_published_[layer]=false;
            ++hotset_adopt_count_;++stats_.ready_adoptions;
            if(cache_trace_) {
                cache_trace_->Row("ready_adopt")<<",\"slot\":"<<s<<",\"victim\":"<<victim
                    <<",\"key\":"<<flat<<",\"old_key\":"<<old_key<<",\"generation\":"<<incoming.generation;
                cache_trace_->End();
            }
        }
        stats_.ready_adoption_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
    }
    DiscardTransient(slots,count);
}

void Qwen4ExpExpertCache::DiscardTransient(const int* slots, int count) {
    if (!frozen_) return;
    for (int i=0;i<count;++i) {
        const int s=slots[i];
        if (s<0 || s>=num_slots_) throw std::invalid_argument("Invalid transient slot");
        if (frozen_slots_[s]) continue;
        auto& record=gpu_slot_records_[s];
        if (record.direct_leases) throw std::logic_error("Cannot discard a leased slot");
        const auto key=slot_to_key_[s];
        if(cache_trace_ && key!=UINT64_MAX) {
            cache_trace_->Row("discard")<<",\"slot\":"<<s;cache_trace_->End();
        }
        if (key!=UINT64_MAX) layer_expert_to_slot_[expert_flat_index(int(key>>32),int(uint32_t(key)))]=-1;
        slot_to_key_[s]=UINT64_MAX; record.key=UINT64_MAX; ++record.generation;
        // Keep DMA and consumer fences: next reuse still waits for submitted readers.
        record.state=ExpertSlotState::Empty;
    }
}

void Qwen4ExpExpertCache::TraceColdCpu(int layer,const int* ids,const int* ranks,int count) {
    stats_.cpu_cold_entries+=count;
    if(!cache_trace_ || !count)return;
    auto& out=cache_trace_->Row("cold_cpu");out<<",\"layer\":"<<layer<<",\"ids\":";
    Qwen4ExpCacheTrace::Ints(out,ids,count);out<<",\"ranks\":";Qwen4ExpCacheTrace::Ints(out,ranks,count);cache_trace_->End();
}

void Qwen4ExpExpertCache::TraceDeviceEvent(const char* label,int layer,sycl::event event) {
    if(!cache_trace_ || event==sycl::event{} || !ctx_->queue().has_property<sycl::property::queue::enable_profiling>())return;
    event.wait_and_throw();
    const auto start=event.get_profiling_info<sycl::info::event_profiling::command_start>();
    const auto end=event.get_profiling_info<sycl::info::event_profiling::command_end>();
    cache_trace_->Row("device")<<",\"forward\":"<<trace_forward_id_<<",\"layer\":"<<layer
        <<",\"label\":\""<<label<<"\",\"clock\":\"sycl_event\",\"start\":"<<start<<",\"end\":"<<end;
    cache_trace_->End();
}

GpuReadyLease Qwen4ExpExpertCache::BindReadyImpl(int layer, const int* ids, int count, const int* selected_slots) {
    if (!is_initialized() || !ids || count < 1 || count > GpuReadyLease::kMaxEntries ||
        layer < 0 || layer >= num_layers_)
        throw std::invalid_argument("Invalid direct resident route");
    for (int i = 0; i < count; ++i)
        if (ids[i] < 0 || ids[i] >= num_experts_per_layer_)
            throw std::invalid_argument("Invalid direct resident expert");
    trace_demand_route(layer, ids, count);
    GpuReadyLease lease;
    lease.layer = layer;
    lease.cache_ = this;
    std::array<int, GpuReadyLease::kMaxEntries> initial_slots{}, direct_slots{};
    direct_slots.fill(-1);
    for (int i = 0; i < count; ++i) {
        const int slot = layer_expert_to_slot_[expert_flat_index(layer, ids[i])];
        initial_slots[i] = slot;
        if (slot >= 0) {
            ++stats_.resident_route_entries;
            // Exactly the original full-route first-occurrence MRU order,
            // including Copying hits that still need the cold transaction.
            bool first = true;
            for (int j = 0; j < i; ++j) if (ids[j] == ids[i]) first = false;
            if (first && !slot_pinned_[slot])
                lru_list_.splice(lru_list_.begin(), lru_list_, slot_iters_[slot]);
                TouchHotsetColdSlot(slot,true);
            auto& record = gpu_slot_records_[slot];
            if (selected_slots && selected_slots[i] >= 0 && selected_slots[i] != slot)
                throw std::logic_error("Device resident route changed identity");
            if ((!frozen_ || frozen_slots_[slot]) && (!selected_slots || selected_slots[i] == slot) &&
                ValidateReadyBinding(layer, ids[i], slot, record.generation)) {
                record.state = ExpertSlotState::Ready; // cache the completed-DMA observation, as acquire does
                const int r = lease.ready_count++;
                lease.ready_slots[r] = slot; lease.ready_ranks[r] = i; lease.ready_ids[r] = ids[i];
                lease.ready_generations[r] = record.generation;
                ++record.direct_leases;
                direct_slots[i] = slot;
                ++stats_.hit_count;
                continue;
            }
        }
        const int c = lease.cold_count++;
        lease.cold_ids[c] = ids[i]; lease.cold_ranks[c] = i;
    }
    stats_.direct_route_entries += count;
    stats_.direct_entries += lease.ready_count;
    stats_.cold_request_entries += lease.cold_count;
    if (cache_trace_) {
        auto& out = cache_trace_->Row("direct_bind");
        out << ",\"forward\":" << trace_forward_id_ << ",\"layer\":" << layer << ",\"ids\":";
        Qwen4ExpCacheTrace::Ints(out, ids, count);
        out << ",\"initial_slots\":"; Qwen4ExpCacheTrace::Ints(out, initial_slots.data(), count);
        out << ",\"direct_slots\":"; Qwen4ExpCacheTrace::Ints(out, direct_slots.data(), count);
        out << ",\"cold_ids\":"; Qwen4ExpCacheTrace::Ints(out, lease.cold_ids.data(), lease.cold_count);
        cache_trace_->End();
    }
    return lease;
}

GpuResidentLayerLease::~GpuResidentLayerLease() {
    if (cache_) cache_->ReleaseResidentLayer(*this, true);
}

GpuResidentLayerLease::GpuResidentLayerLease(GpuResidentLayerLease&& other) noexcept
    : cache_(other.cache_), layer_(other.layer_), count_(other.count_),
      slots_(other.slots_), generations_(other.generations_) { other.cache_ = nullptr; }

GpuReadyLease GpuResidentLayerLease::BindRoute(const int* ids, const int* selected, int count) {
    if (!cache_ || !ids || !selected || count < 1 || count > GpuReadyLease::kMaxEntries)
        throw std::invalid_argument("Invalid resident device route");
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= cache_->num_experts_per_layer_ ||
            selected[i] != cache_->resident_indices_host_[layer_ * cache_->num_experts_per_layer_ + ids[i]])
            throw std::logic_error("Device/host resident classification mismatch");
    }
    auto route = cache_->BindReadyImpl(layer_, ids, count, selected);
    cache_->ReleaseResidentLayer(*this, false);
    return route;
}

void Qwen4ExpExpertCache::EnableResidentDeviceTables() {
    if (!is_initialized() || num_experts_per_layer_ > 512)
        throw std::logic_error("Unsupported resident device table");
    if (resident_descriptors_dev_.valid()) return;
    resident_descriptors_dev_ = Tensor::allocate(*ctx_, {int64_t(num_slots_) * int64_t(sizeof(ops::MoeExpertSlotDev))}, dnnl::memory::data_type::u8);
    resident_indices_dev_ = Tensor::allocate(*ctx_, {num_layers_, num_experts_per_layer_}, dnnl::memory::data_type::s32);
    resident_descriptors_host_.resize(num_slots_);
    resident_descriptor_generations_.resize(num_slots_);
    resident_indices_host_.assign(size_t(num_layers_) * num_experts_per_layer_, -1);
    resident_index_uploads_.resize(num_layers_);
}

void Qwen4ExpExpertCache::BeginResidentDeviceEpoch() {
    if (!resident_descriptors_dev_.valid() || !is_initialized())
        throw std::logic_error("Resident device tables not initialized");
    if(frozen_ && frozen_descriptors_published_)return;
    // The previous forward's route waits already cover this upload. Retain the
    // explicit source-lifetime check for callers outside the normal backend.
    if (resident_descriptor_upload_ != sycl::event{}) resident_descriptor_upload_.wait_and_throw();
    std::copy(slots_dev_.begin(), slots_dev_.end(), resident_descriptors_host_.begin());
    for (int s = 0; s < num_slots_; ++s) resident_descriptor_generations_[s] = gpu_slot_records_[s].generation;
    const size_t bytes = size_t(num_slots_) * sizeof(ops::MoeExpertSlotDev);
    resident_descriptor_upload_ = ctx_->queue().memcpy(resident_descriptors_dev_.data(), resident_descriptors_host_.data(), bytes);
    stats_.resident_metadata_bytes += bytes;
    if(frozen_) {
        frozen_descriptors_published_=true;
        // A layer mask built against the previous descriptor generations may
        // have omitted a newly admitted slot. Rebuild it for this snapshot.
        std::fill(frozen_indices_published_.begin(),frozen_indices_published_.end(),false);
    }
}

GpuResidentLayerLease Qwen4ExpExpertCache::BeginResidentLayer(int layer) {
    if (!resident_descriptors_dev_.valid() || !is_initialized() || layer < 0 || layer >= num_layers_)
        throw std::logic_error("Invalid resident layer epoch");
    if (resident_index_uploads_[layer] != sycl::event{}) resident_index_uploads_[layer].wait_and_throw();
    GpuResidentLayerLease lease;
    lease.cache_ = this; lease.layer_ = layer;
    int* indices = resident_indices_host_.data() + layer * num_experts_per_layer_;
    std::fill_n(indices, num_experts_per_layer_, -1);
    for (int e = 0; e < num_experts_per_layer_; ++e) {
        const int slot = layer_expert_to_slot_[expert_flat_index(layer, e)];
        if (slot < 0 || (frozen_ && !frozen_slots_[slot]) || !ValidateReadyBinding(layer, e, slot, resident_descriptor_generations_[slot])) continue;
        gpu_slot_records_[slot].state = ExpertSlotState::Ready;
        indices[e] = slot;
        lease.slots_[lease.count_] = slot;
        lease.generations_[lease.count_++] = gpu_slot_records_[slot].generation;
        ++gpu_slot_records_[slot].direct_leases;
    }
    const size_t bytes = size_t(num_experts_per_layer_) * sizeof(int);
    if(!frozen_ || !frozen_indices_published_[layer]) {
        resident_index_uploads_[layer] = ctx_->queue().memcpy(resident_indices_dev_.data_as<int>() + layer * num_experts_per_layer_, indices, bytes);
        stats_.resident_metadata_bytes += bytes;
        if(frozen_)frozen_indices_published_[layer]=true;
    }
    return lease;
}

void Qwen4ExpExpertCache::ReleaseResidentLayer(GpuResidentLayerLease& lease, bool drain) {
    if (drain) {
        try { ctx_->synchronize(); } catch (...) { faulted_ = true; }
    }
    for (int i = 0; i < lease.count_; ++i) {
        auto& record = gpu_slot_records_[lease.slots_[i]];
        if (record.generation != lease.generations_[i] || !record.direct_leases) faulted_ = true;
        else --record.direct_leases;
    }
    lease.cache_ = nullptr;
}

sycl::event Qwen4ExpExpertCache::AcquireCold(const GpuReadyLease& lease, int* out_slots) {
    if (lease.cache_ != this) throw std::invalid_argument("Cold request has no active resident lease");
    for (int r = 0; r < lease.ready_count; ++r)
        if (!ValidateReadyBinding(lease.layer, lease.ready_ids[r], lease.ready_slots[r], lease.ready_generations[r]))
            throw std::logic_error("Stale resident lease before cold acquire");
    if (!lease.cold_count) return {};
    const auto begin = std::chrono::steady_clock::now();
    auto result = acquire_impl(lease.layer, lease.cold_ids.data(), lease.cold_count, out_slots, false);
    stats_.direct_cold_acquire_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    return result;
}

void Qwen4ExpExpertCache::ReleaseReady(GpuReadyLease& lease, const sycl::event& last_reader, bool drain) {
    if (drain) {
        try { ctx_->synchronize(); } catch (...) { faulted_ = true; }
    }
    for (int r = 0; r < lease.ready_count; ++r) {
        auto& record = gpu_slot_records_[lease.ready_slots[r]];
        if (record.generation != lease.ready_generations[r] || !record.direct_leases) {
            faulted_ = true;
            continue;
        }
        if (!drain) record.consumer_fence = last_reader;
        --record.direct_leases;
    }
    if (cache_trace_) {
        auto& out = cache_trace_->Row("direct_release");
        out << ",\"forward\":" << trace_forward_id_ << ",\"layer\":" << lease.layer;
        cache_trace_->End();
    }
    lease.cache_ = nullptr;
}

sycl::event Qwen4ExpExpertCache::acquire_impl(int layer, const int* expert_ids,
                                            int count, int* out_slots, bool touch_hits) {
    if (!initialized_ || count <= 0 || expert_ids == nullptr || out_slots == nullptr) {
        return {};
    }
    if (faulted_) {
        throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned after a failed acquire; reinitialize before use");
    }
    if (layer < 0 || layer >= num_layers_) {
        throw std::invalid_argument("Qwen4ExpExpertCache: invalid layer in acquire");
    }
    int unique_experts = 0;
    for (int i = 0; i < count; ++i) {
        const int expert = expert_ids[i];
        if (expert < 0 || expert >= num_experts_per_layer_) {
            throw std::invalid_argument("Qwen4ExpExpertCache: invalid expert id in acquire");
        }
        bool already_seen = false;
        for (int j = 0; j < i; ++j) {
            if (expert_ids[j] == expert) {
                already_seen = true;
                break;
            }
        }
        if (!already_seen) ++unique_experts;
    }
    if (unique_experts > num_slots_) {
        throw std::runtime_error("Qwen4ExpExpertCache: unique expert demand exceeds GPU slot capacity");
    }
    if(frozen_) {
        int cold_unique=0;
        for(int i=0;i<count;++i) {
            if(std::find(expert_ids,expert_ids+i,expert_ids[i])!=expert_ids+i)continue;
            const int s=layer_expert_to_slot_[expert_flat_index(layer,expert_ids[i])];
            if(s<0 || !frozen_slots_[s])++cold_unique;
        }
        if(cold_unique>num_slots_-frozen_count_)throw std::runtime_error("Cold batch exceeds frozen staging capacity");
    }
    ++stats_.demand_microbatches;
    ++stats_.acquire_calls;
    const int64_t trace_begin_ns = cache_trace_ ? Qwen4ExpCacheTrace::NowNs() : 0;
    const ExpertCacheStats trace_stats0 = cache_trace_ ? stats_ : ExpertCacheStats{};
    if (cache_trace_) {
        auto& out = cache_trace_->Row("acquire_begin");
        out << ",\"acquire\":" << ++trace_acquire_id_ << ",\"forward\":" << trace_forward_id_
            << ",\"layer\":" << layer << ",\"ids\":";
        Qwen4ExpCacheTrace::Ints(out, expert_ids, count);
        out << ",\"initial_slots\":[";
        for (int i = 0; i < count; ++i) {
            if (i) out << ',';
            out << layer_expert_to_slot_[expert_flat_index(layer, expert_ids[i])];
        }
        out << "],\"host_states\":[";
        for (int i = 0; i < count; ++i) {
            if (i) out << ',';
            const int h = host_pool_enabled_ ? host_layer_expert_to_slot_[expert_flat_index(layer, expert_ids[i])] : -1;
            out << (h < 0 ? 0 : static_cast<int>(host_slot_state_[h].load(std::memory_order_acquire)));
        }
        out << "],\"gpu_states\":[";
        for (int i = 0; i < count; ++i) {
            if (i) out << ',';
            const int s = layer_expert_to_slot_[expert_flat_index(layer, expert_ids[i])];
            // Copying is the recorded state, not a claim that DMA is still active.
            out << (s < 0 ? 0 : static_cast<int>(gpu_slot_records_[s].state));
        }
        out << "],\"pinned\":" << num_pinned_slots_ << ",\"touch_hits\":" << (touch_hits ? 1 : 0);
        cache_trace_->End();
    }
    if (touch_hits) trace_demand_route(layer, expert_ids, count);
    // AILA_DEBUG_Q4E_FAKE_HITS: treat every expert as resident (no fill/DMA, wrong outputs) to
    // measure the decode time floor of a 100% hit rate.
    static const bool s_fake_hits = aila::env::read_flag("AILA_DEBUG_Q4E_FAKE_HITS", false);
    if (s_fake_hits) {
        for (int i = 0; i < count; ++i) {
            out_slots[i] = (layer * num_experts_per_layer_ + expert_ids[i]) % num_slots_;
            bind_slot_nested_metadata(out_slots[i], layer, expert_ids[i]);
        }
        stats_.hit_count += static_cast<uint64_t>(count);
        return {};
    }

    misses_scratch_.clear();
    batch_reserved_slots_scratch_.clear();
    for (int h : batch_reserved_hslots_scratch_) {
        host_slot_in_batch_readers_[h] = false;
    }
    batch_reserved_hslots_scratch_.clear();

    struct AcquireGuard {
        Qwen4ExpExpertCache* self;
        bool committed = false;
        bool bindings_published = false;
        bool dma_submitted = false;
        ~AcquireGuard() {
            if (!committed && self) {
                if (dma_submitted && self->copy_q_) {
                    try { self->copy_q_->wait(); } catch (...) {}
                }
                if (bindings_published || dma_submitted) {
                    self->faulted_ = true;
                    for (int s : self->batch_reserved_slots_scratch_) {
                        self->gpu_slot_records_[s].state = ExpertSlotState::Faulted;
                    }
                }
                for (int s : self->batch_reserved_slots_scratch_) {
                    self->slot_reserved_[s] = false;
                }
                for (int h : self->batch_reserved_hslots_scratch_) {
                    self->host_slot_reserved_[h] = false;
                    self->host_slot_in_batch_readers_[h] = false;
                }
                self->batch_reserved_slots_scratch_.clear();
                self->batch_reserved_hslots_scratch_.clear();
                self->misses_scratch_.clear();
            }
        }
    } guard{this, false, false, false};

    // Phase 1: check hits and deduplicate within this batch
    int hit_count_in_batch = 0;
    bool hit_dma_pending = false;
    for (int i = 0; i < count; ++i) {
        int expert = expert_ids[i];
        int flat_idx = expert_flat_index(layer, expert);
        int slot = layer_expert_to_slot_[flat_idx];
        if (slot >= 0) {
            out_slots[i] = slot;
            if (!slot_reserved_[slot]) {
                slot_reserved_[slot] = true;
                batch_reserved_slots_scratch_.push_back(slot);
                if (touch_hits && !slot_pinned_[slot]) {
                    lru_list_.splice(lru_list_.begin(), lru_list_, slot_iters_[slot]);
                    TouchHotsetColdSlot(slot,true);
                }
            }
            auto& record = gpu_slot_records_[slot];
            if (record.state == ExpertSlotState::Copying) {
                if (is_event_complete(record.dma_event)) record.state = ExpertSlotState::Ready;
                else hit_dma_pending = true;
            }
            ++stats_.hit_count;
            ++hit_count_in_batch;
        } else {
            out_slots[i] = -1;
        }
    }

    // Fast-path: 100% GPU cache hits!
    // No wait on prefetch, zero stall! Background prefetch continues asynchronously!
    if (hit_count_in_batch == count) {
        guard.committed = true;
        for (int s : batch_reserved_slots_scratch_) {
            slot_reserved_[s] = false;
        }
        batch_reserved_slots_scratch_.clear();
        trace_acquire_end(trace_stats0, trace_begin_ns);
        return hit_dma_pending ? copy_q_->ext_oneapi_submit_barrier() : sycl::event{};
    }

    // Phase 2: assign slots for misses, protecting all batch_reserved_slots from eviction
    for (int i = 0; i < count; ++i) {
        if (out_slots[i] >= 0) continue;
        int expert = expert_ids[i];
        int flat_idx = expert_flat_index(layer, expert);

        // Check if another miss earlier in this same batch already bound this key
        int bound_slot = layer_expert_to_slot_[flat_idx];
        if (bound_slot >= 0) {
            out_slots[i] = bound_slot;
            continue;
        }

        // Search for least recently used unreserved slot from back,
        // with cost-aware preference (B4): among candidates near the LRU tail,
        // prefer evicting an expert that is already cached in Host RAM
        // (lower replacement cost: H2D copy only, vs disk mmap read + CPU repack + H2D).
        int victim_slot = -1;
        int trace_candidates[8] = {};
        int trace_backed[8] = {};
        int trace_candidates_count = 0;
        const bool transient_ring=adaptive_hotset_ && frozen_ && !hotset_cold_slots_.empty();
        auto& victim_lru=hotset_cold_lru_.empty()?lru_list_:hotset_cold_lru_;
        if(transient_ring) {
            // This region never admits demand into the stable tier. Select only
            // its bounded slots, without walking thousands of protected LRU nodes.
            for(size_t n=0;n<hotset_cold_slots_.size();++n) {
                const int s=hotset_cold_slots_[hotset_cold_cursor_];
                hotset_cold_cursor_=(hotset_cold_cursor_+1)%hotset_cold_slots_.size();
                if(!slot_reserved_[s] && !slot_pinned_[s] && !gpu_slot_records_[s].direct_leases) {victim_slot=s;break;}
            }
        }
        if (!transient_ring && cost_aware_eviction_ && host_pool_enabled_) {
            constexpr int kMaxCandidates = 8;
            int candidates_checked = 0;
            int fallback_victim = -1;
            for (auto r_it = victim_lru.rbegin(); r_it != victim_lru.rend() && candidates_checked < kMaxCandidates; ++r_it) {
                int s = *r_it;
                if (!slot_reserved_[s] && !slot_pinned_[s] && !frozen_slots_[s] && !gpu_slot_records_[s].direct_leases) {
                    if (fallback_victim < 0) fallback_victim = s;
                    ++candidates_checked;
                    uint64_t old_k = slot_to_key_[s];
                    if (cache_trace_) trace_candidates[trace_candidates_count++] = s;
                    if (old_k == UINT64_MAX) {
                        victim_slot = s;
                        break;
                    }
                    int old_layer = static_cast<int>(old_k >> 32);
                    int old_exp = static_cast<int>(old_k & 0xFFFFFFFF);
                    int h_flat = expert_flat_index(old_layer, old_exp);
                    int hslot = host_layer_expert_to_slot_[h_flat];
                    if (hslot >= 0 &&
                        host_slot_state_[hslot].load(std::memory_order_acquire) == static_cast<uint8_t>(ExpertSlotState::Ready)) {
                        if (cache_trace_) trace_backed[trace_candidates_count - 1] = 1;
                        victim_slot = s;
                        break;
                    }
                }
            }
            if (victim_slot < 0) victim_slot = fallback_victim;
        }

        if (victim_slot < 0 && !transient_ring) {
            for (auto r_it = victim_lru.rbegin(); r_it != victim_lru.rend(); ++r_it) {
                int s = *r_it;
                if (!slot_reserved_[s] && !slot_pinned_[s] && !frozen_slots_[s] && !gpu_slot_records_[s].direct_leases) {
                    victim_slot = s;
                    break;
                }
            }
        }

        if (victim_slot < 0) {
            for (int s : batch_reserved_slots_scratch_) slot_reserved_[s] = false;
            for (int h : batch_reserved_hslots_scratch_) host_slot_reserved_[h] = false;
            throw std::runtime_error("Qwen4ExpExpertCache: all slots reserved in batch, count exceeds num_slots");
        }

        if (cache_trace_) {
            const uint64_t old = slot_to_key_[victim_slot];
            auto& out = cache_trace_->Row("victim");
            out<<",\"transient_ring\":"<<int(transient_ring);
            out << ",\"acquire\":" << trace_acquire_id_ << ",\"slot\":" << victim_slot
                << ",\"key\":" << flat_idx << ",\"old_key\":"
                << (old == UINT64_MAX ? -1 : expert_flat_index(static_cast<int>(old >> 32), static_cast<int>(old & 0xffffffff)))
                << ",\"candidates\":";
            Qwen4ExpCacheTrace::Ints(out, trace_candidates, trace_candidates_count);
            out << ",\"backed\":";
            Qwen4ExpCacheTrace::Ints(out, trace_backed, trace_candidates_count);
            cache_trace_->End();
        }
        // B1: Consumer reader fence: protect GPU reader kernels from previous token before overwriting
        if (gpu_slot_records_[victim_slot].consumer_fence != sycl::event{}) {
            gpu_slot_records_[victim_slot].consumer_fence.wait_and_throw();
            gpu_slot_records_[victim_slot].consumer_fence = sycl::event{};
        }

        // Validate the descriptor before mutating the cache's LRU and key maps.
        bind_slot_nested_metadata(victim_slot, layer, expert);

        // Evict old key if bound
        uint64_t old_key = slot_to_key_[victim_slot];
        if (old_key != UINT64_MAX) {
            int old_layer = static_cast<int>(old_key >> 32);
            int old_expert = static_cast<int>(old_key & 0xFFFFFFFF);
            layer_expert_to_slot_[expert_flat_index(old_layer, old_expert)] = -1;
        }
        uint64_t key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);
        layer_expert_to_slot_[flat_idx] = victim_slot;
        slot_to_key_[victim_slot] = key;
        gpu_slot_records_[victim_slot].generation++;
        gpu_slot_records_[victim_slot].key = key;
        gpu_slot_records_[victim_slot].state = ExpertSlotState::Copying;
        gpu_slot_records_[victim_slot].fill_owner = 0; // demand
        guard.bindings_published = true;

        // Move victim to MRU (front) and reserve for this batch
        lru_list_.splice(lru_list_.begin(), lru_list_, slot_iters_[victim_slot]);
        TouchHotsetColdSlot(victim_slot,true);

        slot_reserved_[victim_slot] = true;
        batch_reserved_slots_scratch_.push_back(victim_slot);

        out_slots[i] = victim_slot;
        misses_scratch_.push_back({i, layer, expert, victim_slot, nullptr});
        ++stats_.miss_count;
    }

    // Phase 3: Fine-grained DMA transfers
    if (!host_pool_enabled_ && last_dma_event_ != sycl::event{}) {
        try { last_dma_event_.wait_and_throw(); } catch (...) {}
    }

    sycl::event dma_done_event{};
    if (!misses_scratch_.empty()) {
        const auto t_submit_start = std::chrono::high_resolution_clock::now();
        // Worker completions own distinct slot records. The coordinator owns all
        // batch accounting and conservatively drains on failure after this point.
        guard.dma_submitted = true;
        auto copy_to_slot = [&](int slot_id, const uint8_t* src, int hslot = -1) -> sycl::event {
            sycl::event ev = copy_q_->memcpy(slot_dev_base_[slot_id], src, slot_bytes_);
            gpu_slot_records_[slot_id].dma_event = ev;
            gpu_slot_records_[slot_id].state = ExpertSlotState::Copying;
            if (hslot >= 0 && hslot < num_host_slots_) {
                host_slot_dma_events_[hslot] = ev;
                host_slot_records_[hslot].source_reader_fence = ev;
            }
            return ev;
        };
        to_fill_scratch_.clear();

        for (const auto& miss : misses_scratch_) {
            // Check if this miss matches an in-flight prefetch item
            int matched_pf = -1;
            for (size_t p = 0; p < prefetch_items_.size(); ++p) {
                if (prefetch_items_[p].layer == miss.layer && prefetch_items_[p].expert_id == miss.expert_id) {
                    matched_pf = static_cast<int>(p);
                    break;
                }
            }

            if (matched_pf >= 0) {
                auto& pf_item = prefetch_items_[matched_pf];
                ensure_prefetch_item_ready(pf_item, matched_pf);
                const uint8_t* src = (pf_item.hslot >= 0)
                    ? host_slot_ptrs_[pf_item.hslot]
                    : (prefetch_staging_ptr_ + static_cast<size_t>(matched_pf) * slot_bytes_);
                copy_to_slot(miss.slot_id, src, pf_item.hslot);
                stats_.total_transferred_bytes += slot_bytes_;
                stats_.prefetch_useful_bytes += slot_bytes_;
                pf_item.used = true;
                if (pf_item.hslot >= 0) {
                    touch_host_slot(pf_item.hslot);
                    add_batch_reader_hslot(pf_item.hslot);
                    ++stats_.host_pool_hits;
                }
                ++stats_.prefetch_used;
                continue;
            }

            if (host_pool_enabled_) {
                int h_flat = expert_flat_index(miss.layer, miss.expert_id);
                int hslot = host_layer_expert_to_slot_[h_flat];
                if (hslot >= 0 && host_slot_state_[hslot].load(std::memory_order_acquire) == static_cast<uint8_t>(ExpertSlotState::Ready)) {
                    touch_host_slot(hslot);
                    add_batch_reader_hslot(hslot);
                    const uint8_t* src = host_slot_ptrs_[hslot];
                    copy_to_slot(miss.slot_id, src, hslot);
                    stats_.total_transferred_bytes += slot_bytes_;
                    ++stats_.host_pool_hits;
                    continue;
                }
            }

            to_fill_scratch_.push_back(miss);
        }

        // Cancel speculative prefetch before allocating host slots for demand misses,
        // ensuring no background prefetch workers are writing to victim host slots.
        if (!to_fill_scratch_.empty() && prefetch_in_flight_) {
            cancel_prefetch();
        }

        if (host_pool_enabled_ && !to_fill_scratch_.empty()) {
            const auto t0 = std::chrono::steady_clock::now();
            const int parts = fill_parts_;
            const size_t max_chunk = std::max(1, num_host_slots_);

            for (size_t m = 0; m < to_fill_scratch_.size(); ) {
                size_t avail_hslots = 0;
                for (int h = 0; h < num_host_slots_; ++h) {
                    if (!host_slot_reserved_[h]) ++avail_hslots;
                }

                if (avail_hslots == 0) {
                    copy_q_->wait_and_throw();
                    for (int h : batch_reserved_hslots_scratch_) {
                        host_slot_reserved_[h] = false;
                        host_slot_in_batch_readers_[h] = false;
                        host_slot_dma_events_[h] = sycl::event{};
                    }
                    batch_reserved_hslots_scratch_.clear();
                    for (int h = 0; h < num_host_slots_; ++h) {
                        if (!host_slot_reserved_[h]) ++avail_hslots;
                    }
                }

                const size_t chunk_size = std::min(std::max<size_t>(1, avail_hslots), to_fill_scratch_.size() - m);
                assigned_hslots_scratch_.resize(chunk_size);
                assigned_hslot_gens_scratch_.resize(chunk_size);

                for (size_t c = 0; c < chunk_size; ++c) {
                    const auto& miss = to_fill_scratch_[m + c];
                    int victim_hslot = pick_victim_hslot();
                    if (victim_hslot < 0) {
                        copy_q_->wait_and_throw();
                        for (int h : batch_reserved_hslots_scratch_) {
                            host_slot_reserved_[h] = false;
                            host_slot_in_batch_readers_[h] = false;
                            host_slot_dma_events_[h] = sycl::event{};
                            host_slot_records_[h].source_reader_fence = sycl::event{};
                        }
                        batch_reserved_hslots_scratch_.clear();
                        victim_hslot = pick_victim_hslot();
                        if (victim_hslot < 0) {
                            throw std::runtime_error("Qwen4ExpExpertCache: no unreserved host slot available in pool");
                        }
                    }
                    add_batch_reader_hslot(victim_hslot);
                    const uint64_t old_key = host_slot_to_key_[victim_hslot];
                    if (old_key != UINT64_MAX) {
                        int old_layer = static_cast<int>(old_key >> 32);
                        int old_expert = static_cast<int>(old_key & 0xFFFFFFFF);
                        host_layer_expert_to_slot_[expert_flat_index(old_layer, old_expert)] = -1;
                    }
                    const uint64_t key = (static_cast<uint64_t>(miss.layer) << 32) | static_cast<uint32_t>(miss.expert_id);
                    host_layer_expert_to_slot_[expert_flat_index(miss.layer, miss.expert_id)] = victim_hslot;
                    host_slot_to_key_[victim_hslot] = key;
                    const uint64_t next_gen = host_slot_records_[victim_hslot].generation.load(std::memory_order_relaxed) + 1;
                    host_slot_records_[victim_hslot].generation.store(next_gen, std::memory_order_release);
                    host_slot_records_[victim_hslot].key = key;
                    host_slot_records_[victim_hslot].fill_owner = 0; // demand
                    host_slot_state_[victim_hslot].store(static_cast<uint8_t>(ExpertSlotState::Filling), std::memory_order_release);

                    if (old_key == UINT64_MAX) {
                        host_slot_is_protected_[victim_hslot] = false;
                        host_probationary_list_.push_front(victim_hslot);
                        host_slot_iters_[victim_hslot] = host_probationary_list_.begin();
                    } else if (host_slot_is_protected_[victim_hslot]) {
                        host_slot_is_protected_[victim_hslot] = false;
                        host_probationary_list_.splice(host_probationary_list_.begin(),
                                                       host_protected_list_, host_slot_iters_[victim_hslot]);
                    } else {
                        host_probationary_list_.splice(host_probationary_list_.begin(),
                                                       host_probationary_list_, host_slot_iters_[victim_hslot]);
                    }

                    assigned_hslots_scratch_[c] = victim_hslot;
                    assigned_hslot_gens_scratch_[c] = next_gen;
                    parts_left_scratch_[c].store(parts, std::memory_order_relaxed);
                    ++stats_.host_pool_misses;
                }

                const std::function<void(int)> fill_task = [&](int task) {
                    const int c = task / parts;
                    const auto& miss = to_fill_scratch_[m + c];
                    int hslot = assigned_hslots_scratch_[c];
                    uint64_t exp_gen = assigned_hslot_gens_scratch_[c];
                    uint8_t* host_slot_ptr = host_slot_ptrs_[hslot];
                    load_expert_to_staging(miss.layer, miss.expert_id, host_slot_ptr, task % parts, parts, exp_gen);
                    if (parts_left_scratch_[c].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        if (host_slot_records_[hslot].generation.load(std::memory_order_acquire) == exp_gen) {
                            host_slot_state_[hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
                            copy_to_slot(miss.slot_id, host_slot_ptr, hslot);
                        }
                    }
                };
                fill_pool_->run(static_cast<int>(chunk_size) * parts, fill_task);
                stats_.total_transferred_bytes += static_cast<uint64_t>(chunk_size) * slot_bytes_;

                m += chunk_size;
                if (m < to_fill_scratch_.size()) {
                    // Drain in-flight copies of this chunk before the next chunk could evict and overwrite the same host slots
                    copy_q_->wait_and_throw();
                    for (int h : batch_reserved_hslots_scratch_) {
                        host_slot_reserved_[h] = false;
                        host_slot_in_batch_readers_[h] = false;
                        host_slot_dma_events_[h] = sycl::event{};
                        host_slot_records_[h].source_reader_fence = sycl::event{};
                    }
                    batch_reserved_hslots_scratch_.clear();
                }
            }
            const auto fill_ns_elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
            stats_.fill_ns += fill_ns_elapsed;
            stats_.demand_exposed_wait_ns += fill_ns_elapsed;
        } else if (!to_fill_scratch_.empty()) {
            // Double-buffered staging: while one half DMAs, the other half is being filled.
            const int staging_capacity = std::max(1, static_cast<int>(staging_buffer_bytes_ / slot_bytes_));
            const int half_capacity = std::max(1, staging_capacity / 2);
            sycl::event half_done[2];
            bool half_busy[2] = {false, false};
            auto timed_wait = [&](sycl::event& ev) {
                const auto t_sync = std::chrono::steady_clock::now();
                ev.wait_and_throw();
                stats_.dma_wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_sync).count());
            };
            int chunk_index = 0;
            for (size_t m = 0; m < to_fill_scratch_.size(); ++chunk_index) {
                const int chunk_size = std::min<int>(half_capacity, static_cast<int>(to_fill_scratch_.size() - m));
                const int half = chunk_index % 2;
                if (half_busy[half]) timed_wait(half_done[half]);
                uint8_t* chunk_base = pinned_staging_ptr_ + static_cast<size_t>(half) * half_capacity * slot_bytes_;

                const auto t0 = std::chrono::steady_clock::now();
                const int parts = fill_parts_;
                // The thread finishing an expert's last part submits its h2d copy, so DMA of
                // early experts overlaps the fill of later ones (SYCL queues are thread-safe).
                for (size_t c = 0; c < static_cast<size_t>(chunk_size); ++c) {
                    parts_left_scratch_[c].store(parts, std::memory_order_relaxed);
                }
                const std::function<void(int)> fill_task = [&](int task) {
                    const int c = task / parts;
                    const auto& miss = to_fill_scratch_[m + c];
                    uint8_t* staging_slot = chunk_base + static_cast<size_t>(c) * slot_bytes_;
                    const uint64_t exp_gen = gpu_slot_records_[miss.slot_id].generation;
                    load_expert_to_staging(miss.layer, miss.expert_id, staging_slot, task % parts, parts, exp_gen);
                    if (parts_left_scratch_[c].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        copy_to_slot(miss.slot_id, staging_slot, -1);
                    }
                };
                fill_pool_->run(chunk_size * parts, fill_task);
                stats_.total_transferred_bytes += static_cast<uint64_t>(chunk_size) * slot_bytes_;
                const auto fill_ns_elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());
                stats_.fill_ns += fill_ns_elapsed;
                stats_.demand_exposed_wait_ns += fill_ns_elapsed;

                half_done[half] = copy_q_->ext_oneapi_submit_barrier();
                half_busy[half] = true;
                m += static_cast<size_t>(chunk_size);
            }
        } // end of else if (!to_fill.empty())

        // Capture completion of all DMA operations on copy_q_ with a SYCL hardware barrier.
        // Decouple per-expert events on individual slot records from the whole-batch barrier.
        if (copy_q_) {
            dma_done_event = copy_q_->ext_oneapi_submit_barrier();
            last_dma_event_ = dma_done_event;
            if (dma_timing_stats_.last_memcpy_count > 0 && dma_timing_stats_.last_copy_ns == 0) {
                ++dma_timing_stats_.batches_unprofiled;
            }
            batch_dma_events_.clear();
            for (const auto& miss : misses_scratch_) {
                batch_dma_events_.push_back(gpu_slot_records_[miss.slot_id].dma_event);
                if (is_event_complete(gpu_slot_records_[miss.slot_id].dma_event)) {
                    gpu_slot_records_[miss.slot_id].state = ExpertSlotState::Ready;
                }
            }

            const auto t_submit_end = std::chrono::high_resolution_clock::now();
            const uint64_t submit_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                t_submit_end - t_submit_start).count());

            const auto batch_memcpy_count = static_cast<uint32_t>(misses_scratch_.size());
            const uint64_t batch_bytes = static_cast<uint64_t>(batch_memcpy_count) * slot_bytes_;
            dma_timing_stats_.last_batch_bytes = batch_bytes;
            dma_timing_stats_.last_memcpy_count = batch_memcpy_count;
            dma_timing_stats_.last_submit_ns = submit_ns;
            dma_timing_stats_.last_copy_ns = 0;
            dma_timing_stats_.last_consumer_gap_ns = 0;
            dma_timing_stats_.last_effective_gbps = 0.0;
            dma_timing_stats_.total_submit_ns += submit_ns;
            dma_timing_stats_.total_dma_bytes += batch_bytes;
            dma_timing_stats_.batches_measured++;
            last_batch_submit_end_time_ = t_submit_end;
            last_batch_dma_event_ = dma_done_event;
            new_dma_batch_pending_ = true;
        }
    }

    cancel_prefetch();

    // Phase 4: release batch reservations
    guard.committed = true;
    for (int s : batch_reserved_slots_scratch_) {
        slot_reserved_[s] = false;
    }
    for (int h : batch_reserved_hslots_scratch_) {
        host_slot_reserved_[h] = false;
        host_slot_in_batch_readers_[h] = false;
    }
    batch_reserved_slots_scratch_.clear();
    batch_reserved_hslots_scratch_.clear();

    trace_acquire_end(trace_stats0, trace_begin_ns);
    return dma_done_event;
}

void Qwen4ExpExpertCache::prefetch_async(int layer, const int* expert_ids, int count, int quota) {
    if (!initialized_ || count <= 0 || expert_ids == nullptr) {
        return;
    }
    if (faulted_) {
        throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned after a failed acquire; reinitialize before use");
    }
    static const bool s_fake_hits = aila::env::read_flag("AILA_DEBUG_Q4E_FAKE_HITS", false);
    if (s_fake_hits) return; // acquire() binds nothing in this mode

    cancel_prefetch();

    prefetch_items_.clear();
    const int max_slots = host_pool_enabled_ ? num_host_slots_ : prefetch_staging_slots_;
    const int effective_quota = (quota > 0) ? quota : count;
    const int target_quota = std::max(1, std::min(max_slots, effective_quota));
    const int parts = fill_parts_;
    // B3: Filter GPU resident, host Ready, and duplicate items BEFORE quota truncation!
    for (int i = 0; i < count && static_cast<int>(prefetch_items_.size()) < target_quota; ++i) {
        const int expert = expert_ids[i];
        if (expert < 0 || expert >= num_experts_per_layer_) continue;
        if (contains(layer, expert)) continue;
        if (host_pool_enabled_) {
            int hflat = expert_flat_index(layer, expert);
            int cur_hslot = host_layer_expert_to_slot_[hflat];
            if (cur_hslot >= 0 &&
                host_slot_state_[cur_hslot].load(std::memory_order_acquire) == static_cast<uint8_t>(ExpertSlotState::Ready)) {
                continue;
            }
        }
        bool dup = false;
        for (const auto& item : prefetch_items_) {
            if (item.expert_id == expert) { dup = true; break; }
        }
        if (!dup) {
            prefetch_items_.emplace_back(layer, expert, -1, parts, 0);
            if (reader_) reader_->prefetch_expert(layer, expert);
        }
    }
    if (prefetch_items_.empty()) return;

    const int n = static_cast<int>(prefetch_items_.size());
    stats_.prefetch_issued += static_cast<uint64_t>(n);
    stats_.prefetch_issued_bytes += static_cast<uint64_t>(n) * slot_bytes_;
    prefetch_in_flight_ = true;

    if (host_pool_enabled_) {
        for (int c = 0; c < n; ++c) {
            auto& item = prefetch_items_[c];
            int victim_hslot = pick_victim_hslot();
            if (victim_hslot < 0) {
                prefetch_items_.resize(c);
                break;
            }
            host_slot_reserved_[victim_hslot] = true;
            const uint64_t old_key = host_slot_to_key_[victim_hslot];
            if (old_key != UINT64_MAX) {
                int old_layer = static_cast<int>(old_key >> 32);
                int old_expert = static_cast<int>(old_key & 0xFFFFFFFF);
                host_layer_expert_to_slot_[expert_flat_index(old_layer, old_expert)] = -1;
            }
            const uint64_t key = (static_cast<uint64_t>(item.layer) << 32) | static_cast<uint32_t>(item.expert_id);
            host_layer_expert_to_slot_[expert_flat_index(item.layer, item.expert_id)] = victim_hslot;
            host_slot_to_key_[victim_hslot] = key;
            const uint64_t next_gen = host_slot_records_[victim_hslot].generation.load(std::memory_order_relaxed) + 1;
            host_slot_records_[victim_hslot].generation.store(next_gen, std::memory_order_release);
            host_slot_records_[victim_hslot].key = key;
            host_slot_records_[victim_hslot].fill_owner = 1; // prefetch
            host_slot_state_[victim_hslot].store(static_cast<uint8_t>(ExpertSlotState::Filling), std::memory_order_release);

            if (old_key == UINT64_MAX) {
                host_slot_is_protected_[victim_hslot] = false;
                host_probationary_list_.push_front(victim_hslot);
                host_slot_iters_[victim_hslot] = host_probationary_list_.begin();
            } else if (host_slot_is_protected_[victim_hslot]) {
                host_slot_is_protected_[victim_hslot] = false;
                host_probationary_list_.splice(host_probationary_list_.begin(),
                                               host_protected_list_, host_slot_iters_[victim_hslot]);
            } else {
                host_probationary_list_.splice(host_probationary_list_.begin(),
                                               host_probationary_list_, host_slot_iters_[victim_hslot]);
            }

            item.hslot = victim_hslot;
            item.expected_generation = next_gen;
        }
    }

    if (prefetch_items_.empty()) {
        prefetch_in_flight_ = false;
        return;
    }

    const int effective_n = static_cast<int>(prefetch_items_.size());
    fill_pool_->start(effective_n * parts, [this, parts](int task) {
        const int c = task / parts;
        const int p = task % parts;
        if (c >= static_cast<int>(prefetch_items_.size())) return;
        auto& item = prefetch_items_[c];
        if (item.cancelled.load(std::memory_order_relaxed)) return;

        const uint64_t bit = uint64_t{1} << p;
        if (!(item.claimed_parts.load(std::memory_order_relaxed) & bit)) {
            const uint64_t old = item.claimed_parts.fetch_or(bit, std::memory_order_acq_rel);
            if (!(old & bit)) {
                uint8_t* dst = (item.hslot >= 0) ? host_slot_ptrs_[item.hslot]
                                                 : prefetch_staging_ptr_ + static_cast<size_t>(c) * slot_bytes_;
                load_expert_to_staging(item.layer, item.expert_id, dst, p, parts, item.expected_generation);
                if (item.parts_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    item.ready.store(true, std::memory_order_release);
                    if (item.hslot >= 0) {
                        if (host_slot_records_[item.hslot].generation.load(std::memory_order_acquire) == item.expected_generation) {
                            host_slot_state_[item.hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
                        }
                    }
                }
            }
        }
    });
}

void Qwen4ExpExpertCache::wait_prefetch() {
    if (!prefetch_in_flight_) return;
    if (fill_pool_) fill_pool_->wait();
    if (host_pool_enabled_) {
        for (auto& item : prefetch_items_) {
            if (item.hslot >= 0 && item.hslot < num_host_slots_) {
                if (!host_slot_in_batch_readers_[item.hslot]) {
                    host_slot_reserved_[item.hslot] = false;
                }
                if (item.parts_remaining.load(std::memory_order_acquire) == 0) {
                    item.ready.store(true, std::memory_order_release);
                    if (host_slot_records_[item.hslot].generation.load(std::memory_order_acquire) == item.expected_generation) {
                        host_slot_state_[item.hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
                    }
                }
            }
        }
    }
    prefetch_in_flight_ = false;
}

void Qwen4ExpExpertCache::wait_copy() {
    if (copy_q_) {
        try {
            copy_q_->wait_and_throw();
        } catch (...) {}
    }
}

void Qwen4ExpExpertCache::demote(int layer, const int* expert_ids, int count) {
    if (!initialized_ || expert_ids == nullptr) return;
    if (cache_trace_) {
        auto& out = cache_trace_->Row("demote");
        out << ",\"forward\":" << trace_forward_id_ << ",\"layer\":" << layer << ",\"ids\":";
        Qwen4ExpCacheTrace::Ints(out, expert_ids, count);
        cache_trace_->End();
    }
    if (faulted_) {
        throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned after a failed acquire; reinitialize before use");
    }
    for (int i = 0; i < count; ++i) {
        int e = expert_ids[i];
        if (e < 0 || e >= num_experts_per_layer_) continue;
        int slot = layer_expert_to_slot_[expert_flat_index(layer, e)];
        if (slot >= 0 && !slot_pinned_[slot]) {
            lru_list_.splice(lru_list_.end(), lru_list_, slot_iters_[slot]);
            TouchHotsetColdSlot(slot,false);
        }
        if (host_pool_enabled_) {
            int hslot = host_layer_expert_to_slot_[expert_flat_index(layer, e)];
            if (hslot >= 0) {
                if (host_slot_is_protected_[hslot]) {
                    host_slot_is_protected_[hslot] = false;
                    host_probationary_list_.splice(host_probationary_list_.end(),
                                                   host_protected_list_, host_slot_iters_[hslot]);
                } else {
                    host_probationary_list_.splice(host_probationary_list_.end(),
                                                   host_probationary_list_, host_slot_iters_[hslot]);
                }
            }
        }
    }
}

namespace {

inline void part_range(size_t len, int part, int nparts, size_t& begin, size_t& end) {
    begin = len * static_cast<size_t>(part) / static_cast<size_t>(nparts);
    end = len * static_cast<size_t>(part + 1) / static_cast<size_t>(nparts);
}

// Copies packed bytes and either keeps nested absmax codes or expands absmax
// to f32 for the [part, nparts) slice of one expert matrix.
void fill_expert_matrix(const Bnb4BitHostWeightRef& ref, int expert_id,
                        uint8_t* packed_dst, size_t packed_bytes,
                        uint8_t* absmax_dst, size_t absmax_count, bool compact_nested,
                        int part, int nparts) {
    size_t b0 = 0, b1 = 0;
    part_range(packed_bytes, part, nparts, b0, b1);
    if (ref.packed_weight.valid()) {
        const uint8_t* src = ref.packed_weight.ptr + static_cast<size_t>(expert_id) * packed_bytes;
        std::memcpy(packed_dst + b0, src + b0, b1 - b0);
    } else {
        std::memset(packed_dst + b0, 0, b1 - b0);
    }

    size_t a0 = 0, a1 = 0;
    part_range(absmax_count, part, nparts, a0, a1);
    if (compact_nested) {
        if (ref.absmax.valid()) {
            const size_t expert_base = static_cast<size_t>(expert_id) * absmax_count;
            std::memcpy(absmax_dst + a0, ref.absmax_codes_ptr() + expert_base + a0, a1 - a0);
        } else {
            std::memset(absmax_dst + a0, 0, a1 - a0);
        }
        return;
    }
    float* absmax_f32 = reinterpret_cast<float*>(absmax_dst);
    if (!ref.absmax.valid()) {
        std::memset(absmax_f32 + a0, 0, (a1 - a0) * sizeof(float));
        return;
    }
    const size_t expert_base = static_cast<size_t>(expert_id) * absmax_count;
    if (!ref.quant_state.nested) {
        std::memcpy(absmax_f32 + a0, ref.absmax_f32_ptr() + expert_base + a0, (a1 - a0) * sizeof(float));
        return;
    }
    const uint8_t* codes = ref.absmax_codes_ptr() + expert_base;
    const float* nested_absmax = ref.nested_absmax_ptr();
    const float* nested_quant_map = ref.nested_quant_map_ptr();
    const size_t nested_blocksize = ref.quant_state.nested_blocksize > 0
                                        ? static_cast<size_t>(ref.quant_state.nested_blocksize) : 256;
    const float nested_offset = ref.quant_state.nested_offset;
    size_t i = a0;
    while (i < a1) {
        const size_t global = expert_base + i;
        const float scale = nested_absmax[global / nested_blocksize];
        const size_t run_end = std::min(a1, i + (nested_blocksize - global % nested_blocksize));
        for (; i < run_end; ++i) {
            absmax_f32[i] = nested_quant_map[codes[i]] * scale + nested_offset;
        }
    }
}

} // namespace

void Qwen4ExpExpertCache::load_expert_to_staging(int layer, int expert_id, uint8_t* staging_slot,
                                                 int part, int nparts, uint64_t slot_gen) {
    if (reader_) {
        reader_->read_expert_slice(layer, expert_id, staging_slot, part, nparts, slot_gen);
        return;
    }
    if (layer < 0 || layer >= static_cast<int>(layer_weights_.size()) || !layer_weights_[layer].valid) {
        size_t b0 = 0, b1 = 0;
        part_range(slot_bytes_, part, nparts, b0, b1);
        std::memset(staging_slot + b0, 0, b1 - b0);
        return;
    }
    const auto& lw = layer_weights_[layer];
    fill_expert_matrix(lw.gate_up, expert_id,
                       staging_slot + gu_packed_offset_, gu_packed_bytes_,
                       staging_slot + gu_absmax_offset_, gu_absmax_count_, compact_nested_,
                       part, nparts);
    fill_expert_matrix(lw.down, expert_id,
                       staging_slot + dn_packed_offset_, dn_packed_bytes_,
                       staging_slot + dn_absmax_offset_, dn_absmax_count_, compact_nested_,
                       part, nparts);
}

bool Qwen4ExpExpertCache::pin_expert(int layer, int expert_id) {
    if (frozen_) throw std::logic_error("Legacy pinning cannot change a frozen epoch");
    if (!initialized_ || faulted_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
    const int flat = expert_flat_index(layer, expert_id);
    if (layer_expert_to_slot_[flat] >= 0) {
        int existing_slot = layer_expert_to_slot_[flat];
        if (slot_pinned_[existing_slot]) return true;
        slot_pinned_[existing_slot] = true;
        if (slot_iters_[existing_slot] != lru_list_.end()) {
            lru_list_.erase(slot_iters_[existing_slot]);
            slot_iters_[existing_slot] = lru_list_.end();
        }
        ++num_pinned_slots_;
        return true;
    }
    if (num_pinned_slots_ >= num_slots_) return false;

    int slot = -1;
    if (num_pinned_slots_ < num_slots_ && !slot_pinned_[num_pinned_slots_] && !slot_reserved_[num_pinned_slots_] && !gpu_slot_records_[num_pinned_slots_].direct_leases) {
        slot = num_pinned_slots_;
    } else {
        for (int s = 0; s < num_slots_; ++s) {
            if (!slot_pinned_[s] && !slot_reserved_[s] && !gpu_slot_records_[s].direct_leases) {
                slot = s;
                break;
            }
        }
    }
    if (slot < 0) return false;

    // Evict previous occupant if bound
    const uint64_t old_key = slot_to_key_[slot];
    if (old_key != UINT64_MAX) {
        int old_l = static_cast<int>(old_key >> 32);
        int old_e = static_cast<int>(old_key & 0xFFFFFFFFULL);
        int old_flat = expert_flat_index(old_l, old_e);
        if (old_flat >= 0 && old_flat < static_cast<int>(layer_expert_to_slot_.size())) {
            layer_expert_to_slot_[old_flat] = -1;
        }
    }

    if (gpu_slot_records_[slot].consumer_fence != sycl::event{}) {
        gpu_slot_records_[slot].consumer_fence.wait_and_throw();
        gpu_slot_records_[slot].consumer_fence = sycl::event{};
    }

    load_expert_to_staging(layer, expert_id, pinned_staging_ptr_, 0, 1, 1);
    if (copy_q_) {
        copy_q_->memcpy(slot_dev_base_[slot], pinned_staging_ptr_, slot_bytes_).wait_and_throw();
    } else {
        ctx_->memcpy_h2d(slot_dev_base_[slot], pinned_staging_ptr_, slot_bytes_);
    }
    bind_slot_nested_metadata(slot, layer, expert_id);
    const uint64_t key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert_id);
    slot_to_key_[slot] = key;
    layer_expert_to_slot_[flat] = slot;
    slot_pinned_[slot] = true;
    gpu_slot_records_[slot].key = key;
    gpu_slot_records_[slot].generation++;
    gpu_slot_records_[slot].state = ExpertSlotState::Ready;
    gpu_slot_records_[slot].fill_owner = -1;
    if (slot_iters_[slot] != lru_list_.end()) {
        lru_list_.erase(slot_iters_[slot]);
        slot_iters_[slot] = lru_list_.end();
    }
    ++num_pinned_slots_;
    return true;
}

bool Qwen4ExpExpertCache::is_pinned(int layer, int expert_id) const {
    if (!initialized_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) return false;
    int slot = layer_expert_to_slot_[expert_flat_index(layer, expert_id)];
    return slot >= 0 && slot < num_slots_ && slot < static_cast<int>(slot_pinned_.size()) && slot_pinned_[slot];
}

int Qwen4ExpExpertCache::init_pinned_experts(int experts_per_layer, const std::string& custom_file) {
    if (!initialized_ || faulted_) return 0;
    static const bool s_static_pinning = aila::env::read_flag("AILA_Q4E_STATIC_PINNING", false);
    if (!s_static_pinning && custom_file.empty()) return 0;

    int pin_per_layer = aila::env::read_int("AILA_Q4E_HOT_PIN_PER_LAYER", experts_per_layer);
    if (pin_per_layer <= 0) return 0;
    pin_per_layer = std::min(pin_per_layer, 18);

    // Safety headroom: ensure at least num_experts_per_layer_ + 16 dynamic slots remain for LRU
    const int min_dynamic_slots = std::min(num_experts_per_layer_ + 16, num_slots_ / 2);
    const int max_pinnable = std::max(0, num_slots_ - min_dynamic_slots);
    const int max_per_layer = max_pinnable / std::max(1, num_layers_);
    pin_per_layer = std::min(pin_per_layer, max_per_layer);
    if (pin_per_layer <= 0) return 0;

    std::string file_path = aila::env::read_string("AILA_Q4E_HOT_PIN_FILE", custom_file);
    std::vector<std::pair<int, int>> experts_to_pin;
    experts_to_pin.reserve(static_cast<size_t>(num_layers_) * pin_per_layer);

    if (!file_path.empty()) {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            AILA_LOG_WARN("[Q4E ExpertCache] Could not open custom hot-pin file '%s', falling back to defaults", file_path.c_str());
        } else {
            std::string line;
            int line_num = 0;
            while (std::getline(ifs, line)) {
                ++line_num;
                size_t start = line.find_first_not_of(" \t\r\n");
                if (start == std::string::npos) continue;
                if (line[start] == '#' || line[start] == '/' || line[start] == ';') continue;
                std::istringstream iss(line.substr(start));
                int l = -1, e = -1;
                if (iss >> l >> e) {
                    if (l >= 0 && l < num_layers_ && e >= 0 && e < num_experts_per_layer_) {
                        experts_to_pin.emplace_back(l, e);
                    } else {
                        AILA_LOG_WARN("[Q4E ExpertCache] Out-of-range expert (layer=%d, expert=%d) at line %d in '%s'",
                                      l, e, line_num, file_path.c_str());
                    }
                } else {
                    AILA_LOG_WARN("[Q4E ExpertCache] Malformed line %d in '%s': '%s'",
                                  line_num, file_path.c_str(), line.c_str());
                }
            }
        }
    }
    if (experts_to_pin.empty()) {
        for (int L = 0; L < std::min(num_layers_, aila::kMaxHotPinLayers); ++L) {
            for (int p = 0; p < pin_per_layer; ++p) {
                int e = aila::kQwen4ExpDefaultHotExperts[L][p];
                if (e >= 0 && e < num_experts_per_layer_) {
                    experts_to_pin.emplace_back(L, e);
                }
            }
        }
    }

    // Deduplicate entries and exclude already pinned experts
    std::vector<std::pair<int, int>> dedup_experts;
    dedup_experts.reserve(experts_to_pin.size());
    std::vector<bool> seen(static_cast<size_t>(num_layers_) * num_experts_per_layer_, false);
    for (const auto& item : experts_to_pin) {
        int flat = expert_flat_index(item.first, item.second);
        if (flat >= 0 && flat < static_cast<int>(seen.size()) && !seen[flat]) {
            seen[flat] = true;
            if (!is_pinned(item.first, item.second)) {
                dedup_experts.push_back(item);
            }
        }
    }
    experts_to_pin = std::move(dedup_experts);

    const size_t staging_slots = slot_bytes_ > 0 ? (staging_buffer_bytes_ / slot_bytes_) : 0;
    if (staging_slots == 0) return 0;
    const int chunk_size = std::max(1, static_cast<int>(std::min<size_t>(16, staging_slots)));

    int total_pinned = 0;
    size_t idx = 0;
    while (idx < experts_to_pin.size() && num_pinned_slots_ < max_pinnable) {
        size_t cur_chunk = std::min<size_t>(chunk_size, experts_to_pin.size() - idx);
        cur_chunk = std::min<size_t>(cur_chunk, static_cast<size_t>(max_pinnable - num_pinned_slots_));
        for (size_t c = 0; c < cur_chunk; ++c) {
            const int L = experts_to_pin[idx + c].first;
            const int e = experts_to_pin[idx + c].second;
            const int slot = num_pinned_slots_ + static_cast<int>(c);
            uint8_t* staging = pinned_staging_ptr_ + c * slot_bytes_;
            load_expert_to_staging(L, e, staging, 0, 1, 1);
            if (copy_q_) {
                copy_q_->memcpy(slot_dev_base_[slot], staging, slot_bytes_);
            } else {
                ctx_->memcpy_h2d(slot_dev_base_[slot], staging, slot_bytes_);
            }
        }
        if (copy_q_) {
            copy_q_->wait_and_throw();
        }

        for (size_t c = 0; c < cur_chunk; ++c) {
            const int L = experts_to_pin[idx + c].first;
            const int e = experts_to_pin[idx + c].second;
            const int slot = num_pinned_slots_ + static_cast<int>(c);

            // Evict previous occupant if bound
            const uint64_t old_key = slot_to_key_[slot];
            if (old_key != UINT64_MAX) {
                int old_l = static_cast<int>(old_key >> 32);
                int old_e = static_cast<int>(old_key & 0xFFFFFFFFULL);
                int old_flat = expert_flat_index(old_l, old_e);
                if (old_flat >= 0 && old_flat < static_cast<int>(layer_expert_to_slot_.size())) {
                    layer_expert_to_slot_[old_flat] = -1;
                }
            }

            // Invalidate if expert was resident in another slot
            const int cur_flat = expert_flat_index(L, e);
            const int prev_slot = layer_expert_to_slot_[cur_flat];
            if (prev_slot >= 0 && prev_slot != slot) {
                slot_to_key_[prev_slot] = UINT64_MAX;
            }

            bind_slot_nested_metadata(slot, L, e);
            const uint64_t key = (static_cast<uint64_t>(L) << 32) | static_cast<uint32_t>(e);
            slot_to_key_[slot] = key;
            layer_expert_to_slot_[cur_flat] = slot;
            slot_pinned_[slot] = true;
            gpu_slot_records_[slot].key = key;
            gpu_slot_records_[slot].generation++;
            gpu_slot_records_[slot].state = ExpertSlotState::Ready;
            gpu_slot_records_[slot].fill_owner = -1;

            if (slot_iters_[slot] != lru_list_.end()) {
                lru_list_.erase(slot_iters_[slot]);
                slot_iters_[slot] = lru_list_.end();
            }
            ++total_pinned;
        }
        num_pinned_slots_ += static_cast<int>(cur_chunk);
        idx += cur_chunk;
    }

    AILA_LOG_INFO("[Q4E ExpertCache] Static hot-pinning complete: %d experts pinned in slots 0..%d (%.1f MB, %zu LRU slots remain)",
                  num_pinned_slots_, num_pinned_slots_ - 1,
                  (static_cast<double>(num_pinned_slots_) * slot_bytes_) / (1024.0 * 1024.0),
                  lru_list_.size());
    return total_pinned;
}

void Qwen4ExpExpertCache::log_stats(int layer) const {
    uint64_t total = stats_.hit_count + stats_.miss_count + stats_.cpu_cold_entries;
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
                  << " bytes=" << stats_.total_transferred_bytes
                  << " pf_issued_b=" << stats_.prefetch_issued_bytes
                  << " pf_useful_b=" << stats_.prefetch_useful_bytes
                  << " pf_wasted_b=" << stats_.prefetch_wasted_bytes
                  << " pf_cancelled_b=" << stats_.prefetch_cancelled_bytes
                  << " exposed_wait_ms=" << (stats_.demand_exposed_wait_ns / 1000000.0) << "\n";
    }
}

ExpertSlotState Qwen4ExpExpertCache::get_slot_state(int slot_id) const {
    if (faulted_ || slot_id < 0 || slot_id >= num_slots_) return ExpertSlotState::Faulted;
    if (gpu_slot_records_[slot_id].state == ExpertSlotState::Copying) {
        if (is_event_complete(gpu_slot_records_[slot_id].dma_event)) {
            return ExpertSlotState::Ready;
        }
    }
    return gpu_slot_records_[slot_id].state;
}

uint64_t Qwen4ExpExpertCache::get_slot_generation(int slot_id) const {
    if (slot_id < 0 || slot_id >= num_slots_) return 0;
    return gpu_slot_records_[slot_id].generation;
}

ExpertSlotState Qwen4ExpExpertCache::get_host_slot_state(int hslot) const {
    if (faulted_ || !host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return ExpertSlotState::Faulted;
    return static_cast<ExpertSlotState>(host_slot_state_[hslot].load(std::memory_order_acquire));
}

uint64_t Qwen4ExpExpertCache::get_host_slot_generation(int hslot) const {
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return 0;
    return host_slot_records_[hslot].generation.load(std::memory_order_acquire);
}

void Qwen4ExpExpertCache::record_consumer_reader_fence(int slot_id, sycl::event fence) {
    if (slot_id >= 0 && slot_id < num_slots_) {
        gpu_slot_records_[slot_id].consumer_fence = fence;
    }
}

void Qwen4ExpExpertCache::record_consumer_reader_fences(const int* slot_ids, int count, sycl::event fence) {
    if (!slot_ids || count <= 0) return;
    for (int i = 0; i < count; ++i) {
        int s = slot_ids[i];
        if (s >= 0 && s < num_slots_) {
            gpu_slot_records_[s].consumer_fence = fence;
        }
    }
}

ExpertResidency Qwen4ExpExpertCache::query_residency(int layer, int expert_id) const {
    ExpertResidency res{};
    res.layer = layer;
    res.expert_id = expert_id;
    if (!initialized_ || faulted_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) {
        return res;
    }

    const int flat = expert_flat_index(layer, expert_id);
    const int gslot = layer_expert_to_slot_[flat];
    if (gslot >= 0 && gslot < num_slots_) {
        res.gpu_resident = true;
        res.gpu_slot = gslot;
        res.gpu_generation = gpu_slot_records_[gslot].generation;
        res.gpu_dma_event = gpu_slot_records_[gslot].dma_event;
        const auto st = gpu_slot_records_[gslot].state;
        if (st == ExpertSlotState::Ready) {
            res.gpu_ready = true;
        } else if (st == ExpertSlotState::Copying) {
            if (is_event_complete(res.gpu_dma_event)) {
                res.gpu_ready = true;
            } else {
                res.gpu_copying = true;
            }
        }
    }

    if (host_pool_enabled_) {
        const int hslot = host_layer_expert_to_slot_[flat];
        if (hslot >= 0 && hslot < num_host_slots_) {
            res.host_resident = true;
            res.host_slot = hslot;
            res.host_generation = host_slot_records_[hslot].generation.load(std::memory_order_relaxed);
            res.host_cpu_readers = host_slot_records_[hslot].cpu_readers.load(std::memory_order_relaxed);
            const uint8_t hstate = host_slot_state_[hslot].load(std::memory_order_acquire);
            if (hstate == static_cast<uint8_t>(ExpertSlotState::Ready)) {
                res.host_ready = true;
            } else if (hstate == static_cast<uint8_t>(ExpertSlotState::Filling)) {
                res.host_filling = true;
            }
        }
    }

    return res;
}

void Qwen4ExpExpertCache::release_host_lease(int hslot, uint64_t gen) {
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return;
    const uint64_t cur_gen = host_slot_records_[hslot].generation.load(std::memory_order_acquire);
    if (gen != 0 && cur_gen != gen) {
        return;
    }
    int prev = host_slot_records_[hslot].cpu_readers.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        // Underflow protection
        host_slot_records_[hslot].cpu_readers.store(0, std::memory_order_release);
    }
}

HostExpertLease Qwen4ExpExpertCache::acquire_host_lease(int layer, int expert_id) {
    if (!initialized_ || faulted_ || !host_pool_enabled_ ||
        layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) {
        return {};
    }
    const int flat = expert_flat_index(layer, expert_id);
    const int hslot = host_layer_expert_to_slot_[flat];
    if (hslot < 0 || hslot >= num_host_slots_) return {};

    if (host_slot_state_[hslot].load(std::memory_order_acquire) != static_cast<uint8_t>(ExpertSlotState::Ready)) {
        return {};
    }

    const uint64_t expected_key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert_id);
    const uint64_t gen_before = host_slot_records_[hslot].generation.load(std::memory_order_acquire);
    host_slot_records_[hslot].cpu_readers.fetch_add(1, std::memory_order_acq_rel);

    // Recheck key, state, and generation after incrementing reader count to guard against races
    const uint64_t gen_after = host_slot_records_[hslot].generation.load(std::memory_order_acquire);
    if (gen_after != gen_before ||
        host_slot_to_key_[hslot] != expected_key ||
        host_slot_state_[hslot].load(std::memory_order_acquire) != static_cast<uint8_t>(ExpertSlotState::Ready)) {
        host_slot_records_[hslot].cpu_readers.fetch_sub(1, std::memory_order_release);
        return {};
    }

    touch_host_slot(hslot);
    ops::MoeExpertSlotHost slot_host = get_slot_host(hslot);
    return HostExpertLease(this, layer, expert_id, hslot, gen_after, slot_host);
}

ops::MoeExpertSlotHost Qwen4ExpExpertCache::get_slot_host(int hslot) const {
    if (faulted_) throw std::logic_error("Qwen4ExpExpertCache: cache is poisoned; reinitialize before use");
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return {};

    const uint64_t key = host_slot_to_key_[hslot];
    if (key == UINT64_MAX) return {};
    const int layer = static_cast<int>(key >> 32);
    const int expert_id = static_cast<int>(key & 0xFFFFFFFF);
    if (layer < 0 || layer >= num_layers_ || !layer_weights_[layer].valid) return {};

    const uint8_t* base = host_slot_ptrs_[hslot];
    const auto& lw = layer_weights_[layer];
    ops::MoeExpertSlotHost slot{};

    slot.gate_up_packed = base + gu_packed_offset_;
    slot.down_packed = base + dn_packed_offset_;

    if (compact_nested_) {
        const int gu_block = lw.gate_up.quant_state.nested_blocksize;
        const int dn_block = lw.down.quant_state.nested_blocksize;

        slot.gate_up_nested.codes = base + gu_absmax_offset_;
        slot.gate_up_nested.nested_absmax = lw.gate_up.nested_absmax_ptr() +
            static_cast<size_t>(expert_id) * (gu_absmax_count_ / static_cast<size_t>(gu_block));
        slot.gate_up_nested.nested_quant_map = lw.gate_up.nested_quant_map_ptr();
        slot.gate_up_nested.offset = lw.gate_up.quant_state.nested_offset;
        slot.gate_up_nested.blocksize = gu_block;

        slot.down_nested.codes = base + dn_absmax_offset_;
        slot.down_nested.nested_absmax = lw.down.nested_absmax_ptr() +
            static_cast<size_t>(expert_id) * (dn_absmax_count_ / static_cast<size_t>(dn_block));
        slot.down_nested.nested_quant_map = lw.down.nested_quant_map_ptr();
        slot.down_nested.offset = lw.down.quant_state.nested_offset;
        slot.down_nested.blocksize = dn_block;
    } else {
        slot.gate_up_absmax = reinterpret_cast<const float*>(base + gu_absmax_offset_);
        slot.down_absmax = reinterpret_cast<const float*>(base + dn_absmax_offset_);
    }
    return slot;
}

ops::MoeExpertSlotHost Qwen4ExpExpertCache::get_expert_slot_host(int layer, int expert_id) const {
    if (!initialized_ || faulted_ || !host_pool_enabled_ ||
        layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) {
        return {};
    }
    const int flat = expert_flat_index(layer, expert_id);
    const int hslot = host_layer_expert_to_slot_[flat];
    if (hslot < 0 || hslot >= num_host_slots_) return {};
    return get_slot_host(hslot);
}

FillTicket Qwen4ExpExpertCache::reserve_host_preparation(int layer, int expert_id, bool& needs_fill) {
    needs_fill = false;
    FillTicket ticket{};
    ticket.layer = layer;
    ticket.expert_id = expert_id;
    if (!initialized_ || faulted_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) {
        return ticket;
    }
    if (!host_pool_enabled_) {
        return ticket;
    }
    const int flat = expert_flat_index(layer, expert_id);
    int hslot = host_layer_expert_to_slot_[flat];
    if (hslot >= 0 && hslot < num_host_slots_) {
        const uint8_t st = host_slot_state_[hslot].load(std::memory_order_acquire);
        ticket.hslot = hslot;
        ticket.generation = host_slot_records_[hslot].generation.load(std::memory_order_acquire);
        ticket.ready = (st == static_cast<uint8_t>(ExpertSlotState::Ready));
        if (ticket.ready) {
            touch_host_slot(hslot);
        }
        return ticket;
    }

    int victim_hslot = pick_victim_hslot();
    if (victim_hslot < 0) {
        return ticket;
    }
    const uint64_t old_key = host_slot_to_key_[victim_hslot];
    {
        std::unique_lock<std::mutex> lock;
        if (host_list_mutex_) lock = std::unique_lock<std::mutex>(*host_list_mutex_);
        if (old_key == UINT64_MAX) {
            host_slot_is_protected_[victim_hslot] = false;
            try { host_probationary_list_.push_front(victim_hslot); }
            catch (...) { host_empty_slots_.push_back(victim_hslot); throw; }
            host_slot_iters_[victim_hslot] = host_probationary_list_.begin();
        } else if (host_slot_is_protected_[victim_hslot]) {
            host_slot_is_protected_[victim_hslot] = false;
            host_probationary_list_.splice(host_probationary_list_.begin(),
                                           host_protected_list_, host_slot_iters_[victim_hslot]);
        } else {
            host_probationary_list_.splice(host_probationary_list_.begin(),
                                           host_probationary_list_, host_slot_iters_[victim_hslot]);
        }
    }
    if (old_key != UINT64_MAX) {
        host_layer_expert_to_slot_[expert_flat_index(static_cast<int>(old_key >> 32), static_cast<int>(old_key & 0xFFFFFFFF))] = -1;
    }
    const uint64_t key = (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert_id);
    host_layer_expert_to_slot_[flat] = victim_hslot;
    host_slot_to_key_[victim_hslot] = key;
    const uint64_t next_gen = host_slot_records_[victim_hslot].generation.load(std::memory_order_relaxed) + 1;
    host_slot_records_[victim_hslot].generation.store(next_gen, std::memory_order_release);
    host_slot_records_[victim_hslot].key = key;
    host_slot_records_[victim_hslot].fill_owner = 0;
    host_slot_state_[victim_hslot].store(static_cast<uint8_t>(ExpertSlotState::Filling), std::memory_order_release);
    ticket.hslot = victim_hslot;
    ticket.generation = next_gen;
    needs_fill = true;
    return ticket;
}

void Qwen4ExpExpertCache::invalidate_host_preparation(const FillTicket& ticket) {
    const int h = ticket.hslot;
    if (h < 0 || host_slot_records_[h].generation.load(std::memory_order_acquire) != ticket.generation) return;
    std::unique_lock<std::mutex> lock;
    if (host_list_mutex_) lock = std::unique_lock<std::mutex>(*host_list_mutex_);
    host_layer_expert_to_slot_[expert_flat_index(ticket.layer, ticket.expert_id)] = -1;
    host_slot_to_key_[h] = UINT64_MAX;
    host_slot_records_[h].key = UINT64_MAX;
    host_slot_records_[h].generation.fetch_add(1, std::memory_order_release);
    host_slot_records_[h].fill_owner = -1;
    if (host_slot_is_protected_[h]) host_protected_list_.erase(host_slot_iters_[h]);
    else host_probationary_list_.erase(host_slot_iters_[h]);
    host_slot_is_protected_[h] = false;
    host_slot_reserved_[h] = false;
    host_empty_slots_.push_back(h);
    host_slot_state_[h].store(static_cast<uint8_t>(ExpertSlotState::Empty), std::memory_order_release);
}

FillTicket Qwen4ExpExpertCache::prepare_host(int layer, int expert_id, int /*priority*/) {
    bool needs_fill = false;
    auto ticket = reserve_host_preparation(layer, expert_id, needs_fill);
    if (!needs_fill) return ticket;
    try {
        load_expert_to_staging(layer, expert_id, host_slot_ptrs_[ticket.hslot], 0, 1, ticket.generation);
        host_slot_state_[ticket.hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
        ++stats_.host_pool_misses;
        ticket.ready = true;
        return ticket;
    } catch (...) { invalidate_host_preparation(ticket); throw; }
}

int Qwen4ExpExpertCache::prepare_host_batch(int layer, const int* expert_ids, int count, HostExpertLease* leases, bool wait_for_prefetch) {
    if (!initialized_ || faulted_) throw std::logic_error("Host preparation requires an initialized healthy cache");
    if (count < 0 || count > kMaxHostPrepareTasks || layer < 0 || layer >= num_layers_ ||
        (count > 0 && (!expert_ids || !leases))) throw std::invalid_argument("Invalid host preparation batch");
    for (int c = 0; c < count; ++c) {
        if (expert_ids[c] < 0 || expert_ids[c] >= num_experts_per_layer_ || leases[c].is_valid())
            throw std::invalid_argument("Host preparation requires valid keys and empty output leases");
    }
    if (!host_pool_enabled_ || count == 0) return 0;
    std::array<FillTicket, kMaxHostPrepareTasks> cold;
    std::array<std::exception_ptr, kMaxHostPrepareTasks> errors{};
    int fills = 0;
    try {
        // Protect every ready input before selecting any replacement slots.
        for (int c = 0; c < count; ++c) leases[c] = acquire_host_lease(layer, expert_ids[c]);
        if (wait_for_prefetch) {
            for (int c = 0; c < count; ++c) {
                if (leases[c]) continue;
                for (size_t p = 0; p < prefetch_items_.size(); ++p) {
                    auto& item = prefetch_items_[p];
                    if (item.layer != layer || item.expert_id != expert_ids[c]) continue;
                    ensure_prefetch_item_ready(item, static_cast<int>(p));
                    // The legacy prefetch callback publishes item.ready before
                    // the host state. Wait for that final publication as well.
                    while (item.hslot >= 0 && host_slot_records_[item.hslot].generation.load(std::memory_order_acquire) == item.expected_generation &&
                           host_slot_state_[item.hslot].load(std::memory_order_acquire) == static_cast<uint8_t>(ExpertSlotState::Filling))
                        std::this_thread::yield();
                    leases[c] = acquire_host_lease(layer, expert_ids[c]);
                    break;
                }
            }
        }
        for (int c = 0; c < count; ++c) {
            if (leases[c]) continue;
            bool needs_fill = false;
            const auto ticket = reserve_host_preparation(layer, expert_ids[c], needs_fill);
            if (needs_fill) {
                cold[fills++] = ticket;
                host_slot_reserved_[ticket.hslot] = true;
            }
        }
        // Catch inside the callback: no exception may unwind a pool worker.
        // One complete expert per task amortizes ReadFile and pool scheduling.
        fill_pool_->run_demand(fills, [&](int c) {
            try {
                const auto& ticket = cold[c];
                load_expert_to_staging(layer, ticket.expert_id, host_slot_ptrs_[ticket.hslot], 0, 1, ticket.generation);
            } catch (...) { errors[c] = std::current_exception(); }
        });
        for (int c = 0; c < fills; ++c) if (errors[c]) std::rethrow_exception(errors[c]);
        for (int c = 0; c < fills; ++c)
            host_slot_state_[cold[c].hslot].store(static_cast<uint8_t>(ExpertSlotState::Ready), std::memory_order_release);
        for (int c = 0; c < count; ++c)
            if (!leases[c]) leases[c] = acquire_host_lease(layer, expert_ids[c]);
        for (int c = 0; c < fills; ++c) host_slot_reserved_[cold[c].hslot] = false;
        stats_.host_pool_misses += fills;
        return fills;
    } catch (...) {
        // run_demand returned only after all callbacks drained. Old ready slots
        // remain valid; discard newly filled slots, including partial reads.
        for (int c = 0; c < count; ++c) leases[c].release();
        for (int c = 0; c < fills; ++c) invalidate_host_preparation(cold[c]);
        throw;
    }
}

GpuExpertTicket Qwen4ExpExpertCache::ensure_gpu(int layer, int expert_id, int /*priority*/) {
    GpuExpertTicket ticket{};
    ticket.layer = layer;
    ticket.expert_id = expert_id;
    if (!initialized_ || faulted_ || layer < 0 || layer >= num_layers_ || expert_id < 0 || expert_id >= num_experts_per_layer_) {
        return ticket;
    }
    const int flat = expert_flat_index(layer, expert_id);
    const int gslot = layer_expert_to_slot_[flat];
    if (gslot >= 0 && gslot < num_slots_) {
        ticket.gpu_slot = gslot;
        ticket.gpu_generation = gpu_slot_records_[gslot].generation;
        ticket.dma_event = gpu_slot_records_[gslot].dma_event;
        const auto st = gpu_slot_records_[gslot].state;
        if (st == ExpertSlotState::Ready || is_event_complete(ticket.dma_event)) {
            ticket.ready = true;
        }
        return ticket;
    }

    int out_slot = -1;
    int req = expert_id;
    sycl::event ev = acquire(layer, &req, 1, &out_slot);
    ticket.gpu_slot = out_slot;
    ticket.dma_event = ev;
    ticket.gpu_generation = (out_slot >= 0 && out_slot < num_slots_)
        ? gpu_slot_records_[out_slot].generation : 0;
    ticket.ready = is_event_complete(ev);
    return ticket;
}

GpuExpertTicket Qwen4ExpExpertCache::ensure_gpu(const HostExpertLease& lease, int priority) {
    if (!lease.is_valid()) return {};
    return ensure_gpu(lease.layer(), lease.expert_id(), priority);
}

sycl::event Qwen4ExpExpertCache::get_slot_dma_event(int slot_id) const {
    if (slot_id < 0 || slot_id >= num_slots_) return {};
    return gpu_slot_records_[slot_id].dma_event;
}

const DmaTimingStats& Qwen4ExpExpertCache::dma_timing_stats() {
    update_dma_profiling();
    return dma_timing_stats_;
}

void Qwen4ExpExpertCache::update_dma_profiling() {
    if (dma_timing_stats_.last_memcpy_count > 0 && dma_timing_stats_.last_copy_ns == 0) {
        try {
            if (is_event_complete(last_batch_dma_event_)) {
                uint64_t start = UINT64_MAX;
                uint64_t end = 0;
                // Host completion order is not copy-queue submission order.
                for (const auto& ev : batch_dma_events_) {
                    start = std::min(start, ev.get_profiling_info<sycl::info::event_profiling::command_start>());
                    end = std::max(end, ev.get_profiling_info<sycl::info::event_profiling::command_end>());
                }
                if (end > start) {
                    dma_timing_stats_.last_copy_ns = end - start;
                    dma_timing_stats_.total_copy_ns += (end - start);
                    ++dma_timing_stats_.batches_profiled;
                    dma_timing_stats_.last_effective_gbps =
                        (static_cast<double>(dma_timing_stats_.last_batch_bytes) / static_cast<double>(end - start));
                }
            }
        } catch (...) {
        }
    }
}

void Qwen4ExpExpertCache::record_first_consumer_start() {
    if (!new_dma_batch_pending_ || dma_timing_stats_.last_batch_bytes == 0) return;
    new_dma_batch_pending_ = false;

    const auto now = std::chrono::high_resolution_clock::now();
    uint64_t gap_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - last_batch_submit_end_time_).count());
    dma_timing_stats_.last_consumer_gap_ns = gap_ns;
    dma_timing_stats_.total_consumer_gap_ns += gap_ns;

    static const bool s_prof = aila::env::read_flag("AILA_Q4E_DMA_PROF", false);
    if (s_prof) {
        update_dma_profiling();
        std::cout << "[Q4E DMA Prof] bytes=" << dma_timing_stats_.last_batch_bytes
                  << " count=" << dma_timing_stats_.last_memcpy_count
                  << " submit_ms=" << (dma_timing_stats_.last_submit_ns / 1e6)
                  << " consumer_submit_gap_ms=" << (dma_timing_stats_.last_consumer_gap_ns / 1e6)
                  << " copy_span_rate=" << dma_timing_stats_.last_effective_gbps << " GB/s\n";
    }
}
