#include "Qwen4ExpExpertCache.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <immintrin.h>


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

Qwen4ExpExpertCache::Qwen4ExpExpertCache() = default;

Qwen4ExpExpertCache::~Qwen4ExpExpertCache() {
    cleanup();
}

Qwen4ExpExpertCache::Qwen4ExpExpertCache(Qwen4ExpExpertCache&& other) noexcept {
    *this = std::move(other);
}

Qwen4ExpExpertCache& Qwen4ExpExpertCache::operator=(Qwen4ExpExpertCache&& other) noexcept {
    if (this == &other) return *this;

    try {
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
    slots_dev_ = std::move(other.slots_dev_);
    stats_ = other.stats_;
    initialized_ = other.initialized_;
    fill_pool_ = std::move(other.fill_pool_);
    fill_parts_ = other.fill_parts_;

    host_pool_enabled_ = other.host_pool_enabled_;
    host_pool_bytes_ = other.host_pool_bytes_;
    num_host_slots_ = other.num_host_slots_;
    host_pool_chunks_ = std::move(other.host_pool_chunks_);
    host_slot_ptrs_ = std::move(other.host_slot_ptrs_);
    host_layer_expert_to_slot_ = std::move(other.host_layer_expert_to_slot_);
    host_slot_to_key_ = std::move(other.host_slot_to_key_);
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
    parts_left_scratch_ = std::move(other.parts_left_scratch_);
    parts_left_capacity_ = other.parts_left_capacity_;

    other.ctx_ = nullptr;
    other.weights_ = nullptr;
    other.pinned_staging_ptr_ = nullptr;
    other.prefetch_staging_ptr_ = nullptr;
    other.prefetch_staging_slots_ = 0;
    other.prefetch_in_flight_ = false;
    other.last_dma_event_ = sycl::event{};
    other.initialized_ = false;
    other.num_slots_ = 0;
    other.num_host_slots_ = 0;
    other.host_pool_enabled_ = false;
    other.parts_left_capacity_ = 0;
    other.protected_ratio_ = 0.75f;
    other.stats_ = {};

    return *this;
}

void Qwen4ExpExpertCache::cleanup() {
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
    layer_meta_allocs_.clear();
    slot_dev_base_.clear();
    ctx_ = nullptr;
    weights_ = nullptr;
    dev_pool_bytes_ = 0;
    staging_buffer_bytes_ = 0;
    num_slots_ = 0;
    compact_nested_ = false;
    initialized_ = false;
    layer_weights_.clear();
    lru_list_.clear();
    slot_iters_.clear();
    layer_expert_to_slot_.clear();
    slot_to_key_.clear();
    slot_reserved_.clear();
    slots_dev_.clear();
    host_pool_chunks_.clear();
    host_slot_ptrs_.clear();
    host_layer_expert_to_slot_.clear();
    host_slot_to_key_.clear();
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
                              bool force_nested_codes) {
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
            copy_q_ = std::make_unique<sycl::queue>(
                ctx_->queue().get_context(), ctx_->queue().get_device(),
                sycl::property_list{sycl::property::queue::in_order(),
                                    sycl::ext::intel::property::queue::immediate_command_list()});
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
                host_slot_state_[h].store(0 /* Empty */, std::memory_order_relaxed);
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

    fill_pool_ = std::make_unique<ExpertFillPool>(fill_threads);

    misses_scratch_.clear();
    misses_scratch_.reserve(num_experts_per_layer_);
    batch_reserved_slots_scratch_.clear();
    batch_reserved_slots_scratch_.reserve(num_experts_per_layer_);
    to_fill_scratch_.clear();
    to_fill_scratch_.reserve(num_experts_per_layer_);
    assigned_hslots_scratch_.clear();
    assigned_hslots_scratch_.reserve(std::max(1, num_host_slots_));
    parts_left_capacity_ = std::max(static_cast<size_t>(num_experts_per_layer_), static_cast<size_t>(std::max(1, num_host_slots_)));
    parts_left_scratch_ = std::make_unique<std::atomic<int>[]>(parts_left_capacity_);
    prefetch_items_.reserve(std::max(prefetch_staging_slots_, num_host_slots_));

    AILA_LOG_INFO("[Q4E ExpertCache] slots=%d slot_bytes=%zu pool=%.1fMB nested_codes=%d fill_threads=%d fill_parts=%d",
                  num_slots_, slot_bytes_, dev_pool_bytes_ / (1024.0 * 1024.0),
                  compact_nested_ ? 1 : 0, fill_threads, fill_parts_);

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
    if (compact_nested_) configure_layer_nested_metadata(layer);
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
    auto is_event_complete = [](const sycl::event& ev) -> bool {
        if (ev == sycl::event{}) return true;
        try {
            return ev.get_info<sycl::info::event::command_execution_status>() ==
                   sycl::info::event_command_status::complete;
        } catch (...) {
            return false;
        }
    };

    // 1. Try unreserved empty slots first
    for (auto it = host_empty_slots_.rbegin(); it != host_empty_slots_.rend(); ++it) {
        int h = *it;
        if (h >= 0 && h < num_host_slots_ && !host_slot_reserved_[h]) {
            host_empty_slots_.erase(std::next(it).base());
            return h;
        }
    }
    // 2. Try unreserved probationary slots whose DMA is already complete
    for (auto r_it = host_probationary_list_.rbegin(); r_it != host_probationary_list_.rend(); ++r_it) {
        int h = *r_it;
        if (!host_slot_reserved_[h] && is_event_complete(host_slot_dma_events_[h])) {
            return h;
        }
    }
    // 3. Try unreserved protected slots whose DMA is already complete
    for (auto r_it = host_protected_list_.rbegin(); r_it != host_protected_list_.rend(); ++r_it) {
        int h = *r_it;
        if (!host_slot_reserved_[h] && is_event_complete(host_slot_dma_events_[h])) {
            return h;
        }
    }
    // 4. If all unreserved candidates have in-flight DMAs, pick LRU unreserved slot and wait on its DMA
    for (auto r_it = host_probationary_list_.rbegin(); r_it != host_probationary_list_.rend(); ++r_it) {
        int h = *r_it;
        if (!host_slot_reserved_[h]) {
            if (host_slot_dma_events_[h] != sycl::event{}) {
                host_slot_dma_events_[h].wait_and_throw();
                host_slot_dma_events_[h] = sycl::event{};
            }
            return h;
        }
    }
    for (auto r_it = host_protected_list_.rbegin(); r_it != host_protected_list_.rend(); ++r_it) {
        int h = *r_it;
        if (!host_slot_reserved_[h]) {
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
    if (host_slot_dma_events_[hslot] == sycl::event{}) return false;
    try {
        return host_slot_dma_events_[hslot].get_info<sycl::info::event::command_execution_status>() !=
               sycl::info::event_command_status::complete;
    } catch (...) {
        return false;
    }
}

void Qwen4ExpExpertCache::touch_host_slot(int hslot) {
    if (!host_pool_enabled_ || hslot < 0 || hslot >= num_host_slots_) return;
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
                load_expert_to_staging(item.layer, item.expert_id, dst, p, parts);
                if (item.parts_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    item.ready.store(true, std::memory_order_release);
                    if (item.hslot >= 0) {
                        host_slot_state_[item.hslot].store(2 /* Ready */, std::memory_order_release);
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

    stats_.prefetch_wait_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t_start).count());
}

void Qwen4ExpExpertCache::cancel_prefetch() {
    if (!prefetch_in_flight_ && prefetch_items_.empty()) return;

    if (prefetch_in_flight_ && fill_pool_) {
        fill_pool_->cancel();
    }

    if (host_pool_enabled_) {
        for (auto& item : prefetch_items_) {
            if (item.hslot >= 0 && item.hslot < num_host_slots_) {
                const int h = item.hslot;
                const bool in_batch = host_slot_in_batch_readers_[h];
                if (!in_batch) {
                    host_slot_reserved_[h] = false;
                }
                if (!item.ready.load(std::memory_order_acquire)) {
                    const int flat = expert_flat_index(item.layer, item.expert_id);
                    if (host_layer_expert_to_slot_[flat] == h) {
                        host_layer_expert_to_slot_[flat] = -1;
                    }
                    host_slot_to_key_[h] = UINT64_MAX;
                    host_slot_state_[h].store(0 /* Empty */, std::memory_order_release);
                    if (host_slot_is_protected_[h]) {
                        host_protected_list_.erase(host_slot_iters_[h]);
                    } else {
                        host_probationary_list_.erase(host_slot_iters_[h]);
                    }
                    host_slot_is_protected_[h] = false;
                    host_empty_slots_.push_back(h);
                } else {
                    host_slot_state_[h].store(2 /* Ready */, std::memory_order_release);
                }
            }
        }
    }

    prefetch_items_.clear();
    prefetch_in_flight_ = false;
}

sycl::event Qwen4ExpExpertCache::acquire(int layer,
                                         const int* expert_ids,
                                         int count,
                                         int* out_slots) {
    if (!initialized_ || count <= 0 || expert_ids == nullptr || out_slots == nullptr) {
        return {};
    }
    // AILA_DEBUG_Q4E_ROUTE_TRACE: append int32 {layer, count, ids[count]} per call, for offline
    // replacement-policy simulation (perf/qwen4exp/q4e_cache_sim.py).
    static FILE* s_trace = [] {
        const std::string path = aila::env::read_string("AILA_DEBUG_Q4E_ROUTE_TRACE", "");
        return path.empty() ? nullptr : std::fopen(path.c_str(), "wb");
    }();
    if (s_trace) {
        const int header[2] = {layer, count};
        std::fwrite(header, sizeof(int), 2, s_trace);
        std::fwrite(expert_ids, sizeof(int), static_cast<size_t>(count), s_trace);
        std::fflush(s_trace);
    }
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
        ~AcquireGuard() {
            if (!committed && self) {
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
    } guard{this, false};

    // Phase 1: check hits and deduplicate within this batch
    int hit_count_in_batch = 0;
    for (int i = 0; i < count; ++i) {
        int expert = expert_ids[i];
        int flat_idx = expert_flat_index(layer, expert);
        int slot = layer_expert_to_slot_[flat_idx];
        if (slot >= 0) {
            out_slots[i] = slot;
            if (!slot_reserved_[slot]) {
                slot_reserved_[slot] = true;
                batch_reserved_slots_scratch_.push_back(slot);
                lru_list_.splice(lru_list_.begin(), lru_list_, slot_iters_[slot]);
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
        return {};
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
            for (int s : batch_reserved_slots_scratch_) slot_reserved_[s] = false;
            for (int h : batch_reserved_hslots_scratch_) host_slot_reserved_[h] = false;
            throw std::runtime_error("Qwen4ExpExpertCache: all slots reserved in batch, count exceeds num_slots");
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

        // Move victim to MRU (front) and reserve for this batch
        lru_list_.splice(lru_list_.begin(), lru_list_, slot_iters_[victim_slot]);

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
        auto copy_to_slot = [&](int slot_id, const uint8_t* src) {
            copy_q_->submit([&](sycl::handler& cgh) {
                cgh.memcpy(slot_dev_base_[slot_id], src, slot_bytes_);
            });
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
                copy_to_slot(miss.slot_id, src);
                stats_.total_transferred_bytes += slot_bytes_;
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
                if (hslot >= 0 && host_slot_state_[hslot].load(std::memory_order_acquire) == 2 /* Ready */) {
                    touch_host_slot(hslot);
                    add_batch_reader_hslot(hslot);
                    const uint8_t* src = host_slot_ptrs_[hslot];
                    copy_to_slot(miss.slot_id, src);
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

                for (size_t c = 0; c < chunk_size; ++c) {
                    const auto& miss = to_fill_scratch_[m + c];
                    int victim_hslot = pick_victim_hslot();
                    if (victim_hslot < 0) {
                        copy_q_->wait_and_throw();
                        for (int h : batch_reserved_hslots_scratch_) {
                            host_slot_reserved_[h] = false;
                            host_slot_in_batch_readers_[h] = false;
                            host_slot_dma_events_[h] = sycl::event{};
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
                    host_slot_state_[victim_hslot].store(1 /* Filling */, std::memory_order_release);

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
                    parts_left_scratch_[c].store(parts, std::memory_order_relaxed);
                    ++stats_.host_pool_misses;
                }

                const std::function<void(int)> fill_task = [&](int task) {
                    const int c = task / parts;
                    const auto& miss = to_fill_scratch_[m + c];
                    int hslot = assigned_hslots_scratch_[c];
                    uint8_t* host_slot_ptr = host_slot_ptrs_[hslot];
                    load_expert_to_staging(miss.layer, miss.expert_id, host_slot_ptr, task % parts, parts);
                    if (parts_left_scratch_[c].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        host_slot_state_[hslot].store(2 /* Ready */, std::memory_order_release);
                        copy_to_slot(miss.slot_id, host_slot_ptr);
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
                    }
                    batch_reserved_hslots_scratch_.clear();
                }
            }
            stats_.fill_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
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
                    load_expert_to_staging(miss.layer, miss.expert_id, staging_slot, task % parts, parts);
                    if (parts_left_scratch_[c].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        copy_to_slot(miss.slot_id, staging_slot);
                    }
                };
                fill_pool_->run(chunk_size * parts, fill_task);
                stats_.total_transferred_bytes += static_cast<uint64_t>(chunk_size) * slot_bytes_;
                stats_.fill_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());

                half_done[half] = copy_q_->ext_oneapi_submit_barrier();
                half_busy[half] = true;
                m += static_cast<size_t>(chunk_size);
            }
        } // end of else if (!to_fill.empty())

        // Capture completion of all DMA operations on copy_q_ with a SYCL hardware barrier.
        // The GPU compute queue establishes a hardware dependency on this event (cgh.depends_on),
        // completely eliminating CPU thread blocking and idle GPU wait.
        if (copy_q_) {
            dma_done_event = copy_q_->ext_oneapi_submit_barrier();
            last_dma_event_ = dma_done_event;
            for (int h : batch_reserved_hslots_scratch_) {
                host_slot_dma_events_[h] = dma_done_event;
            }
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

    return dma_done_event;
}

void Qwen4ExpExpertCache::prefetch_async(int layer, const int* expert_ids, int count) {
    if (!initialized_ || count <= 0 || expert_ids == nullptr) {
        return;
    }
    static const bool s_fake_hits = aila::env::read_flag("AILA_DEBUG_Q4E_FAKE_HITS", false);
    if (s_fake_hits) return; // acquire() binds nothing in this mode

    cancel_prefetch();

    prefetch_items_.clear();
    const int max_slots = host_pool_enabled_ ? num_host_slots_ : prefetch_staging_slots_;
    const int parts = fill_parts_;
    for (int i = 0; i < count && static_cast<int>(prefetch_items_.size()) < max_slots; ++i) {
        const int expert = expert_ids[i];
        if (expert < 0 || expert >= num_experts_per_layer_ || contains(layer, expert)) continue;
        if (host_pool_enabled_) {
            int hflat = expert_flat_index(layer, expert);
            int cur_hslot = host_layer_expert_to_slot_[hflat];
            if (cur_hslot >= 0 && host_slot_state_[cur_hslot].load(std::memory_order_acquire) == 2 /* Ready */) continue;
        }
        bool dup = false;
        for (const auto& item : prefetch_items_) {
            if (item.expert_id == expert) { dup = true; break; }
        }
        if (!dup) prefetch_items_.emplace_back(layer, expert, -1, parts);
    }
    if (prefetch_items_.empty()) return;

    const int n = static_cast<int>(prefetch_items_.size());
    stats_.prefetch_issued += static_cast<uint64_t>(n);
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
            host_slot_state_[victim_hslot].store(1 /* Filling */, std::memory_order_release);

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
                load_expert_to_staging(item.layer, item.expert_id, dst, p, parts);
                if (item.parts_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    item.ready.store(true, std::memory_order_release);
                    if (item.hslot >= 0) {
                        host_slot_state_[item.hslot].store(2 /* Ready */, std::memory_order_release);
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
                    host_slot_state_[item.hslot].store(2 /* Ready */, std::memory_order_release);
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
    for (int i = 0; i < count; ++i) {
        int e = expert_ids[i];
        if (e < 0 || e >= num_experts_per_layer_) continue;
        int slot = layer_expert_to_slot_[expert_flat_index(layer, e)];
        if (slot >= 0) {
            lru_list_.splice(lru_list_.end(), lru_list_, slot_iters_[slot]);
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
                                                 int part, int nparts) {
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
