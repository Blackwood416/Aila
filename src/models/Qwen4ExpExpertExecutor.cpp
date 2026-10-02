#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <array>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <tbb/task_arena.h>

namespace aila {
namespace {
uint64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
void compute_expert(const ops::MoeExpertSlotHost& slot, const bf16* x, bf16* h, float* out, int hidden, int intermediate) {
    static const bool gpu_order = aila::env::read_flag("AILA_Q4E_CPU_GPU_ORDER", false);
    ops::cpu::cpu_expert_forward(slot, x, out, 1, h, nullptr, 1.0f, false, hidden, intermediate, hidden, 64,
        gpu_order ? ops::cpu::CpuKernelBackend::GpuOrderAvx2 : ops::cpu::CpuKernelBackend::Auto);
}
}

// One execution slot, no queued backlog. Only the coordinator owns cache leases.
struct Qwen4ExpExpertExecutor::CpuWorker {
    struct Job {
        std::array<ops::cpu::CpuExpertTask, ops::cpu::kMaxExpertBatchTasks> tasks;
        int count = 0, hidden = 0, intermediate = 0, tokens = 0;
        const bf16* x = nullptr;
        bf16* h = nullptr;
        float* out = nullptr;
        sycl::event activation;
        bool resident_prepared = false;
    };
    struct Result { uint64_t activation_ns = 0, begin_ns = 0, end_ns = 0; std::exception_ptr error; };
    std::mutex mutex;
    std::condition_variable cv;
    bool pending = false, busy = false, done = false, stop = false;
    Job job;
    Result result;
    tbb::task_arena arena;
    bool batched;
    std::thread thread;

    explicit CpuWorker(int threads) : arena(threads), batched(aila::env::read_flag("AILA_Q4E_CPU_BATCH", false)), thread([this] { run(); }) {}
    ~CpuWorker() {
        drain();
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        cv.notify_one();
        thread.join();
    }
    void submit(const Job& next) {
        std::lock_guard<std::mutex> lock(mutex);
        if (busy) throw std::logic_error("CPU overlap slot still busy");
        job = next; result = {}; busy = pending = true; done = false;
        cv.notify_one();
    }
    bool complete() { std::lock_guard<std::mutex> lock(mutex); return busy && done; }
    Result finish() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!busy) return {};
        cv.wait(lock, [&] { return done; });
        auto completed = result;
        busy = false;
        if (completed.error) std::rethrow_exception(completed.error);
        return completed;
    }
    void drain() noexcept { try { finish(); } catch (...) {} }
    void run() {
        for (;;) {
            Job current;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return pending || stop; });
                if (stop) return;
                current = job; pending = false;
            }
            Result completed;
            try {
                const auto begin = steady_ns();
                current.activation.wait_and_throw();
                completed.activation_ns = steady_ns() - begin;
                completed.begin_ns = steady_ns();
                arena.execute([&] {
                    if (batched || current.resident_prepared) {
                        if (current.hidden != ops::cpu::kHiddenDim || current.intermediate != ops::cpu::kIntermediateDim)
                            throw std::invalid_argument("CPU batch requires real Q4E geometry");
                        static const bool gpu_order = aila::env::read_flag("AILA_Q4E_CPU_GPU_ORDER", false);
                        static const bool prepared = aila::env::read_flag("AILA_Q4E_CPU_PREPARED_INPUT", false);
                        ops::cpu::cpu_expert_batch_forward(current.tasks.data(), current.count, current.x, current.tokens, current.h, current.out, nullptr,
                            current.resident_prepared ? ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2 :
                            gpu_order ? (prepared ? ops::cpu::CpuKernelBackend::GpuOrderPreparedAvx2 : ops::cpu::CpuKernelBackend::GpuOrderAvx2)
                                      : ops::cpu::CpuKernelBackend::Avx2);
                    } else for (int c = 0; c < current.count; ++c)
                        compute_expert(current.tasks[c].slot, current.x + current.tasks[c].token * current.hidden,
                            current.h + c * current.intermediate, current.out + c * current.hidden,
                            current.hidden, current.intermediate);
                });
                completed.end_ns = steady_ns();
            } catch (...) { completed.error = std::current_exception(); }
            { std::lock_guard<std::mutex> lock(mutex); result = completed; done = true; }
            cv.notify_one();
        }
    }
};

Qwen4ExpExpertExecutor::Qwen4ExpExpertExecutor() = default;
bool Qwen4ExpExpertExecutor::cpu_work_complete_for_testing() const {
    return cpu_worker_ && cpu_worker_->complete();
}

Qwen4ExpExpertExecutor::~Qwen4ExpExpertExecutor() {
    cleanup();
}

void Qwen4ExpExpertExecutor::cleanup() {
    cpu_worker_.reset();
    if (ctx_) {
        try { ctx_->synchronize(); } catch (...) {}
    }
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
    max_cpu_credits_ = std::clamp(env_max_cpu, 0, ops::cpu::kMaxExpertBatchTasks);
    cpu_placement_ = std::clamp(aila::env::read_int("AILA_Q4E_CPU_PLACEMENT", 0), 0, 2);
    cpu_fixed_credits_ = aila::env::read_flag("AILA_Q4E_CPU_FIXED_CREDITS", false);
    cpu_batch_prepare_ = aila::env::read_flag("AILA_Q4E_CPU_BATCH_PREPARE", false);
    cpu_wait_prefetch_ = aila::env::read_flag("AILA_Q4E_CPU_WAIT_PREFETCH", false);
    cpu_coalesce_results_ = aila::env::read_flag("AILA_Q4E_CPU_COALESCE_RESULTS", false);
    const int env_max_tokens = aila::env::read_int("AILA_Q4E_CPU_MAX_TOKENS", 4);
    max_cpu_tokens_ = std::max(1, std::min(4, env_max_tokens));

    const size_t max_entries = static_cast<size_t>(max_seq_len_) * top_k_;
    active_plan_.entries.reserve(max_entries);
    active_plan_.ready_gpu_indices.reserve(max_entries);
    active_plan_.pending_gpu_indices.reserve(max_entries);
    active_plan_.cpu_indices.reserve(max_entries);
    active_plan_.top_k = top_k_;

    active_cpu_leases_.reserve(ops::cpu::kMaxExpertBatchTasks);

    // Preallocate pinned host and device buffers
    x_host_ = static_cast<bf16*>(ctx_->alloc_host(4ULL * hidden_size_ * sizeof(bf16)));
    cpu_out_host_ = static_cast<float*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * hidden_size_ * sizeof(float)));
    scratch_h_ = static_cast<bf16*>(ctx_->alloc_host(static_cast<size_t>(ops::cpu::kMaxExpertBatchTasks) * intermediate_size_ * sizeof(bf16)));

    const size_t max_gpu_tasks = static_cast<size_t>(max_seq_len_) * top_k_;
    gpu_slots_host_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_host(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_slots_dev_ = static_cast<ops::MoeExpertSlotDev*>(ctx_->alloc_device(max_gpu_tasks * sizeof(ops::MoeExpertSlotDev)));
    gpu_ranks_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_ranks_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    gpu_tokens_host_ = static_cast<int*>(ctx_->alloc_host(max_gpu_tasks * sizeof(int)));
    gpu_tokens_dev_ = static_cast<int*>(ctx_->alloc_device(max_gpu_tasks * sizeof(int)));
    weights_dev_ = static_cast<float*>(ctx_->alloc_device(max_gpu_tasks * sizeof(float)));

    const size_t max_down_acc = 4ULL * top_k_ * hidden_size_;
    down_acc_dev_ = static_cast<float*>(ctx_->alloc_device(max_down_acc * sizeof(float)));

    const size_t max_pending = static_cast<size_t>(max_seq_len_) * top_k_;
    pending_experts_buf_.assign(max_pending, 0);
    assigned_slots_buf_.assign(max_pending, -1);
    prev_cpu_h2d_ev_ = sycl::event{};

    stats_.reset();
    slot_seen_.assign(cache_->num_slots(), false);
    straggler_penalty_active_ = false;
    consecutive_stragglers_ = 0;
    cpu_retry_countdown_ = 0;
    cold_budget_ns_ = 1000.0 * std::clamp(aila::env::read_int("AILA_Q4E_COLD_BUDGET_US",1000),100,10000);
    for (int n=0;n<=16;++n) cold_cost_ns_[n]=100000.0+400000.0*n;
    if (aila::env::read_flag("AILA_Q4E_CPU_OVERLAP", false) || aila::env::read_int("AILA_Q4E_COLD_MODE",0)>0) {
        const int threads = std::clamp(aila::env::read_int("AILA_Q4E_CPU_THREADS", 2), 1, 12);
        cpu_worker_ = std::make_unique<CpuWorker>(threads);
    }
    initialized_ = true;
}

void Qwen4ExpExpertExecutor::StageResidentInput(const bf16* x_dev) {
    if(cold_count_)throw std::logic_error("CPU cold slot has an unfinished batch");
    if(resident_input_event_!=sycl::event{})resident_input_event_.wait_and_throw();
    resident_input_event_=ctx_->queue().memcpy(x_host_,x_dev,size_t(hidden_size_)*sizeof(bf16));
}

int Qwen4ExpExpertExecutor::StartResidentCold(GpuReadyLease& route, const bf16* x_dev) {
    if (!cpu_worker_ || !cache_->frozen() || route.cold_count>16 || cold_count_)
        throw std::logic_error("Invalid resident CPU cold service");
    if (prev_cpu_h2d_ev_ != sycl::event{}) prev_cpu_h2d_ev_.wait_and_throw();
    prev_cpu_h2d_ev_={}; active_cpu_leases_.clear(); cold_layer_=route.layer;
    int gpu_count=0;
    try {
        std::array<HostExpertLease,16> chosen;
        int selected=0;
        static const bool low_rank=aila::env::read_flag("AILA_Q4E_COLD_LOW_RANK",false);
        for(int scan=0;scan<route.cold_count;++scan) {
            const int i=low_rank?route.cold_count-1-scan:scan;
            const int e=route.cold_ids[i];
            const auto residency=cache_->query_residency(route.layer,e);
            if (!residency.gpu_ready && !residency.gpu_copying && residency.host_ready) {
                ++stats_.cold_eligible;
                if(selected<std::min(16,max_cpu_credits_) && cold_cost_ns_[selected+1]<=cold_budget_ns_) {
                    auto lease=cache_->acquire_host_lease(route.layer,e);
                    if(lease.is_valid()) {
                        chosen[i]=std::move(lease);++selected;
                    }
                } else ++stats_.cold_budget_rejected;
            }
        }
        for(int i=0;i<route.cold_count;++i) {
            const int e=route.cold_ids[i],rank=route.cold_ranks[i];
            if(chosen[i].is_valid()) {
                cold_ranks_[cold_count_]=rank;cold_ids_[cold_count_++]=e;
                active_cpu_leases_.push_back(std::move(chosen[i]));
            } else {route.cold_ids[gpu_count]=e;route.cold_ranks[gpu_count++]=rank;}
        }
        route.cold_count=gpu_count;
        if(!cold_count_) return 0;
        CpuWorker::Job job;job.count=cold_count_;job.hidden=hidden_size_;job.intermediate=intermediate_size_;job.tokens=1;
        job.x=x_host_;job.h=scratch_h_;job.out=cpu_out_host_;job.resident_prepared=true;
        job.activation=resident_input_event_!=sycl::event{} ? resident_input_event_ :
            ctx_->queue().memcpy(x_host_,x_dev,size_t(hidden_size_)*sizeof(bf16));
        for(int c=0;c<cold_count_;++c)job.tasks[c]={active_cpu_leases_[c].slot_host(),0};
        cold_submit_ns_=steady_ns();cpu_worker_->submit(job);
        cache_->TraceColdCpu(cold_layer_,cold_ids_.data(),cold_ranks_.data(),cold_count_);
        stats_.cpu_tasks_dispatched+=cold_count_;
        if(cache_->tracing())cache_->trace_host_interval("cold_cpu_submit",cold_layer_,cold_submit_ns_,steady_ns());
        return cold_count_;
    } catch(...) {AbortResidentCold();throw;}
}

sycl::event Qwen4ExpExpertExecutor::JoinResidentCold(float* output) {
    if(!cold_count_)return {};
    const auto join=steady_ns();
    const auto result=cpu_worker_->finish();
    stats_.cpu_join_wait_ns+=steady_ns()-join;
    stats_.cpu_activation_wait_ns+=result.activation_ns;
    stats_.cpu_compute_ns+=result.end_ns-result.begin_ns;
    const auto completion_ns=result.end_ns-cold_submit_ns_;
    stats_.cold_queue_ns+=(result.begin_ns-cold_submit_ns_)-std::min(result.activation_ns,result.begin_ns-cold_submit_ns_);
    cold_cost_ns_[cold_count_]=0.8*cold_cost_ns_[cold_count_]+0.2*double(completion_ns);
    if(cache_->tracing()) {
        cache_->trace_host_interval("cold_cpu_compute",cold_layer_,result.begin_ns,result.end_ns);
        cache_->trace_host_interval("cold_cpu_join",cold_layer_,join,steady_ns());
        cache_->TraceDeviceEvent("cold_input_copy",cold_layer_,resident_input_event_);
    }
    for(int c=0;c<cold_count_;++c)if(cpu_observer_) {
        RouteRankEntry entry;entry.rank=cold_ranks_[c];entry.expert_id=cold_ids_[c];entry.is_gpu=false;
        cpu_observer_(cpu_observer_state_,*ctx_,cold_layer_,entry,active_cpu_leases_[c].slot_host(),x_host_,
            scratch_h_+size_t(c)*intermediate_size_,cpu_out_host_+size_t(c)*hidden_size_);
    }
    for(int c=0;c<cold_count_;) {
        int end=c+1;while(end<cold_count_ && cold_ranks_[end]==cold_ranks_[end-1]+1)++end;
        const size_t bytes=size_t(end-c)*hidden_size_*sizeof(float);
        prev_cpu_h2d_ev_=ctx_->queue().memcpy(output+size_t(cold_ranks_[c])*hidden_size_,cpu_out_host_+size_t(c)*hidden_size_,bytes);
        ++stats_.cpu_result_copies;stats_.cpu_result_bytes+=bytes;c=end;
    }
    stats_.cpu_tasks_completed+=cold_count_;++stats_.cpu_async_layers;
    stats_.cpu_max_batch_tasks=std::max(stats_.cpu_max_batch_tasks,uint64_t(cold_count_));
    active_cpu_leases_.clear();cold_count_=0;
    return prev_cpu_h2d_ev_;
}

void Qwen4ExpExpertExecutor::AbortResidentCold() noexcept {
    if(cpu_worker_)cpu_worker_->drain();
    try {if(ctx_)ctx_->synchronize();}catch(...){}
    active_cpu_leases_.clear();cold_count_=0;
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
    if (token_count > max_seq_len_ || top_k <= 0 || top_k > top_k_) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: route exceeds execution capacity");
    }

    const int total = token_count * top_k;
    active_plan_.total_tasks = total;

    // H3: Determine effective CPU credits for this layer
    int effective_credits = !cpu_fixed_credits_ && straggler_penalty_active_ ? std::max(0, max_cpu_credits_ - 1) : max_cpu_credits_;
    // A one-credit policy must not remain disabled forever after warmup.
    if (!cpu_fixed_credits_ && enable_hetero && token_count <= max_cpu_tokens_ && straggler_penalty_active_ && max_cpu_credits_ == 1) {
        if (cpu_retry_countdown_ > 0) --cpu_retry_countdown_;
        else { effective_credits = 1; cpu_retry_countdown_ = 8; }
    }
    const bool allow_cpu_dispatch = enable_hetero && (token_count <= max_cpu_tokens_) && (effective_credits > 0);
    const bool batch_prepare = allow_cpu_dispatch && cpu_placement_ == 2 && cpu_batch_prepare_;
    static_assert(ops::cpu::kMaxExpertBatchTasks <= Qwen4ExpExpertCache::kMaxHostPrepareTasks);
    std::array<HostExpertLease, ops::cpu::kMaxExpertBatchTasks> prepared_leases;
    std::array<int, ops::cpu::kMaxExpertBatchTasks> prepared_ids{}, prepared_ranks{};
    int prepared_count = 0;
    if (batch_prepare) {
        for (int idx = 0; idx < total && prepared_count < effective_credits; ++idx) {
            const auto res = cache_->query_residency(layer, router_host_ids[idx]);
            if (!cpu_wait_prefetch_ && (res.gpu_copying || (!res.host_ready && res.host_filling))) continue;
            prepared_ids[prepared_count] = router_host_ids[idx];
            prepared_ranks[prepared_count++] = idx;
        }
        if (prepared_count > 0) {
            const auto begin = steady_ns();
            stats_.cpu_host_prepared_experts += cache_->prepare_host_batch(layer, prepared_ids.data(), prepared_count, prepared_leases.data(), cpu_wait_prefetch_);
            stats_.cpu_host_prepare_ns += steady_ns() - begin;
            ++stats_.cpu_host_prepare_batches;
        }
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

            auto res = cache_->query_residency(layer, expert_id);
            const bool cpu_credit = allow_cpu_dispatch && active_plan_.cpu_tasks < effective_credits;
            if (!batch_prepare && cpu_credit && cpu_placement_ == 2 && !res.host_ready && !res.host_filling && !res.gpu_copying) {
                const auto begin = steady_ns();
                cache_->prepare_host(layer, expert_id);
                stats_.cpu_host_prepare_ns += steady_ns() - begin;
                res = cache_->query_residency(layer, expert_id);
            }
            int prepared_idx = -1;
            if (batch_prepare) {
                for (int c = 0; c < prepared_count; ++c) if (prepared_ranks[c] == idx) { prepared_idx = c; break; }
            }
            const bool choose_cpu = batch_prepare
                ? prepared_idx >= 0 && prepared_leases[prepared_idx].is_valid()
                : cpu_credit && res.host_ready && !res.gpu_copying && (cpu_placement_ > 0 || !res.gpu_ready);
            if (res.gpu_ready && !choose_cpu) {
                // Hit in GPU cache: always keep on GPU (zero PCIe transfer needed)
                entry.is_gpu = true;
                entry.gpu_slot = res.gpu_slot;
                entry.gpu_generation = res.gpu_generation;
                active_plan_.entries.push_back(entry);
                active_plan_.ready_gpu_indices.push_back(idx);
                ++active_plan_.gpu_tasks;
                ++active_plan_.ready_gpu_tasks;
            } else if (choose_cpu) {
                // GPU miss, but Host-Ready and within CPU admission credits!
                // Attempt to acquire thread-safe host lease
                auto lease = batch_prepare ? std::move(prepared_leases[prepared_idx]) : cache_->acquire_host_lease(layer, expert_id);
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
    if (!cache_ || plan.gpu_tasks == 0) {
        return;
    }
    const auto acquire_start = std::chrono::steady_clock::now();
    for (int idx : plan.ready_gpu_indices) {
        const auto& entry = plan.entries[idx];
        const auto res = cache_->query_residency(plan.layer, entry.expert_id);
        if (!res.gpu_ready || res.gpu_slot != entry.gpu_slot ||
            res.gpu_generation != entry.gpu_generation) {
            throw std::logic_error("Qwen4ExpExpertExecutor: stale ready expert plan");
        }
    }

    int* pending_experts = pending_experts_buf_.data();
    int* assigned_slots = assigned_slots_buf_.data();

    int num_pending = 0;
    for (const auto& entry : plan.entries) {
        if (!entry.is_gpu) continue;
        int eid = entry.expert_id;
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

    acquire_begin_ns_ = steady_ns();
    sycl::event dma_ev = cache_->acquire(plan.layer, pending_experts,
                                         num_pending,
                                         assigned_slots);
    acquire_end_ns_ = steady_ns();
    plan.final_dma_event = dma_ev;

    // Include ready keys in acquire's reservation set, even when the cache is full.
    for (auto& entry : plan.entries) {
        if (!entry.is_gpu) continue;
        int eid = entry.expert_id;
        for (int i = 0; i < num_pending; ++i) {
            if (pending_experts[i] == eid) {
                entry.gpu_slot = assigned_slots[i];
                entry.gpu_generation = cache_->get_slot_generation(assigned_slots[i]);
                entry.dma_event = cache_->get_slot_dma_event(assigned_slots[i]);
                break;
            }
        }
    }
    stats_.gpu_acquire_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - acquire_start).count();
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
    bool uses_nested,
    bool defer_gpu_acquire,
    const ExpertPrefetchRequest* next_prefetch) {
    if (!initialized_) throw std::logic_error("Qwen4ExpExpertExecutor: execution before initialization");
    // CPU admission is independent of the preallocated GPU/merge capacity.
    if (token_count <= 0 || token_count > 4 || token_count > max_seq_len_) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: execution exceeds buffer capacity");
    }
    if (token_count != plan.token_count || top_k != plan.top_k) {
        throw std::invalid_argument("Qwen4ExpExpertExecutor: execution shape differs from plan");
    }
    if (plan.cpu_tasks < 0 || plan.cpu_tasks > ops::cpu::kMaxExpertBatchTasks ||
        static_cast<size_t>(plan.cpu_tasks) != active_cpu_leases_.size())
        throw std::logic_error("Qwen4ExpExpertExecutor: CPU lease count differs from plan");
    auto validate_entry = [&](const RouteRankEntry& entry) {
        const auto res = cache_->query_residency(plan.layer, entry.expert_id);
        if (!res.gpu_resident || res.gpu_slot != entry.gpu_slot ||
            res.gpu_generation != entry.gpu_generation) {
            throw std::logic_error("Qwen4ExpExpertExecutor: stale GPU expert plan");
        }
    };
    if (defer_gpu_acquire) {
        for (int idx : plan.ready_gpu_indices) validate_entry(plan.entries[idx]);
    } else {
        for (const auto& entry : plan.entries) {
            if (entry.is_gpu) validate_entry(entry);
        }
    }
    struct ExecutionGuard {
        Context& ctx;
        CpuWorker* worker;
        std::vector<HostExpertLease>& leases;
        bool submitted = false;
        ~ExecutionGuard() {
            if (!submitted) {
                if (worker) worker->drain();
                try { ctx.synchronize(); } catch (...) {}
                leases.clear();
            }
        }
    } execution_guard{ctx, cpu_worker_.get(), active_cpu_leases_};
    bool prefetch_started = false;
    auto start_next_prefetch = [&] {
        if (next_prefetch && !prefetch_started) {
            cache_->prefetch_async(next_prefetch->layer, next_prefetch->expert_ids,
                                   next_prefetch->count, next_prefetch->quota);
            prefetch_started = true;
        }
    };

    // Drain previous layer's CPU result copy if still in-flight to prevent host buffer reuse race
    if (prev_cpu_h2d_ev_ != sycl::event{}) {
        prev_cpu_h2d_ev_.wait();
        prev_cpu_h2d_ev_ = sycl::event{};
    }

    const size_t total_acts = static_cast<size_t>(token_count) * top_k;
    // Copy router weights to device for ordered rank accumulation
    ctx.queue().memcpy(weights_dev_, router_weights_host, total_acts * sizeof(float));

    // 1. If CPU tasks exist, initiate async fetch of activations x_dev to pinned host memory
    sycl::event x_d2h_ev{};
    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        x_d2h_ev = ctx.queue().memcpy(x_host_, x_dev, static_cast<size_t>(token_count) * hidden_size_ * sizeof(bf16));
    }
    const bool async_cpu = cpu_worker_ && plan.cpu_tasks > 0;
    acquire_begin_ns_ = acquire_end_ns_ = 0;
    if (async_cpu) {
        CpuWorker::Job job;
        job.count = plan.cpu_tasks; job.hidden = hidden_size_; job.intermediate = intermediate_size_; job.tokens = token_count;
        job.x = x_host_; job.h = scratch_h_; job.out = cpu_out_host_; job.activation = x_d2h_ev;
        for (int c = 0; c < plan.cpu_tasks; ++c)
            job.tasks[c] = {active_cpu_leases_[c].slot_host(), plan.entries[plan.cpu_indices[c]].token_idx};
        cpu_worker_->submit(job);
    }

    // 2. Launch GPU execution FIRST so GPU hardware runs asynchronously while CPU computes
    sycl::event down_ev{};
    if (plan.gpu_tasks > 0) {
        cache_->record_first_consumer_start();
        stats_.gpu_tasks_dispatched += plan.gpu_tasks;
        const int num_slots = cache_->num_slots();

        if (plan.ready_gpu_tasks > 0 && plan.pending_gpu_tasks > 0) {
            const int ready_count = plan.ready_gpu_tasks;
            const int pending_count = plan.pending_gpu_tasks;

            // 1. Setup ready subset at indices [0 .. ready_count - 1]
            for (int r = 0; r < ready_count; ++r) {
                int idx = plan.ready_gpu_indices[r];
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid ready gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[r] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[r] = plan.entries[idx].rank;
                gpu_tokens_host_[r] = plan.entries[idx].token_idx;
            }

            const ops::MoeExpertSlotDev* eff_slots = gpu_slots_host_;
            const int* eff_ranks = gpu_ranks_host_;
            const int* eff_tokens = gpu_tokens_host_;
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_, gpu_slots_host_, ready_count * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_, gpu_ranks_host_, ready_count * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_, gpu_tokens_host_, ready_count * sizeof(int));
                eff_slots = gpu_slots_dev_;
                eff_ranks = gpu_ranks_dev_;
                eff_tokens = gpu_tokens_dev_;
            }

            sycl::event gu_ev_ready = ops::moe_decode_gate_up_swiglu_grouped(
                ctx, eff_slots, nullptr, x_dev, moe_h_dev, ready_count,
                hidden_size_, intermediate_size_, 64,
                nullptr, eff_tokens, uses_nested);

            sycl::event down_ev_ready = ops::moe_decode_down_to_rank_f32(
                ctx, eff_slots, eff_ranks, moe_h_dev, down_acc_dev_,
                ready_count, top_k, intermediate_size_, hidden_size_, 64,
                eff_tokens, nullptr, &gu_ev_ready, uses_nested);

            for (int idx : plan.ready_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev_ready);
            }
            plan.ready_subset_event = down_ev_ready;
            if (defer_gpu_acquire) {
                // Ready compute is already queued while the host services misses.
                ensure_gpu_pending(plan);
                start_next_prefetch();
                cache_->record_first_consumer_start();
                ++stats_.ready_first_layers;
            }

            for (int p = 0; p < pending_count; ++p) {
                int idx = plan.pending_gpu_indices[p];
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid pending gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[ready_count + p] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[ready_count + p] = plan.entries[idx].rank;
                gpu_tokens_host_[ready_count + p] = plan.entries[idx].token_idx;
            }
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_ + ready_count, gpu_slots_host_ + ready_count,
                                   pending_count * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_ + ready_count, gpu_ranks_host_ + ready_count, pending_count * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_ + ready_count, gpu_tokens_host_ + ready_count, pending_count * sizeof(int));
            }

            sycl::event gu_ev_pending = ops::moe_decode_gate_up_swiglu_grouped(
                ctx, eff_slots + ready_count, nullptr, x_dev,
                moe_h_dev + static_cast<size_t>(ready_count) * intermediate_size_,
                pending_count, hidden_size_, intermediate_size_, 64,
                plan.final_dma_event != sycl::event{} ? &plan.final_dma_event : nullptr,
                eff_tokens + ready_count, uses_nested);

            sycl::event down_ev_pending = ops::moe_decode_down_to_rank_f32(
                ctx, eff_slots + ready_count, eff_ranks + ready_count,
                moe_h_dev + static_cast<size_t>(ready_count) * intermediate_size_,
                down_acc_dev_, pending_count, top_k, intermediate_size_, hidden_size_, 64,
                eff_tokens + ready_count, nullptr, &gu_ev_pending, uses_nested);

            for (int idx : plan.pending_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev_pending);
            }

            down_ev = down_ev_pending;
        } else {
            if (defer_gpu_acquire) {
                ensure_gpu_pending(plan);
                start_next_prefetch();
                cache_->record_first_consumer_start();
            }
            int g_idx = 0;
            for (int idx : plan.ready_gpu_indices) {
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid ready gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[g_idx] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
                gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
                ++g_idx;
            }
            for (int idx : plan.pending_gpu_indices) {
                const int slot = plan.entries[idx].gpu_slot;
                if (slot < 0 || slot >= num_slots) {
                    throw std::runtime_error("Qwen4ExpExpertExecutor: invalid pending gpu_slot " + std::to_string(slot));
                }
                gpu_slots_host_[g_idx] = cache_->get_slot_dev(slot);
                gpu_ranks_host_[g_idx] = plan.entries[idx].rank;
                gpu_tokens_host_[g_idx] = plan.entries[idx].token_idx;
                ++g_idx;
            }

            const ops::MoeExpertSlotDev* eff_slots = gpu_slots_host_;
            const int* eff_ranks = gpu_ranks_host_;
            const int* eff_tokens = gpu_tokens_host_;
            if (plan.gpu_tasks > 16) {
                ctx.queue().memcpy(gpu_slots_dev_, gpu_slots_host_, plan.gpu_tasks * sizeof(ops::MoeExpertSlotDev));
                ctx.queue().memcpy(gpu_ranks_dev_, gpu_ranks_host_, plan.gpu_tasks * sizeof(int));
                ctx.queue().memcpy(gpu_tokens_dev_, gpu_tokens_host_, plan.gpu_tasks * sizeof(int));
                eff_slots = gpu_slots_dev_;
                eff_ranks = gpu_ranks_dev_;
                eff_tokens = gpu_tokens_dev_;
            }

            const sycl::event* dep = (plan.pending_gpu_tasks > 0 && plan.final_dma_event != sycl::event{})
                ? &plan.final_dma_event : nullptr;

            sycl::event gu_ev = ops::moe_decode_gate_up_swiglu_grouped(
                ctx, eff_slots, nullptr, x_dev, moe_h_dev, plan.gpu_tasks,
                hidden_size_, intermediate_size_, 64,
                dep, eff_tokens, uses_nested);

            down_ev = ops::moe_decode_down_to_rank_f32(
                ctx, eff_slots, eff_ranks, moe_h_dev, down_acc_dev_,
                plan.gpu_tasks, top_k, intermediate_size_, hidden_size_, 64,
                eff_tokens, nullptr, &gu_ev, uses_nested);

            for (int idx : plan.ready_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
            }
            for (int idx : plan.pending_gpu_indices) {
                cache_->record_consumer_reader_fence(plan.entries[idx].gpu_slot, down_ev);
            }
        }
    }

    start_next_prefetch();

    // 3. Concurrent CPU execution for Host-ready miss experts (runs in parallel with GPU!)
    sycl::event last_cpu_h2d_ev{};
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    bool async_join_blocked = false;

    if (plan.cpu_tasks > 0 && !active_cpu_leases_.empty()) {
        auto copy_results = [&](int begin, int end) {
            const auto submit_start = steady_ns();
            const size_t bytes = static_cast<size_t>(end - begin) * hidden_size_ * sizeof(float);
            float* dst = down_acc_dev_ + static_cast<size_t>(plan.cpu_indices[begin]) * hidden_size_;
            const float* src = cpu_out_host_ + static_cast<size_t>(begin) * hidden_size_;
            last_cpu_h2d_ev = ctx.queue().memcpy(dst, src, bytes);
            ++stats_.cpu_result_copies;
            stats_.cpu_result_bytes += bytes;
            stats_.cpu_result_submit_ns += steady_ns() - submit_start;
        };
        if (async_cpu) {
            async_join_blocked = !cpu_worker_->complete();
            const auto join_start = steady_ns();
            const auto completed = cpu_worker_->finish();
            stats_.cpu_join_wait_ns += steady_ns() - join_start;
            stats_.cpu_activation_wait_ns += completed.activation_ns;
            stats_.cpu_compute_ns += completed.end_ns - completed.begin_ns;
            if (acquire_end_ns_ > acquire_begin_ns_) {
                const auto start = std::max(acquire_begin_ns_, completed.begin_ns);
                const auto end = std::min(acquire_end_ns_, completed.end_ns);
                if (end > start) stats_.cpu_acquire_overlap_ns += end - start;
            }
            ++stats_.cpu_async_layers;
        } else {
            if (x_d2h_ev != sycl::event{}) x_d2h_ev.wait_and_throw();
            stats_.cpu_activation_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - cpu_t0).count();
        }

        for (int c = 0; c < plan.cpu_tasks; ++c) {
            const int plan_idx = plan.cpu_indices[c];
            const auto& entry = plan.entries[plan_idx];
            const int t = entry.token_idx;
            const bf16* token_x = x_host_ + static_cast<size_t>(t) * hidden_size_;
            float* out_slot = cpu_out_host_ + static_cast<size_t>(c) * hidden_size_;

            bf16* task_h = scratch_h_ + static_cast<size_t>(c) * intermediate_size_;
            if (!async_cpu) {
                const auto compute_start = steady_ns();
                compute_expert(active_cpu_leases_[c].slot_host(), token_x, task_h, out_slot, hidden_size_, intermediate_size_);
                stats_.cpu_compute_ns += steady_ns() - compute_start;
            }
            if (cpu_observer_) {
                cpu_observer_(cpu_observer_state_, ctx, plan.layer, entry,
                              active_cpu_leases_[c].slot_host(), token_x, task_h, out_slot);
            }

            // Release lease immediately after reading weights
            active_cpu_leases_[c].release();

            if (!cpu_coalesce_results_) copy_results(c, c + 1);
        }
        if (cpu_coalesce_results_) {
            for (int begin = 0; begin < plan.cpu_tasks; ) {
                int end = begin + 1;
                while (end < plan.cpu_tasks && plan.cpu_indices[end] == plan.cpu_indices[end - 1] + 1) ++end;
                copy_results(begin, end);
                begin = end;
            }
        }

        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        const uint64_t cpu_elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_t1 - cpu_t0).count();
        stats_.cpu_tasks_dispatched += plan.cpu_tasks;
        stats_.cpu_tasks_completed += plan.cpu_tasks;
        stats_.cpu_max_batch_tasks = std::max(stats_.cpu_max_batch_tasks, static_cast<uint64_t>(plan.cpu_tasks));
        active_cpu_leases_.clear();

        // Async work that is not ready at merge admission exposes a host wait.
        // GPU status sampled after joining cannot classify that earlier overlap.
        if (plan.gpu_tasks > 0) {
            const auto gpu_status = down_ev.get_info<sycl::info::event::command_execution_status>();
            if (async_cpu ? async_join_blocked : gpu_status == sycl::info::event_command_status::complete) {
                // Penalize an observed blocking join, or the original synchronous tail.
                stats_.cpu_straggler_count++;
                consecutive_stragglers_++;
                stats_.cpu_straggler_interval_ns += cpu_elapsed_ns;
                if (consecutive_stragglers_ >= 2) {
                    set_straggler_penalty(true);
                }
            } else {
                // No blocking join observed for async, or GPU still active for sync.
                stats_.cpu_overlapped_count++;
                consecutive_stragglers_ = 0;
                set_straggler_penalty(false);
            }
        }
    }

    // 5. Ordered Rank Merge: merges all ranks in strict rank order (0..top_k-1)
    sycl::event merge_ev = ops::moe_ordered_accumulate_down_rows(
        ctx, down_acc_dev_, weights_dev_, y_dev, token_count, top_k,
        hidden_size_, shared_out, shared_gate,
        plan.gpu_tasks > 0 ? &down_ev : (last_cpu_h2d_ev != sycl::event{} ? &last_cpu_h2d_ev : nullptr));

    prev_cpu_h2d_ev_ = last_cpu_h2d_ev;
    stats_.layers_hetero_executed++;
    execution_guard.submitted = true;
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
