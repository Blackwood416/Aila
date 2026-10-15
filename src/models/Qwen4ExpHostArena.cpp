#include "models/Qwen4ExpHostArena.hpp"
#include "memory/LargePages.hpp"
#include "utils/EnvUtils.hpp"
#include "profile/Profiling.hpp"
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {
#ifdef _WIN32
// F1-style advisory prefetch (same pattern as Qwen4ExpPle::prefetch_gather_pages):
// hand the cache manager the file-backed source ranges in bounded chunks so the
// reads queue up with real depth instead of serializing as demand faults on a
// single file section (C7 screen: ~0.5-0.66 GB/s regardless of fault threads).
// PrefetchVirtualMemory accepts at most 64 entries per call. Failures are
// ignored: the normal memcpy path still faults pages in.
// chunk_bytes defaults to 1 MiB at the call sites, which reproduces the
// original call sequence byte for byte.
void prefetch_source_range(const uint8_t* base, size_t bytes, size_t chunk_bytes) {
    if (!base || !bytes) return;
    if (!chunk_bytes) chunk_bytes = size_t(1) << 20;
    constexpr size_t kBatch = 64;
    WIN32_MEMORY_RANGE_ENTRY ranges[kBatch];
    const HANDLE process = GetCurrentProcess();
    size_t count = 0;
    for (size_t off = 0; off < bytes; off += chunk_bytes) {
        ranges[count].VirtualAddress = const_cast<uint8_t*>(base + off);
        ranges[count].NumberOfBytes = std::min(chunk_bytes, bytes - off);
        if (++count == kBatch) {
            PrefetchVirtualMemory(process, count, ranges, 0);
            count = 0;
        }
    }
    if (count) PrefetchVirtualMemory(process, count, ranges, 0);
}
#else
void prefetch_source_range(const uint8_t*, size_t, size_t) {}
#endif

#ifdef _WIN32
// R1 round 3 trim audit (AILA_Q4E_LOAD_TIMING only): read the process working
// set, private commit and cumulative page-fault count. Read-only observation;
// the default path never calls this. `valid == false` means the query failed
// and the zeroed fields are printed as-is.
struct ProcessMemorySample {
    double working_set_mib = 0.0;
    double private_mib = 0.0;
    unsigned long page_faults = 0;
    bool valid = false;
};
ProcessMemorySample sample_process_memory() {
    ProcessMemorySample sample;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters))) {
        sample.working_set_mib = double(counters.WorkingSetSize) / 1048576.0;
        sample.private_mib = double(counters.PrivateUsage) / 1048576.0;
        sample.page_faults = counters.PageFaultCount;
        sample.valid = true;
    }
    return sample;
}
#endif
} // namespace

namespace aila {
Qwen4ExpHostArena::~Qwen4ExpHostArena() { Release(); }
Qwen4ExpHostArena::Qwen4ExpHostArena(Qwen4ExpHostArena&& other) noexcept { *this=std::move(other); }
Qwen4ExpHostArena& Qwen4ExpHostArena::operator=(Qwen4ExpHostArena&& other) noexcept {
    if (this == &other) return *this;
    Release();
    config_=other.config_; geom_=other.geom_;
    total_experts_=std::exchange(other.total_experts_,0);
    resident_experts_=std::exchange(other.resident_experts_,0);
    allocated_bytes_=std::exchange(other.allocated_bytes_,0);
    ctx_=std::exchange(other.ctx_,nullptr);
    standalone_queue_=std::move(other.standalone_queue_);
    chunks_=std::move(other.chunks_); chunk_sizes_=std::move(other.chunk_sizes_);
    expert_ptrs_=std::move(other.expert_ptrs_); initial_gpu_owners_=std::move(other.initial_gpu_owners_);
    load_staging_=std::exchange(other.load_staging_,nullptr);
    load_copy_event_=other.load_copy_event_;
    load_copy_pending_=std::exchange(other.load_copy_pending_,false);
    layer_nested_=std::move(other.layer_nested_);
    layer_nested_dev_allocs_=std::move(other.layer_nested_dev_allocs_);
    return *this;
}

bool Qwen4ExpHostArena::Allocate(Context& ctx, const HostArenaConfig& cfg, std::string* error,
                                const std::vector<uint8_t*>& gpu_owners) {
    Release(); ctx_=&ctx;
    return AllocateStorage(cfg,gpu_owners,error);
}
bool Qwen4ExpHostArena::Allocate(sycl::queue& q, const HostArenaConfig& cfg, std::string* error,
                                const std::vector<uint8_t*>& gpu_owners) {
    Release(); standalone_queue_=std::make_unique<sycl::queue>(q);
    return AllocateStorage(cfg,gpu_owners,error);
}
bool Qwen4ExpHostArena::AllocateStorage(const HostArenaConfig& cfg,
                                       const std::vector<uint8_t*>& gpu_owners, std::string* error) {
    std::string failure_detail;
    try {
        config_=cfg;
        if (cfg.num_layers<=0 || cfg.num_experts_per_layer<=0 || cfg.hidden_size<=0 ||
            cfg.moe_intermediate_size<=0 || cfg.blocksize!=64 || !cfg.compact_nested ||
            cfg.hidden_size%64 || cfg.moe_intermediate_size%64)
            throw std::invalid_argument("Qwen4Exp arena requires block-64 compact NF4 geometry");
        geom_.gu_packed_bytes=size_t(2*cfg.moe_intermediate_size)*cfg.hidden_size/2;
        geom_.gu_absmax_count=size_t(2*cfg.moe_intermediate_size)*cfg.hidden_size/64;
        geom_.gu_absmax_bytes=geom_.gu_absmax_count;
        geom_.dn_packed_bytes=size_t(cfg.hidden_size)*cfg.moe_intermediate_size/2;
        geom_.dn_absmax_count=size_t(cfg.hidden_size)*cfg.moe_intermediate_size/64;
        geom_.dn_absmax_bytes=geom_.dn_absmax_count;
        geom_.compact_nested=true;
        geom_.gu_packed_offset=0; geom_.gu_absmax_offset=geom_.gu_packed_bytes;
        geom_.dn_packed_offset=geom_.gu_absmax_offset+geom_.gu_absmax_bytes;
        geom_.dn_absmax_offset=geom_.dn_packed_offset+geom_.dn_packed_bytes;
        geom_.slot_bytes=geom_.dn_absmax_offset+geom_.dn_absmax_bytes;
        total_experts_=size_t(cfg.num_layers)*cfg.num_experts_per_layer;
        if (!gpu_owners.empty() && gpu_owners.size()!=total_experts_)
            throw std::invalid_argument("Qwen4Exp arena GPU ownership table size mismatch");
        initial_gpu_owners_=gpu_owners;
        expert_ptrs_.assign(total_experts_,nullptr);
        resident_experts_=total_experts_;
        for (auto* p:gpu_owners) if (p) --resident_experts_;
        const size_t per_chunk=std::max(size_t(1),std::min(cfg.chunk_bytes,size_t(256ULL<<20))/geom_.slot_bytes);
        // Default: native large pages. AILA_Q4E_HOST_ARENA_PINNED=1 switches
        // the complement to the pinned host allocator so the two can be
        // compared on CPU TLB pressure versus large-page availability.
        const bool pinned_arena = aila::env::read_flag("AILA_Q4E_HOST_ARENA_PINNED", false);
        for (size_t n=0;n<resident_experts_;) {
            const size_t count=std::min(per_chunk,resident_experts_-n);
            const size_t bytes=memory::LargePageAllocationSize(count*geom_.slot_bytes);
            uint8_t* ptr=nullptr;
            try {
                ptr=static_cast<uint8_t*>(
                    ctx_ ? (pinned_arena ? ctx_->alloc_host(bytes) : ctx_->alloc_large_host(bytes))
                         : memory::AllocateLargePages(bytes));
            } catch (const std::exception& e) {
                // Fatal startup path: bypass the deferred log buffer so the
                // numbers survive an early exit, and keep them in the error
                // string the caller reports.
                unsigned long long avail_mb=0,total_mb=0;
#ifdef _WIN32
                MEMORYSTATUSEX status{};
                status.dwLength=sizeof(status);
                if(GlobalMemoryStatusEx(&status)) {
                    avail_mb=status.ullAvailPhys>>20; total_mb=status.ullTotalPhys>>20;
                }
#endif
                char detail[512];
                std::snprintf(detail,sizeof(detail),
                    "chunks=%zu allocated=%.1f MiB chunk_bytes=%zu per_chunk=%zu "
                    "count=%zu bytes=%.1f MiB resident_experts=%zu host_avail=%.0f MiB "
                    "host_total=%.0f MiB pinned=%d err=%s",
                    chunks_.size(),double(allocated_bytes_)/1048576.0,cfg.chunk_bytes,per_chunk,
                    count,double(bytes)/1048576.0,resident_experts_,double(avail_mb),
                    double(total_mb),pinned_arena?1:0,e.what());
                failure_detail=detail;
                std::fprintf(stderr,"[Q4E Arena] allocation failed: %s\n",detail);
                std::fflush(stderr);
                if (error) *error=std::string("Qwen4Exp host arena allocation failed: ")+detail;
                Release();
                return false;
            }
            chunks_.push_back(ptr); chunk_sizes_.push_back(count);
            allocated_bytes_+=bytes; n+=count;
        }
        size_t c=0,offset=0;
        for (size_t key=0;key<total_experts_;++key) {
            if (!gpu_owners.empty() && gpu_owners[key]) continue;
            if (offset==chunk_sizes_[c]) { ++c; offset=0; }
            expert_ptrs_[key]=chunks_[c]+offset++*geom_.slot_bytes;
        }
        load_staging_=static_cast<uint8_t*>(ctx_ ? ctx_->alloc_host(geom_.slot_bytes) : sycl::malloc_host(geom_.slot_bytes,Queue()));
        if (!load_staging_) throw std::bad_alloc();
        layer_nested_.resize(cfg.num_layers);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = failure_detail.empty()
            ? std::string(e.what())
            : std::string(e.what()) + " (" + failure_detail + ")";
        Release(); return false;
    }
}

void Qwen4ExpHostArena::Release() noexcept {
    if (load_copy_pending_) { try { load_copy_event_.wait_and_throw(); } catch (...) {} }
    if (ctx_ || standalone_queue_) { try { Queue().wait_and_throw(); } catch (...) {} }
    for (void* p:layer_nested_dev_allocs_) {
        try { if (ctx_) ctx_->free_device(p); else sycl::free(p,Queue()); } catch (...) {}
    }
    if (load_staging_) {
        try { if (ctx_) ctx_->free_host(load_staging_); else sycl::free(load_staging_,Queue()); } catch (...) {}
    }
    for (auto* p:chunks_) {
        try { if (ctx_) ctx_->free_host(p); else memory::FreeLargePages(p); } catch (...) {}
    }
    chunks_.clear(); chunk_sizes_.clear(); expert_ptrs_.clear(); initial_gpu_owners_.clear();
    layer_nested_.clear(); layer_nested_dev_allocs_.clear();
    total_experts_=resident_experts_=allocated_bytes_=0;
    load_staging_=nullptr; load_copy_pending_=false; load_copy_event_={};
    ctx_=nullptr; standalone_queue_.reset();
}

sycl::event Qwen4ExpHostArena::DmaTransferToGpu(sycl::queue& q, void* dst, int layer, int expert) const {
    const auto* src=GetExpertPointer(layer,expert);
    if (!src || !dst) throw std::logic_error("Qwen4Exp DMA requires a RAM owner and a GPU destination");
    if (load_copy_pending_) load_copy_event_.wait_and_throw();
    std::memcpy(load_staging_,src,geom_.slot_bytes);
    load_copy_event_=q.memcpy(dst,load_staging_,geom_.slot_bytes);
    load_copy_pending_=true;
    return load_copy_event_;
}

void Qwen4ExpHostArena::LoadLayerDirect(int layer, const Bnb4BitHostWeightRef& gu,
                                        const Bnb4BitHostWeightRef& dn) {
    if (!IsAllocated() || layer<0 || layer>=config_.num_layers)
        throw std::invalid_argument("Qwen4Exp arena layer is not allocated");
    // R1 round 3 (see docs/plans/qwen38-r1-startup-arena-2026-10-15.md section
    // 7): gate for the per-layer process-wide working-set trim. Default 1 keeps
    // the historical 2026-10-08 behaviour (consumed mmap source pages become
    // reclaimable after every layer); =0 skips the syscall so the audit can
    // measure what the trim actually buys. The audit sampling below is read
    // only and runs exclusively under AILA_Q4E_LOAD_TIMING.
#ifdef _WIN32
    static const bool s_working_set_trim =
        aila::env::read_flag("AILA_Q4E_ARENA_WORKING_SET_TRIM", true);
    static const bool s_trim_audit = aila::env::read_flag("AILA_Q4E_LOAD_TIMING", false);
    ProcessMemorySample layer_entry_sample;
    if (s_trim_audit) layer_entry_sample = sample_process_memory();
#endif
    const size_t experts=config_.num_experts_per_layer;
    auto check=[&](const Bnb4BitHostWeightRef& r,size_t packed,size_t codes) {
        if (!r.packed_weight.valid() || r.packed_weight.bytes!=experts*packed ||
            !r.absmax.valid() || r.absmax.bytes!=experts*codes || !r.quant_state.nested ||
            r.quant_state.blocksize!=64 || r.quant_state.nested_blocksize!=256 || codes%256 ||
            !r.nested_absmax.valid() || r.nested_absmax.bytes<experts*codes/256*sizeof(float) ||
            !r.nested_quant_map.valid() || r.nested_quant_map.bytes!=256*sizeof(float))
            throw std::invalid_argument("Invalid Qwen4Exp compact nested expert weight: "+r.name);
    };
    check(gu,geom_.gu_packed_bytes,geom_.gu_absmax_count);
    check(dn,geom_.dn_packed_bytes,geom_.dn_absmax_count);
    if (layer_nested_[layer].valid) throw std::logic_error("Qwen4Exp arena layer already loaded");
    const auto copy_t0 = std::chrono::steady_clock::now();
    // Strata C7 screen (opt-in): the 58 GiB arena fill runs at ~0.5 GB/s when
    // single-threaded because it is page-fault/IO bound. Spread the RAM-destined
    // experts over worker threads; GPU-owned slots keep the serialized staging
    // path below. Default 1 keeps the historical path byte-identical.
    static const int s_fill_threads = std::clamp(
        aila::env::read_int("AILA_Q4E_ARENA_FILL_THREADS", 1), 1, 16);
    auto fill_range = [&](size_t begin, size_t end) {
        for (size_t e=begin;e<end;++e) {
            auto* dst=GetExpertPointer(layer,int(e));
            if (!dst) continue;
            std::memcpy(dst+geom_.gu_packed_offset,gu.packed_weight.ptr+e*geom_.gu_packed_bytes,geom_.gu_packed_bytes);
            std::memcpy(dst+geom_.gu_absmax_offset,gu.absmax.ptr+e*geom_.gu_absmax_count,geom_.gu_absmax_bytes);
            std::memcpy(dst+geom_.dn_packed_offset,dn.packed_weight.ptr+e*geom_.dn_packed_bytes,geom_.dn_packed_bytes);
            std::memcpy(dst+geom_.dn_absmax_offset,dn.absmax.ptr+e*geom_.dn_absmax_count,geom_.dn_absmax_bytes);
        }
    };
    if (s_fill_threads>1) {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<size_t>(s_fill_threads));
        const size_t per=(experts+static_cast<size_t>(s_fill_threads)-1)/static_cast<size_t>(s_fill_threads);
        for (int t=0;t<s_fill_threads;++t) {
            const size_t begin=static_cast<size_t>(t)*per;
            const size_t end=std::min(experts,begin+per);
            if (begin<end) pool.emplace_back(fill_range,begin,end);
        }
        for (auto& th:pool) th.join();
    } else {
        fill_range(0,experts);
    }
    const auto ram_copy_done = std::chrono::steady_clock::now();
    if (aila::env::read_flag("AILA_Q4E_LOAD_TIMING", false)) {
        // The layer loop is serial (fill threads have joined above), so plain
        // statics are enough for this opt-in diagnostic.
        static double s_ram_copy_ms = 0.0;
        static std::vector<double> s_fill_ms_by_layer;
        static int s_layers = 0;
        const double fill_ms =
            std::chrono::duration<double, std::milli>(ram_copy_done - copy_t0).count();
        s_ram_copy_ms += fill_ms;
        s_fill_ms_by_layer.push_back(fill_ms);
        ++s_layers;
        if (s_layers == config_.num_layers) {
            AILA_LOG_WARN("[Q4E LoadTiming] arena RAM copies: %.1f s total (%.1f ms/layer)",
                          s_ram_copy_ms / 1000.0,
                          s_ram_copy_ms / std::max(1, s_layers));
            // R1 round 3: compact per-layer fill list so the trim audit can
            // pair "trim at layer L" with "fill of layer L+1" (the fill right
            // after that layer's trim) without extra log lines. %zu=layer.
            std::string per_layer;
            per_layer.reserve(size_t(config_.num_layers) * 12);
            for (size_t i = 0; i < s_fill_ms_by_layer.size(); ++i) {
                char cell[32];
                std::snprintf(cell, sizeof(cell), "%s%zu=%.0f", i ? " " : "", i,
                              s_fill_ms_by_layer[i]);
                per_layer += cell;
            }
            AILA_LOG_WARN("[Q4E LoadTiming] fill ms/layer: %s", per_layer.c_str());
        }
    }
    // R1 round 2 residual attribution (AILA_Q4E_LOAD_TIMING only): split the
    // per-layer copy+upload window after the RAM fill into GPU-owned staging
    // (per-expert staging memcpy / H2D completion wait / enqueue), nested
    // metadata + device uploads, the closing device drain, and the per-layer
    // working-set trim. Purely diagnostic: when the gate is off `mark()` only
    // evaluates the bool and never reads the clock.
    static const bool s_residual_timing = aila::env::read_flag("AILA_Q4E_LOAD_TIMING", false);
    auto mark = [&]() {
        return s_residual_timing ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    };
    double stage_wait_ms = 0.0, stage_memcpy_ms = 0.0, stage_enqueue_ms = 0.0;
    double nested_ms = 0.0, drain_ms = 0.0, trim_ms = 0.0;
    size_t stage_experts = 0;
    for (size_t e=0;e<experts;++e) {
        auto* dst=GetExpertPointer(layer,int(e));
        if (dst) continue;
        if (initial_gpu_owners_.empty()) throw std::logic_error("Missing Qwen4Exp expert owner while loading");
        uint8_t* gpu=initial_gpu_owners_[size_t(layer)*experts+e];
        if (!gpu) throw std::logic_error("Missing Qwen4Exp initial GPU owner");
        const auto stage_t0 = mark();
        if (load_copy_pending_) load_copy_event_.wait_and_throw();
        const auto stage_t1 = mark();
        dst=load_staging_;
        std::memcpy(dst+geom_.gu_packed_offset,gu.packed_weight.ptr+e*geom_.gu_packed_bytes,geom_.gu_packed_bytes);
        std::memcpy(dst+geom_.gu_absmax_offset,gu.absmax.ptr+e*geom_.gu_absmax_count,geom_.gu_absmax_bytes);
        std::memcpy(dst+geom_.dn_packed_offset,dn.packed_weight.ptr+e*geom_.dn_packed_bytes,geom_.dn_packed_bytes);
        std::memcpy(dst+geom_.dn_absmax_offset,dn.absmax.ptr+e*geom_.dn_absmax_count,geom_.dn_absmax_bytes);
        const auto stage_t2 = mark();
        load_copy_event_=Queue().memcpy(gpu,dst,geom_.slot_bytes);
        load_copy_pending_=true;
        if (s_residual_timing) {
            stage_wait_ms += std::chrono::duration<double, std::milli>(stage_t1 - stage_t0).count();
            stage_memcpy_ms += std::chrono::duration<double, std::milli>(stage_t2 - stage_t1).count();
            stage_enqueue_ms += std::chrono::duration<double, std::milli>(mark() - stage_t2).count();
            ++stage_experts;
        }
    }
    const auto nested_t0 = mark();
    auto& m=layer_nested_[layer];
    m.gu_nested_absmax.assign(gu.nested_absmax_ptr(),gu.nested_absmax_ptr()+gu.nested_absmax.bytes/sizeof(float));
    m.dn_nested_absmax.assign(dn.nested_absmax_ptr(),dn.nested_absmax_ptr()+dn.nested_absmax.bytes/sizeof(float));
    m.gu_nested_quant_map.assign(gu.nested_quant_map_ptr(),gu.nested_quant_map_ptr()+256);
    m.dn_nested_quant_map.assign(dn.nested_quant_map_ptr(),dn.nested_quant_map_ptr()+256);
    m.gu_nested_offset=gu.quant_state.nested_offset; m.dn_nested_offset=dn.quant_state.nested_offset;
    const size_t gu_bytes=m.gu_nested_absmax.size()*sizeof(float), dn_bytes=m.dn_nested_absmax.size()*sizeof(float);
    const size_t map_bytes=256*sizeof(float), total=gu_bytes+dn_bytes+2*map_bytes;
    auto* dev=static_cast<uint8_t*>(ctx_ ? ctx_->alloc_device(total) : sycl::malloc_device(total,Queue()));
    if (!dev) throw std::bad_alloc();
    layer_nested_dev_allocs_.push_back(dev);
    m.gu_nested_absmax_dev=reinterpret_cast<float*>(dev);
    m.dn_nested_absmax_dev=reinterpret_cast<float*>(dev+gu_bytes);
    m.gu_nested_quant_map_dev=reinterpret_cast<float*>(dev+gu_bytes+dn_bytes);
    m.dn_nested_quant_map_dev=reinterpret_cast<float*>(dev+gu_bytes+dn_bytes+map_bytes);
    Queue().memcpy(dev,m.gu_nested_absmax.data(),gu_bytes);
    Queue().memcpy(dev+gu_bytes,m.dn_nested_absmax.data(),dn_bytes);
    Queue().memcpy(dev+gu_bytes+dn_bytes,m.gu_nested_quant_map.data(),map_bytes);
    Queue().memcpy(dev+gu_bytes+dn_bytes+map_bytes,m.dn_nested_quant_map.data(),map_bytes);
    const auto drain_t0 = mark();
    Queue().wait_and_throw();
    const auto drain_t1 = mark();
    m.valid=true;
    if (s_residual_timing) {
        nested_ms += std::chrono::duration<double, std::milli>(drain_t0 - nested_t0).count();
        drain_ms += std::chrono::duration<double, std::milli>(drain_t1 - drain_t0).count();
    }
#ifdef _WIN32
    // Release consumed mmap source pages after each layer, before they compete
    // with the next layer. Native large pages are outside this working set.
    // R1 round 3: gated by AILA_Q4E_ARENA_WORKING_SET_TRIM (default 1 = the
    // 2026-10-08 behaviour); the samples around the trim point are the audit
    // evidence and are only taken under AILA_Q4E_LOAD_TIMING.
    ProcessMemorySample trim_before, trim_after;
    if (s_trim_audit) trim_before = sample_process_memory();
    const auto trim_t0 = mark();
    if (s_working_set_trim) {
        SetProcessWorkingSetSize(GetCurrentProcess(),SIZE_T(-1),SIZE_T(-1));
    }
    if (s_residual_timing) {
        trim_ms += std::chrono::duration<double, std::milli>(mark() - trim_t0).count();
        trim_after = sample_process_memory();
    }
#endif
    if (s_residual_timing) {
        // Same print-once trigger as the RAM-copy timer above.
        static double s_stage_wait_ms = 0.0, s_stage_memcpy_ms = 0.0, s_stage_enqueue_ms = 0.0;
        static double s_nested_ms = 0.0, s_drain_ms = 0.0, s_trim_ms = 0.0;
        static size_t s_stage_experts = 0, s_timing_layers = 0;
#ifdef _WIN32
        static double s_ws_before_mib = 0.0, s_ws_after_mib = 0.0;
        static double s_ws_drop_min_mib = 1e18, s_ws_drop_max_mib = 0.0;
        static double s_private_after_mib = 0.0, s_layer_faults = 0.0;
#endif
        s_stage_wait_ms += stage_wait_ms;
        s_stage_memcpy_ms += stage_memcpy_ms;
        s_stage_enqueue_ms += stage_enqueue_ms;
        s_nested_ms += nested_ms;
        s_drain_ms += drain_ms;
        s_trim_ms += trim_ms;
        s_stage_experts += stage_experts;
#ifdef _WIN32
        s_ws_before_mib += trim_before.working_set_mib;
        s_ws_after_mib += trim_after.working_set_mib;
        s_private_after_mib += trim_after.private_mib;
        const double ws_drop_mib = trim_before.working_set_mib - trim_after.working_set_mib;
        s_ws_drop_min_mib = std::min(s_ws_drop_min_mib, ws_drop_mib);
        s_ws_drop_max_mib = std::max(s_ws_drop_max_mib, ws_drop_mib);
        // Page faults paid across this layer's fill + staging window (entry
        // sample to the trim point), not just the trim syscall itself.
        s_layer_faults += double(trim_after.page_faults - layer_entry_sample.page_faults);
#endif
        ++s_timing_layers;
        if (s_timing_layers == static_cast<size_t>(config_.num_layers)) {
            AILA_LOG_WARN("[Q4E LoadTiming] arena residual split: stage memcpy %.1f s, stage wait %.1f s, stage enqueue %.1f s (%zu experts), nested+device %.1f s, device drain %.1f s, working-set trim %.1f s",
                          s_stage_memcpy_ms / 1000.0, s_stage_wait_ms / 1000.0,
                          s_stage_enqueue_ms / 1000.0, s_stage_experts,
                          s_nested_ms / 1000.0, s_drain_ms / 1000.0, s_trim_ms / 1000.0);
#ifdef _WIN32
            // R1 round 3 trim audit: what the per-layer trim reclaimed (ws_drop
            // = before - after), the residual process working set and private
            // commit after it, and the page-fault rate this layer paid. With
            // AILA_Q4E_ARENA_WORKING_SET_TRIM=0 the same samples are printed,
            // so the two arms can be compared field by field.
            AILA_LOG_WARN("[Q4E LoadTiming] trim audit: trim_on=%d layers=%zu ws_before_avg=%.0f MiB ws_after_avg=%.0f MiB ws_drop_avg=%.0f MiB drop_min=%.0f MiB drop_max=%.0f MiB private_after_avg=%.0f MiB faults_avg=%.0f",
                          s_working_set_trim ? 1 : 0, s_timing_layers,
                          s_ws_before_mib / double(s_timing_layers),
                          s_ws_after_mib / double(s_timing_layers),
                          (s_ws_before_mib - s_ws_after_mib) / double(s_timing_layers),
                          s_ws_drop_min_mib, s_ws_drop_max_mib,
                          s_private_after_mib / double(s_timing_layers),
                          s_layer_faults / double(s_timing_layers));
#endif
        }
    }
}

bool Qwen4ExpHostArena::LoadFromModelWeights(const ModelWeights& weights, std::string* error) {
    try {
        const bool load_timing = aila::env::read_flag("AILA_Q4E_LOAD_TIMING", false);
        // C7 deep dive: reading the layer through mmap page faults is capped at
        // ~0.66 GB/s on Windows (parallel faults on one file mapping serialize).
        // Read the four source ranges with explicit parallel ReadFile calls into
        // a reusable bounce buffer and let the existing per-expert copies run
        // from RAM (~6.7 GB/s). Buffered reads keep the memory behaviour of the
        // historical mmap path, and the per-layer working-set trim stays on.
        // AILA_Q4E_ARENA_READ_THREADS>0 enables it; default 0 keeps the mmap
        // path, because with a warm OS cache the mmap path is already fast
        // (measured 68 s vs 82 s explicit) while a cold cache favours explicit
        // reads (116 s -> 64-83 s). Opt-in for cold starts only.
        static const int s_read_threads = std::clamp(
            aila::env::read_int("AILA_Q4E_ARENA_READ_THREADS", 0), 0, 32);
        // F1-style advisory prefetch of the mmap source ranges (default ON).
        // Cold starts fault ~58 GiB of file-backed pages from the fill loop;
        // the hint is skipped whenever the explicit-read path owns the data.
        static const bool s_host_arena_prefetch =
            aila::env::read_flag("AILA_Q4E_HOST_ARENA_PREFETCH", true);
        // R1 round 2 (default ON after the A/B/A screen: 79.1/58.5/78.6 s
        // worker startup, -26%): issue the layer L+1 hint on a
        // helper thread while layer L's fill and GPU staging run, so the PFV
        // enqueue time stops gating the serial layer loop. Set `=0` to return
        // to the synchronous inline hint. The helper only
        // calls PrefetchVirtualMemory (advisory); the copy path keeps its
        // exact order and bytes. Active only in the mmap path -- with explicit
        // reads (READ_THREADS>0) the sources are a bounce buffer and a hint
        // would be wasted disk work, the same rule the inline path follows.
        static const bool s_prefetch_async =
            aila::env::read_flag("AILA_Q4E_HOST_ARENA_PREFETCH_ASYNC", true);
        // PFV slice size. Default 1024 KiB reproduces the fe1bf9e call
        // sequence exactly; 256 / 4096 are the alternatives under screen.
        // 64 entries per PrefetchVirtualMemory call stays fixed (API limit).
        static const size_t s_prefetch_chunk_bytes = static_cast<size_t>(std::clamp(
            aila::env::read_int("AILA_Q4E_HOST_ARENA_PREFETCH_CHUNK_KB", 1024), 64, 16384)) * 1024;
        // R1 round 3 variant (opt-in, default 1 keeps 58f7874 byte-identical):
        // hint-window depth. 1 = the round-2 schedule (one helper reading layer
        // L+1 while layer L is filled); 2 = two helper slots so L+1 and L+2 are
        // both being read during layer L. Values <= 1 or unset keep the
        // historical single-slot path; read_int maps 0 to the default.
        static const int s_prefetch_lookahead = std::clamp(
            aila::env::read_int("AILA_Q4E_HOST_ARENA_PREFETCH_LOOKAHEAD", 1), 1, 2);
        const bool async_prefetch =
            s_prefetch_async && s_host_arena_prefetch && s_read_threads == 0;
        const auto& shard_paths = weights.shard_paths();
        std::unique_ptr<uint8_t[]> bounce;
        size_t bounce_bytes = 0;
        bool read_fallback_logged = false;
        auto read_range = [](const std::string& path, uint64_t offset, uint8_t* dst,
                             size_t bytes, int threads) -> bool {
            const auto wide = std::filesystem::u8path(path).wstring();
            std::vector<std::thread> pool;
            std::atomic<bool> ok{true};
            pool.reserve(static_cast<size_t>(threads));
            for (int t = 0; t < threads; ++t) {
                pool.emplace_back([&, t] {
                    const size_t per = (bytes + static_cast<size_t>(threads) - 1) /
                                       static_cast<size_t>(threads);
                    const size_t begin = std::min(bytes, per * static_cast<size_t>(t));
                    const size_t end = std::min(bytes, begin + per);
                    if (begin >= end) return;
                    HANDLE h = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                           OPEN_EXISTING,
                                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                                           nullptr);
                    if (h == INVALID_HANDLE_VALUE) {
                        ok = false;
                        return;
                    }
                    LARGE_INTEGER pos{};
                    pos.QuadPart = static_cast<LONGLONG>(offset + begin);
                    bool good = SetFilePointerEx(h, pos, nullptr, FILE_BEGIN) != 0;
                    uint8_t* p = dst + begin;
                    size_t left = end - begin;
                    while (good && left > 0) {
                        DWORD got = 0;
                        const DWORD want = static_cast<DWORD>(std::min<size_t>(left, 4u << 20));
                        if (!ReadFile(h, p, want, &got, nullptr) || got == 0) {
                            good = false;
                            break;
                        }
                        p += got;
                        left -= got;
                    }
                    if (!good) ok = false;
                    CloseHandle(h);
                });
            }
            for (auto& th : pool) th.join();
            return ok.load();
        };
        // Layer weight refs are metadata-only (measured 0.000-0.002 s/layer),
        // so resolving the next layer's refs one iteration early for the async
        // hint costs nothing and keeps the same error strings.
        auto resolve_layer_refs = [&](int target, Bnb4BitHostWeightRef& gu,
                                      Bnb4BitHostWeightRef& dn) {
            const auto prefix="model.language_model.layers."+std::to_string(target)+".mlp.experts.";
            auto load=[&](const std::string& base,Bnb4BitHostWeightRef& ref,int hint) {
                const std::string name=weights.has_host(base) ? base : base+".weight";
                std::string detail;
                if (!LoadBnb4BitHostRef(weights,name,ref,&detail,hint)) throw std::runtime_error(detail);
            };
            load(prefix+"gate_up_proj",gu,config_.hidden_size);
            load(prefix+"down_proj",dn,config_.moe_intermediate_size);
        };
        // Two hint slots exist so the opt-in lookahead=2 window can keep two
        // layers in flight; with the default lookahead=1 only pf_thread is
        // ever used and the call sequence matches 58f7874 exactly.
        std::thread pf_thread, pf_thread2;
        Bnb4BitHostWeightRef pf_gu{}, pf_dn{};
        int pf_target = -1, pf_target2 = -1; // layer each slot is reading
        int pf_next_target = 1;              // next layer needing a hint
        // Exception-safe: a joinable helper must never reach ~thread.
        struct PfThreadGuard {
            std::thread& thread;
            std::thread& thread2;
            ~PfThreadGuard() {
                if (thread.joinable()) thread.join();
                if (thread2.joinable()) thread2.join();
            }
        } pf_guard{pf_thread, pf_thread2};
        double pf_inline_ms = 0.0, pf_issue_ms = 0.0, pf_join_ms = 0.0;
        auto pf_join_thread = [&](std::thread& thread) {
            if (!thread.joinable()) return;
            const auto join_t0 = load_timing ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
            thread.join();
            if (load_timing) {
                pf_join_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - join_t0).count();
            }
        };
        auto pf_launch = [&](std::thread& thread, const Bnb4BitHostWeightRef& gu_ref,
                             const Bnb4BitHostWeightRef& dn_ref) {
            const uint8_t* p0=gu_ref.packed_weight.ptr; const size_t n0=gu_ref.packed_weight.bytes;
            const uint8_t* p1=gu_ref.absmax.ptr;        const size_t n1=gu_ref.absmax.bytes;
            const uint8_t* p2=dn_ref.packed_weight.ptr; const size_t n2=dn_ref.packed_weight.bytes;
            const uint8_t* p3=dn_ref.absmax.ptr;        const size_t n3=dn_ref.absmax.bytes;
            const size_t chunk=s_prefetch_chunk_bytes;
            const auto issue_t0 = load_timing ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            thread=std::thread([p0,n0,p1,n1,p2,n2,p3,n3,chunk]{
                prefetch_source_range(p0,n0,chunk);
                prefetch_source_range(p1,n1,chunk);
                prefetch_source_range(p2,n2,chunk);
                prefetch_source_range(p3,n3,chunk);
            });
            if (load_timing) {
                pf_issue_ms += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - issue_t0).count();
            }
        };
        for (int layer=0;layer<config_.num_layers;++layer) {
            const auto layer_t0 = std::chrono::steady_clock::now();
            Bnb4BitHostWeightRef gu,dn;
            resolve_layer_refs(layer,gu,dn);
            const auto ref_done = std::chrono::steady_clock::now();
            // Close the lookahead window opened during the previous layer
            // before this layer's sources are consumed. Depth 1 has exactly
            // one pending slot (historical behaviour); depth 2 keeps the other
            // slot reading the next layer. No-op in sync mode.
            if (pf_target == layer) { pf_join_thread(pf_thread); pf_target = -1; }
            if (pf_target2 == layer) { pf_join_thread(pf_thread2); pf_target2 = -1; }
            Bnb4BitHostWeightRef gu_src = gu;
            Bnb4BitHostWeightRef dn_src = dn;
            // True while gu_src/dn_src still point at the SafeTensors mmap
            // host_view (the F1 prefetch only makes sense for those pages).
            bool mmap_sources = true;
            if (s_read_threads > 0) {
                const size_t need = gu.packed_weight.bytes + gu.absmax.bytes +
                                    dn.packed_weight.bytes + dn.absmax.bytes;
                if (!bounce || bounce_bytes < need) {
                    bounce.reset(new uint8_t[need]);
                    bounce_bytes = need;
                }
                std::array<HostTensorView*, 4> views{
                    {&gu_src.packed_weight, &gu_src.absmax,
                     &dn_src.packed_weight, &dn_src.absmax}};
                size_t cursor = 0;
                bool read_ok = true;
                for (auto* view : views) {
                    const FileSpan& span = view->file_span;
                    if (!span.valid() ||
                        span.shard_id >= static_cast<int>(shard_paths.size()) ||
                        span.bytes < view->bytes ||
                        !read_range(shard_paths[static_cast<size_t>(span.shard_id)],
                                    span.absolute_offset, bounce.get() + cursor, view->bytes,
                                    s_read_threads)) {
                        read_ok = false;
                        break;
                    }
                    view->ptr = bounce.get() + cursor;
                    cursor += view->bytes;
                }
                if (!read_ok) {
                    gu_src = gu;
                    dn_src = dn;
                    if (!read_fallback_logged) {
                        AILA_LOG_WARN("[Q4E ArenaRead] explicit read failed; falling back to mmap path");
                        read_fallback_logged = true;
                    }
                } else {
                    mmap_sources = false;
                    if (load_timing) {
                        static double s_read_ms = 0.0;
                        s_read_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - ref_done).count();
                        if (layer + 1 == config_.num_layers) {
                            AILA_LOG_WARN("[Q4E LoadTiming] explicit reads: %.1f s total",
                                          s_read_ms / 1000.0);
                        }
                    }
                }
            }
            // F1-style advisory prefetch (default on; =0 restores the historical
            // path). The four per-layer source ranges are contiguous mmap spans
            // that the fill and the GPU-owned staging copies will read exactly
            // once; queuing them up front lets the cache manager read with real
            // depth instead of faulting serially inside the memcpy loop.
            if (s_host_arena_prefetch && mmap_sources && !async_prefetch) {
                const auto pfv_t0 = std::chrono::steady_clock::now();
                prefetch_source_range(gu_src.packed_weight.ptr, gu_src.packed_weight.bytes,
                                      s_prefetch_chunk_bytes);
                prefetch_source_range(gu_src.absmax.ptr, gu_src.absmax.bytes,
                                      s_prefetch_chunk_bytes);
                prefetch_source_range(dn_src.packed_weight.ptr, dn_src.packed_weight.bytes,
                                      s_prefetch_chunk_bytes);
                prefetch_source_range(dn_src.absmax.ptr, dn_src.absmax.bytes,
                                      s_prefetch_chunk_bytes);
                if (load_timing) {
                    static double s_pfv_ms = 0.0;
                    s_pfv_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pfv_t0).count();
                    if (layer + 1 == config_.num_layers) {
                        AILA_LOG_WARN("[Q4E LoadTiming] host arena prefetch (PFV): %.1f s total",
                                      s_pfv_ms / 1000.0);
                    }
                }
            }
            if (async_prefetch) {
                // Layer 0 has no lookahead queued yet: hint it inline, the one
                // serialized enqueue the sync path makes on every layer.
                if (layer == 0 && mmap_sources) {
                    const auto pfv_t0 = std::chrono::steady_clock::now();
                    prefetch_source_range(gu_src.packed_weight.ptr, gu_src.packed_weight.bytes,
                                          s_prefetch_chunk_bytes);
                    prefetch_source_range(gu_src.absmax.ptr, gu_src.absmax.bytes,
                                          s_prefetch_chunk_bytes);
                    prefetch_source_range(dn_src.packed_weight.ptr, dn_src.packed_weight.bytes,
                                          s_prefetch_chunk_bytes);
                    prefetch_source_range(dn_src.absmax.ptr, dn_src.absmax.bytes,
                                          s_prefetch_chunk_bytes);
                    if (load_timing) {
                        pf_inline_ms += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - pfv_t0).count();
                    }
                }
                // Queue the L+1 (and, at lookahead 2, L+2) hint on the helper
                // slots; each runs during the layers before its target and is
                // joined at the top of the target layer's iteration. Depth 1
                // launches exactly the historical single target; depth 2 keeps
                // L+1..L+2 in flight on the two slots.
                if (mmap_sources) {
                    const int window_end = std::min(config_.num_layers - 1,
                                                    layer + s_prefetch_lookahead);
                    while (pf_next_target <= window_end) {
                        if (pf_target < 0) {
                            resolve_layer_refs(pf_next_target, pf_gu, pf_dn);
                            pf_launch(pf_thread, pf_gu, pf_dn);
                            pf_target = pf_next_target;
                        } else if (pf_target2 < 0) {
                            resolve_layer_refs(pf_next_target, pf_gu, pf_dn);
                            pf_launch(pf_thread2, pf_gu, pf_dn);
                            pf_target2 = pf_next_target;
                        } else {
                            break;
                        }
                        ++pf_next_target;
                    }
                }
            }
            // C7 deep dive: the RAM copies run at ~0.43 GB/s because every
            // copy faults in mmap source pages one after another. Warm the
            // whole layer's source ranges in parallel first so the OS runs the
            // disk reads concurrently; the copies then hit resident pages.
            static const int s_prefetch_threads = std::clamp(
                aila::env::read_int("AILA_Q4E_ARENA_FILL_PREFETCH", 0), 0, 16);
            const auto prefetch_t0 = std::chrono::steady_clock::now();
            // With explicit reads the mmap source pages are never touched, so
            // the prefetch would only add faults.
            if (s_prefetch_threads > 0 && s_read_threads == 0) {
                auto touch_range = [](const uint8_t* p, size_t bytes) {
                    volatile uint8_t sink = 0;
                    for (size_t off = 0; off < bytes; off += 4096) {
                        sink ^= p[off];
                    }
                    (void)sink;
                };
                const std::array<std::pair<const uint8_t*, size_t>, 4> ranges{{
                    {gu.packed_weight.ptr, gu.packed_weight.bytes},
                    {gu.absmax.ptr, gu.absmax.bytes},
                    {dn.packed_weight.ptr, dn.packed_weight.bytes},
                    {dn.absmax.ptr, dn.absmax.bytes}}};
                std::vector<std::thread> warm;
                warm.reserve(static_cast<size_t>(s_prefetch_threads));
                for (int t = 0; t < s_prefetch_threads; ++t) {
                    warm.emplace_back([&, t] {
                        for (const auto& range : ranges) {
                            const size_t per =
                                (range.second + s_prefetch_threads - 1) / s_prefetch_threads;
                            const size_t begin = std::min(range.second, per * t);
                            const size_t end = std::min(range.second, begin + per);
                            if (begin < end) touch_range(range.first + begin, end - begin);
                        }
                    });
                }
                for (auto& th : warm) th.join();
            }
            if (load_timing && s_prefetch_threads > 0) {
                static double s_prefetch_ms = 0.0;
                s_prefetch_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - prefetch_t0).count();
                if (layer + 1 == config_.num_layers) {
                    AILA_LOG_WARN("[Q4E LoadTiming] source prefetch (%d threads): %.1f s total",
                                  s_prefetch_threads, s_prefetch_ms / 1000.0);
                }
            }
            LoadLayerDirect(layer,gu_src,dn_src);
            // Per-layer lines go through the IPC log pipe; keep them sparse so
            // the diagnostic itself does not distort the timing.
            if (load_timing && (layer % 8 == 0 || layer + 1 == config_.num_layers)) {
                const auto now = std::chrono::steady_clock::now();
                AILA_LOG_WARN("[Q4E LoadTiming] arena layer %d: total %.3f s (weight refs %.3f s, copy+upload %.3f s)",
                              layer,
                              std::chrono::duration<double>(now - layer_t0).count(),
                              std::chrono::duration<double>(ref_done - layer_t0).count(),
                              std::chrono::duration<double>(now - ref_done).count());
            }
        }
        // No lookahead is pending after the last layer; symmetric guard.
        pf_join_thread(pf_thread);
        pf_join_thread(pf_thread2);
        if (load_timing && async_prefetch) {
            AILA_LOG_WARN("[Q4E LoadTiming] host arena prefetch (PFV async chunk %zu KiB): inline %.1f s, issue %.1f s, join %.1f s total, lookahead %d",
                          s_prefetch_chunk_bytes / 1024, pf_inline_ms / 1000.0,
                          pf_issue_ms / 1000.0, pf_join_ms / 1000.0,
                          s_prefetch_lookahead);
        }
        Queue().wait_and_throw();
        initial_gpu_owners_.clear(); // no runtime file-reader or mmap fallback
        return true;
    } catch (const std::exception& e) { if (error) *error=e.what(); return false; }
}

void Qwen4ExpHostArena::ExchangeOwner(int layer,int incoming,int outgoing) {
    auto* p=GetExpertPointer(layer,incoming);
    if (!p || GetExpertPointer(layer,outgoing) || outgoing<0 || outgoing>=config_.num_experts_per_layer)
        throw std::logic_error("Invalid Qwen4Exp RAM/GPU ownership exchange");
    expert_ptrs_[size_t(layer)*config_.num_experts_per_layer+outgoing]=p;
    expert_ptrs_[size_t(layer)*config_.num_experts_per_layer+incoming]=nullptr;
}

ops::MoeExpertSlotHost Qwen4ExpHostArena::GetExpertSlotHost(int layer,int expert) const noexcept {
    ops::MoeExpertSlotHost s{};
    const auto* p=GetExpertPointer(layer,expert);
    if (!p) return s;
    const auto& m=layer_nested_[layer];
    if (!m.valid) return s;
    s.gate_up_packed=p+geom_.gu_packed_offset; s.down_packed=p+geom_.dn_packed_offset;
    s.gate_up_nested.codes=p+geom_.gu_absmax_offset;
    s.down_nested.codes=p+geom_.dn_absmax_offset;
    s.gate_up_nested.nested_absmax=m.gu_nested_absmax.data()+size_t(expert)*geom_.gu_absmax_count/256;
    s.down_nested.nested_absmax=m.dn_nested_absmax.data()+size_t(expert)*geom_.dn_absmax_count/256;
    s.gate_up_nested.nested_quant_map=m.gu_nested_quant_map.data();
    s.down_nested.nested_quant_map=m.dn_nested_quant_map.data();
    s.gate_up_nested.offset=m.gu_nested_offset; s.down_nested.offset=m.dn_nested_offset;
    s.gate_up_nested.blocksize=s.down_nested.blocksize=256;
    return s;
}
} // namespace aila
