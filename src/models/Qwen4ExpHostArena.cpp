#include "models/Qwen4ExpHostArena.hpp"
#include "memory/LargePages.hpp"
#include "utils/EnvUtils.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#endif

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
    for (size_t e=0;e<experts;++e) {
        auto* dst=GetExpertPointer(layer,int(e));
        uint8_t* gpu=nullptr;
        if (!dst) {
            if (initial_gpu_owners_.empty()) throw std::logic_error("Missing Qwen4Exp expert owner while loading");
            gpu=initial_gpu_owners_[size_t(layer)*experts+e];
            if (!gpu) throw std::logic_error("Missing Qwen4Exp initial GPU owner");
            if (load_copy_pending_) load_copy_event_.wait_and_throw();
            dst=load_staging_;
        }
        std::memcpy(dst+geom_.gu_packed_offset,gu.packed_weight.ptr+e*geom_.gu_packed_bytes,geom_.gu_packed_bytes);
        std::memcpy(dst+geom_.gu_absmax_offset,gu.absmax.ptr+e*geom_.gu_absmax_count,geom_.gu_absmax_bytes);
        std::memcpy(dst+geom_.dn_packed_offset,dn.packed_weight.ptr+e*geom_.dn_packed_bytes,geom_.dn_packed_bytes);
        std::memcpy(dst+geom_.dn_absmax_offset,dn.absmax.ptr+e*geom_.dn_absmax_count,geom_.dn_absmax_bytes);
        if (gpu) {
            load_copy_event_=Queue().memcpy(gpu,dst,geom_.slot_bytes);
            load_copy_pending_=true;
        }
    }
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
    Queue().wait_and_throw();
    m.valid=true;
#ifdef _WIN32
    // Release consumed mmap source pages after each layer, before they compete
    // with the next layer. Native large pages are outside this working set.
    SetProcessWorkingSetSize(GetCurrentProcess(),SIZE_T(-1),SIZE_T(-1));
#endif
}

bool Qwen4ExpHostArena::LoadFromModelWeights(const ModelWeights& weights, std::string* error) {
    try {
        for (int layer=0;layer<config_.num_layers;++layer) {
            const auto prefix="model.language_model.layers."+std::to_string(layer)+".mlp.experts.";
            Bnb4BitHostWeightRef gu,dn;
            auto load=[&](const std::string& base,Bnb4BitHostWeightRef& ref,int hint) {
                const std::string name=weights.has_host(base) ? base : base+".weight";
                std::string detail;
                if (!LoadBnb4BitHostRef(weights,name,ref,&detail,hint)) throw std::runtime_error(detail);
            };
            load(prefix+"gate_up_proj",gu,config_.hidden_size);
            load(prefix+"down_proj",dn,config_.moe_intermediate_size);
            LoadLayerDirect(layer,gu,dn);
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
