#include "models/Qwen4ExpAdaptiveCache.hpp"
#include "models/Qwen4ExpExpertExecutor.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <utility>
#include <stdexcept>

namespace aila {
namespace {
uint64_t work_time_ns() {
    static const bool enabled = env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    return enabled ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()) : 0;
}
}

Qwen4ExpAdaptiveCache::~Qwen4ExpAdaptiveCache() {
    Release();
}

Qwen4ExpAdaptiveCache::Qwen4ExpAdaptiveCache(Qwen4ExpAdaptiveCache&& other) noexcept
    : config_(other.config_),
      geom_(other.geom_),
      num_slots_(other.num_slots_),
      total_experts_(other.total_experts_),
      ctx_(other.ctx_),
      standalone_queue_(std::move(other.standalone_queue_)),
      dev_pool_chunks_(std::move(other.dev_pool_chunks_)),
      slot_dev_base_(std::move(other.slot_dev_base_)),
      slot_table_(std::move(other.slot_table_)),
      slot_table_dev_(other.slot_table_dev_),
      slot_to_expert_(std::move(other.slot_to_expert_)),
      residency_generation_(other.residency_generation_),
      descriptor_generation_(other.descriptor_generation_),
      usage_(std::move(other.usage_)),
      access_count_(std::move(other.access_count_)),
      stats_(other.stats_)
{
    for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
        staging_dev_[i]=std::exchange(other.staging_dev_[i],nullptr);
        staging_host_[i]=std::exchange(other.staging_host_[i],nullptr);
        staging_experts_[i]=other.staging_experts_[i];
        staging_copy_[i]=other.staging_copy_[i]; staging_reader_[i]=other.staging_reader_[i];
        staging_pending_[i]=std::exchange(other.staging_pending_[i],false);
        staging_read_[i]=std::exchange(other.staging_read_[i],false);
    }
    resident_ready_=other.resident_ready_; resident_reader_=other.resident_reader_;
    resident_pending_=std::exchange(other.resident_pending_,false);
    resident_read_=std::exchange(other.resident_read_,false);

    other.ctx_ = nullptr;
    other.slot_table_dev_ = nullptr;
    other.num_slots_ = 0;
    other.total_experts_ = 0;
}

Qwen4ExpAdaptiveCache& Qwen4ExpAdaptiveCache::operator=(Qwen4ExpAdaptiveCache&& other) noexcept {
    if (this != &other) {
        Release();

        config_ = other.config_;
        geom_ = other.geom_;
        num_slots_ = other.num_slots_;
        total_experts_ = other.total_experts_;
        ctx_ = other.ctx_;
        standalone_queue_ = std::move(other.standalone_queue_);
        dev_pool_chunks_ = std::move(other.dev_pool_chunks_);
        slot_dev_base_ = std::move(other.slot_dev_base_);
    for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
        staging_dev_[i]=std::exchange(other.staging_dev_[i],nullptr);
        staging_host_[i]=std::exchange(other.staging_host_[i],nullptr);
        staging_experts_[i]=other.staging_experts_[i];
        staging_copy_[i]=other.staging_copy_[i]; staging_reader_[i]=other.staging_reader_[i];
        staging_pending_[i]=std::exchange(other.staging_pending_[i],false);
        staging_read_[i]=std::exchange(other.staging_read_[i],false);
    }
    resident_ready_=other.resident_ready_; resident_reader_=other.resident_reader_;
    resident_pending_=std::exchange(other.resident_pending_,false);
    resident_read_=std::exchange(other.resident_read_,false);
        slot_table_ = std::move(other.slot_table_);
        slot_table_dev_ = other.slot_table_dev_;
        slot_to_expert_ = std::move(other.slot_to_expert_);
        residency_generation_ = std::max(
            residency_generation_, other.residency_generation_ + 1);
        descriptor_generation_ = std::max(
            descriptor_generation_, other.descriptor_generation_ + 1);
        usage_ = std::move(other.usage_);
        access_count_ = std::move(other.access_count_);
        stats_ = other.stats_;

        other.ctx_ = nullptr;
        other.slot_table_dev_ = nullptr;
        other.num_slots_ = 0;
        other.total_experts_ = 0;
    }
    return *this;
}

void Qwen4ExpAdaptiveCache::InitGeometry() {
    const int gu_rows = 2 * config_.moe_intermediate_size;
    geom_.gu_packed_bytes = static_cast<size_t>(gu_rows) * (config_.hidden_size / 2);
    geom_.gu_absmax_count = static_cast<size_t>(gu_rows) * (config_.hidden_size / config_.blocksize);
    geom_.gu_absmax_bytes = geom_.gu_absmax_count * (config_.compact_nested ? sizeof(uint8_t) : sizeof(float));

    const int dn_rows = config_.hidden_size;
    geom_.dn_packed_bytes = static_cast<size_t>(dn_rows) * (config_.moe_intermediate_size / 2);
    geom_.dn_absmax_count = static_cast<size_t>(dn_rows) * (config_.moe_intermediate_size / config_.blocksize);
    geom_.dn_absmax_bytes = geom_.dn_absmax_count * (config_.compact_nested ? sizeof(uint8_t) : sizeof(float));

    geom_.compact_nested = config_.compact_nested;

    geom_.gu_packed_offset = 0;
    geom_.gu_absmax_offset = geom_.gu_packed_bytes;
    geom_.dn_packed_offset = geom_.gu_absmax_offset + geom_.gu_absmax_bytes;
    geom_.dn_absmax_offset = geom_.dn_packed_offset + geom_.dn_packed_bytes;
    geom_.slot_bytes = geom_.dn_absmax_offset + geom_.dn_absmax_bytes;
}

bool Qwen4ExpAdaptiveCache::Allocate(
    Context& ctx,
    const AdaptiveCacheConfig& cfg,
    std::string* error_message)
{
    Release();
    ctx_ = &ctx;
    standalone_queue_.reset();
    config_ = cfg;
    // Prefill supply depth (E6): number of staging banks. Default 2 keeps the
    // historical ping-pong layout and ordering byte-for-byte identical.
    config_.staging_banks = std::clamp(
        env::read_int("AILA_Q4E_PREFILL_STAGING_BANKS", config_.staging_banks),
        2, AdaptiveCacheConfig::kMaxStagingBanks);
    if (cfg.num_layers<=0 || cfg.num_experts_per_layer<=0 || cfg.hidden_size<=0 ||
        cfg.moe_intermediate_size<=0 || cfg.blocksize!=64 || !cfg.compact_nested ||
        cfg.hidden_size%64 || cfg.moe_intermediate_size%64 || cfg.staging_slots_per_buffer<1 ||
        cfg.staging_slots_per_buffer>AdaptiveCacheConfig::kMaxStagingSlots || cfg.num_gpu_slots>cfg.num_layers*cfg.num_experts_per_layer) {
        if (error_message) *error_message="Invalid Qwen4Exp adaptive cache geometry";
        return false;
    }
    InitGeometry();

    num_slots_ = config_.num_gpu_slots;
    total_experts_ = static_cast<size_t>(config_.num_layers) * static_cast<size_t>(config_.num_experts_per_layer);

    if (num_slots_ <= 0 || total_experts_ == 0 || geom_.slot_bytes == 0) {
        if (error_message) *error_message = "Qwen4ExpAdaptiveCache: invalid slot count or geometry configuration";
        return false;
    }

    const size_t staging_bytes = static_cast<size_t>(config_.staging_slots_per_buffer) * geom_.slot_bytes;
    const size_t table_dev_bytes = total_experts_ * sizeof(int32_t);

    try {
        // Chunk resident pool into <= 1GB allocations to comply with Level Zero / Arc limits
        const int slots_per_chunk = std::max(1, static_cast<int>(1024ULL * 1024 * 1024 / geom_.slot_bytes));
        slot_dev_base_.resize(num_slots_, nullptr);
        for (int s0 = 0; s0 < num_slots_; s0 += slots_per_chunk) {
            const int n = std::min(slots_per_chunk, num_slots_ - s0);
            const size_t chunk_bytes = static_cast<size_t>(n) * geom_.slot_bytes;
            uint8_t* chunk = static_cast<uint8_t*>(ctx_->alloc_device(chunk_bytes));
            dev_pool_chunks_.push_back(chunk);
            for (int i = 0; i < n; ++i) {
                slot_dev_base_[s0 + i] = chunk + static_cast<size_t>(i) * geom_.slot_bytes;
            }
        }
        for (int b=0;b<config_.staging_banks;++b)
            staging_dev_[b] = static_cast<uint8_t*>(ctx_->alloc_device(staging_bytes));
        slot_table_dev_ = static_cast<int32_t*>(ctx_->alloc_device(table_dev_bytes));
        for (int b=0;b<config_.staging_banks;++b)
            staging_host_[b]=static_cast<uint8_t*>(ctx_->alloc_host(staging_bytes));
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = std::string("Qwen4ExpAdaptiveCache: device allocation failed: ") + e.what();
        }
        Release();
        return false;
    }

    slot_table_.assign(total_experts_, kAdaptiveNotResident);
    slot_to_expert_.assign(num_slots_, ResidentSlotEntry{-1, -1});
    usage_.assign(total_experts_, 0.0f);
    access_count_.assign(total_experts_, 0);

    // Initialize device slot table to kAdaptiveNotResident
    ctx_->queue().fill(slot_table_dev_, kAdaptiveNotResident, total_experts_).wait_and_throw();

    BumpResidentMappingGenerations();
    return true;
}

bool Qwen4ExpAdaptiveCache::Allocate(
    sycl::queue& q,
    const AdaptiveCacheConfig& cfg,
    std::string* error_message)
{
    Release();
    ctx_ = nullptr;
    standalone_queue_ = std::make_unique<sycl::queue>(q);
    config_ = cfg;
    // Prefill supply depth (E6): see the Context overload above.
    config_.staging_banks = std::clamp(
        env::read_int("AILA_Q4E_PREFILL_STAGING_BANKS", config_.staging_banks),
        2, AdaptiveCacheConfig::kMaxStagingBanks);
    if (cfg.num_layers<=0 || cfg.num_experts_per_layer<=0 || cfg.hidden_size<=0 ||
        cfg.moe_intermediate_size<=0 || cfg.blocksize!=64 || !cfg.compact_nested ||
        cfg.hidden_size%64 || cfg.moe_intermediate_size%64 || cfg.staging_slots_per_buffer<1 ||
        cfg.staging_slots_per_buffer>AdaptiveCacheConfig::kMaxStagingSlots || cfg.num_gpu_slots>cfg.num_layers*cfg.num_experts_per_layer) {
        if (error_message) *error_message="Invalid Qwen4Exp adaptive cache geometry";
        return false;
    }
    InitGeometry();

    num_slots_ = config_.num_gpu_slots;
    total_experts_ = static_cast<size_t>(config_.num_layers) * static_cast<size_t>(config_.num_experts_per_layer);

    if (num_slots_ <= 0 || total_experts_ == 0 || geom_.slot_bytes == 0) {
        if (error_message) *error_message = "Qwen4ExpAdaptiveCache: invalid slot count or geometry configuration";
        return false;
    }

    const size_t staging_bytes = static_cast<size_t>(config_.staging_slots_per_buffer) * geom_.slot_bytes;
    const size_t table_dev_bytes = total_experts_ * sizeof(int32_t);

    try {
        const int slots_per_chunk = std::max(1, static_cast<int>(1024ULL * 1024 * 1024 / geom_.slot_bytes));
        slot_dev_base_.resize(num_slots_, nullptr);
        for (int s0 = 0; s0 < num_slots_; s0 += slots_per_chunk) {
            const int n = std::min(slots_per_chunk, num_slots_ - s0);
            const size_t chunk_bytes = static_cast<size_t>(n) * geom_.slot_bytes;
            uint8_t* chunk = sycl::malloc_device<uint8_t>(chunk_bytes, *standalone_queue_);
            if (!chunk) throw std::runtime_error("malloc_device returned nullptr");
            dev_pool_chunks_.push_back(chunk);
            for (int i = 0; i < n; ++i) {
                slot_dev_base_[s0 + i] = chunk + static_cast<size_t>(i) * geom_.slot_bytes;
            }
        }
        for (int b=0;b<config_.staging_banks;++b)
            staging_dev_[b] = sycl::malloc_device<uint8_t>(staging_bytes, *standalone_queue_);
        slot_table_dev_ = sycl::malloc_device<int32_t>(total_experts_, *standalone_queue_);
        for (int b=0;b<config_.staging_banks;++b)
            staging_host_[b]=sycl::malloc_host<uint8_t>(staging_bytes,*standalone_queue_);
    } catch (const std::exception& e) {
        if (error_message) {
            *error_message = std::string("Qwen4ExpAdaptiveCache: standalone device allocation failed: ") + e.what();
        }
        Release();
        return false;
    }

    for (int b=0;b<config_.staging_banks;++b)
        if (!staging_dev_[b] || !staging_host_[b]) {
            if (error_message) *error_message = "Qwen4ExpAdaptiveCache: malloc_device returned nullptr";
            Release();
            return false;
        }
    if (!slot_table_dev_) {
        if (error_message) *error_message = "Qwen4ExpAdaptiveCache: malloc_device returned nullptr";
        Release();
        return false;
    }

    slot_table_.assign(total_experts_, kAdaptiveNotResident);
    slot_to_expert_.assign(num_slots_, ResidentSlotEntry{-1, -1});
    usage_.assign(total_experts_, 0.0f);
    access_count_.assign(total_experts_, 0);

    // Initialize device slot table to kAdaptiveNotResident
    standalone_queue_->fill(slot_table_dev_, kAdaptiveNotResident, total_experts_).wait_and_throw();

    BumpResidentMappingGenerations();
    return true;
}

void Qwen4ExpAdaptiveCache::Release() noexcept {
    BumpResidentMappingGenerations();
    // Copy queue is independent of the compute queue. Drain both before any
    // USM or native source can be released, including partial startup failures.
    for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
        if (staging_pending_[i]) { try { staging_copy_[i].wait_and_throw(); } catch (...) {} }
        if (staging_read_[i]) { try { staging_reader_[i].wait_and_throw(); } catch (...) {} }
        staging_pending_[i]=staging_read_[i]=false;
    }
    if (resident_pending_) { try { resident_ready_.wait_and_throw(); } catch (...) {} }
    if (resident_read_) { try { resident_reader_.wait_and_throw(); } catch (...) {} }
    resident_pending_=resident_read_=false;
    for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
        if (staging_host_[i]) {
            try { if (ctx_) ctx_->free_host(staging_host_[i]); else sycl::free(staging_host_[i],*standalone_queue_); } catch (...) {}
            staging_host_[i]=nullptr;
        }
    }
    if (ctx_) {
        try { ctx_->synchronize(); } catch (...) {}
        for (auto* chunk : dev_pool_chunks_) {
            if (chunk) {
                try { ctx_->free_device(chunk); } catch (...) {}
            }
        }
        for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
            if (staging_dev_[i]) { try { ctx_->free_device(staging_dev_[i]); } catch (...) {} staging_dev_[i] = nullptr; }
        }
        if (slot_table_dev_) { try { ctx_->free_device(slot_table_dev_); } catch (...) {} slot_table_dev_ = nullptr; }
        ctx_ = nullptr;
    } else if (standalone_queue_) {
        try { standalone_queue_->wait(); } catch (...) {}
        for (auto* chunk : dev_pool_chunks_) {
            if (chunk) sycl::free(chunk, *standalone_queue_);
        }
        for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
            if (staging_dev_[i]) { sycl::free(staging_dev_[i], *standalone_queue_); staging_dev_[i] = nullptr; }
        }
        if (slot_table_dev_) { sycl::free(slot_table_dev_, *standalone_queue_); slot_table_dev_ = nullptr; }
        standalone_queue_.reset();
    }

    dev_pool_chunks_.clear();
    slot_dev_base_.clear();
    slot_table_.clear();
    slot_to_expert_.clear();
    usage_.clear();
    access_count_.clear();
    num_slots_ = 0;
    total_experts_ = 0;
}

namespace {
ops::MoeExpertSlotDev make_slot(const uint8_t* base, const ExpertSlotGeometry& g,
                               int expert, const LayerNestedMetadata* m) {
    ops::MoeExpertSlotDev d{};
    if (!base) return d;
    d.gate_up_packed=base+g.gu_packed_offset; d.down_packed=base+g.dn_packed_offset;
    d.gate_up_nested.codes=base+g.gu_absmax_offset; d.down_nested.codes=base+g.dn_absmax_offset;
    if (!m) return d; // raw payload inspection only, never used by production kernels
    if (!m->valid || expert<0 || !m->gu_nested_absmax_dev || !m->dn_nested_absmax_dev ||
        !m->gu_nested_quant_map_dev || !m->dn_nested_quant_map_dev ||
        m->gu_nested_blocksize!=256 || m->dn_nested_blocksize!=256 ||
        (size_t(expert)+1)*g.gu_absmax_count/256>m->gu_nested_absmax.size() ||
        (size_t(expert)+1)*g.dn_absmax_count/256>m->dn_nested_absmax.size())
        throw std::logic_error("Qwen4Exp GPU descriptor requires this expert's device nested metadata");
    d.gate_up_nested.nested_absmax=m->gu_nested_absmax_dev+size_t(expert)*g.gu_absmax_count/256;
    d.down_nested.nested_absmax=m->dn_nested_absmax_dev+size_t(expert)*g.dn_absmax_count/256;
    d.gate_up_nested.nested_quant_map=m->gu_nested_quant_map_dev;
    d.down_nested.nested_quant_map=m->dn_nested_quant_map_dev;
    d.gate_up_nested.offset=m->gu_nested_offset; d.down_nested.offset=m->dn_nested_offset;
    d.gate_up_nested.blocksize=d.down_nested.blocksize=256;
    return d;
}
}
ops::MoeExpertSlotDev Qwen4ExpAdaptiveCache::GetResidentSlotDev(
    int slot, const LayerNestedMetadata* nested) const {
    return make_slot(GetSlotDevicePtr(slot),geom_,GetResidentEntry(slot).expert_id,nested);
}
ops::MoeExpertSlotDev Qwen4ExpAdaptiveCache::GetStagingSlotDev(
    int pp, int slot, const LayerNestedMetadata* nested) const {
    if (slot<0 || slot>=config_.staging_slots_per_buffer) return {};
    const int bank=(pp%config_.staging_banks+config_.staging_banks)%config_.staging_banks;
    return make_slot(GetStagingSlotDevicePtr(bank,slot),geom_,staging_experts_[bank][slot],nested);
}

std::vector<uint8_t*> Qwen4ExpAdaptiveCache::InitializeResidency(
    const std::vector<std::pair<int,int>>* profile) {
    if (!IsAllocated()) throw std::logic_error("Adaptive cache is not allocated");
    // Invalidate even if validation below rejects a partially edited profile.
    BumpResidentMappingGenerations();
    std::fill(slot_table_.begin(),slot_table_.end(),kAdaptiveNotResident);
    std::fill(slot_to_expert_.begin(),slot_to_expert_.end(),ResidentSlotEntry{});
    std::vector<uint8_t*> owners(total_experts_,nullptr);
    int slot=0;
    auto add=[&](int l,int e,bool strict) {
        if (l<0 || l>=config_.num_layers || e<0 || e>=config_.num_experts_per_layer)
            throw std::invalid_argument("Invalid initial resident expert");
        const size_t key=size_t(l)*config_.num_experts_per_layer+e;
        if (owners[key]) {
            if (strict) throw std::invalid_argument("Duplicate initial resident expert");
            return;
        }
        if (slot==num_slots_) return;
        owners[key]=GetSlotDevicePtr(slot); slot_table_[key]=slot;
        slot_to_expert_[slot++]={l,e};
    };
    if (profile) {
        for (const auto& key:*profile) add(key.first,key.second,true);
    } else if (config_.num_layers==48 && config_.num_experts_per_layer==512) {
        for (int r=0;r<kDefaultHotExpertsPerLayer;++r)
            for (int l=0;l<48;++l) add(l,kQwen4ExpDefaultHotExperts[l][r],false);
    }
    // Enumerate each key exactly once. Skipping a hot expert must not load the
    // next non-hot expert twice (the previous current_expert loop did so).
    for (int e=0;e<config_.num_experts_per_layer && slot<num_slots_;++e)
        for (int l=0;l<config_.num_layers && slot<num_slots_;++l) add(l,e,false);
    return owners;
}
void Qwen4ExpAdaptiveCache::FinishInitialLoad(sycl::queue& q) {
    SyncSlotTableToDevice(q).wait_and_throw();
    stats_.warm_bytes_transferred+=TotalResidentBytes();
}
bool Qwen4ExpAdaptiveCache::Warmup(const Qwen4ExpHostArena& arena, sycl::queue& q,
    const std::vector<std::pair<int,int>>* profile, std::string* error) {
    try {
        InitializeResidency(profile);
        for (int s=0;s<num_slots_;++s) {
            const auto key=slot_to_expert_[s];
            arena.DmaTransferToGpu(q,GetSlotDevicePtr(s),key.layer,key.expert_id).wait_and_throw();
        }
        FinishInitialLoad(q);
        return true;
    } catch (const std::exception& e) { if (error) *error=e.what(); return false; }
}

sycl::event Qwen4ExpAdaptiveCache::SyncSlotTableToDevice(sycl::queue& q) {
    if (!slot_table_dev_ || slot_table_.empty()) return sycl::event{};
    BumpDescriptorGeneration();
    return q.memcpy(slot_table_dev_, slot_table_.data(), slot_table_.size() * sizeof(int32_t));
}

RouteResolution Qwen4ExpAdaptiveCache::ResolveRoute(int layer, const int* expert_ids, int count) const {
    RouteResolution res;
    if (!expert_ids || count <= 0) return res;

    res.hit_expert_ids.reserve(count);
    res.hit_slots.reserve(count);
    res.miss_expert_ids.reserve(count);

    for (int i = 0; i < count; ++i) {
        const int e = expert_ids[i];
        const int slot = GetSlot(layer, e);
        if (slot >= 0) {
            res.num_hits++;
            res.hit_expert_ids.push_back(e);
            res.hit_slots.push_back(slot);
        } else {
            res.num_misses++;
            res.miss_expert_ids.push_back(e);
        }
    }
    return res;
}

void Qwen4ExpAdaptiveCache::ObserveRouting(int layer, const int* expert_ids, int count, float weight) {
    if (!expert_ids || count <= 0 || layer < 0 || layer >= config_.num_layers) return;

    stats_.route_queries += count;
    const size_t layer_offset = static_cast<size_t>(layer) * static_cast<size_t>(config_.num_experts_per_layer);

    for (int i = 0; i < count; ++i) {
        const int e = expert_ids[i];
        if (e < 0 || e >= config_.num_experts_per_layer) continue;

        const size_t idx = layer_offset + static_cast<size_t>(e);
        usage_[idx] += weight;
        access_count_[idx]++;

        if (slot_table_[idx] >= 0) {
            stats_.cache_hits++;
        } else {
            stats_.cache_misses++;
        }
    }
}

void Qwen4ExpAdaptiveCache::RecordAccess(int layer, int expert_id, float weight) {
    if (layer < 0 || layer >= config_.num_layers ||
        expert_id < 0 || expert_id >= config_.num_experts_per_layer) {
        return;
    }

    stats_.route_queries++;
    const size_t idx = static_cast<size_t>(layer) * static_cast<size_t>(config_.num_experts_per_layer) + static_cast<size_t>(expert_id);
    usage_[idx] += weight;
    access_count_[idx]++;

    if (slot_table_[idx] >= 0) {
        stats_.cache_hits++;
    } else {
        stats_.cache_misses++;
    }
}

void Qwen4ExpAdaptiveCache::Decay(float factor) {
    const float f = factor > 0.0f ? factor : config_.decay_factor;
    for (float& u : usage_) {
        u *= f;
    }
}

float Qwen4ExpAdaptiveCache::GetUsage(int layer, int expert_id) const {
    if (layer < 0 || layer >= config_.num_layers ||
        expert_id < 0 || expert_id >= config_.num_experts_per_layer) {
        return 0.0f;
    }
    return usage_[static_cast<size_t>(layer) * static_cast<size_t>(config_.num_experts_per_layer) + static_cast<size_t>(expert_id)];
}

std::vector<AdaptiveSwapPlan> Qwen4ExpAdaptiveCache::PlanSwaps(
    int max_swaps_per_layer,
    int max_total_swaps) const
{
    std::vector<AdaptiveSwapPlan> plans;
    if (!IsAllocated()) return plans;

    for (int l = 0; l < config_.num_layers; ++l) {
        std::vector<int> victims;
        std::vector<int> candidates;
        const size_t layer_offset = static_cast<size_t>(l) * static_cast<size_t>(config_.num_experts_per_layer);

        for (int e = 0; e < config_.num_experts_per_layer; ++e) {
            const size_t idx = layer_offset + static_cast<size_t>(e);
            const int slot = slot_table_[idx];
            if (slot >= 0) {
                victims.push_back(e);
            } else if (usage_[idx] >= config_.candidate_min_usage) {
                candidates.push_back(e);
            }
        }

        if (victims.empty() || candidates.empty()) continue;

        // Sort victims ascending by usage (coldest first; tie breaker: expert_id)
        std::sort(victims.begin(), victims.end(), [this, layer_offset](int a, int b) {
            const float ua = usage_[layer_offset + static_cast<size_t>(a)];
            const float ub = usage_[layer_offset + static_cast<size_t>(b)];
            if (ua != ub) return ua < ub;
            return a < b;
        });

        // Sort candidates descending by usage (hottest first; tie breaker: expert_id)
        std::sort(candidates.begin(), candidates.end(), [this, layer_offset](int a, int b) {
            const float ua = usage_[layer_offset + static_cast<size_t>(a)];
            const float ub = usage_[layer_offset + static_cast<size_t>(b)];
            if (ua != ub) return ua > ub;
            return a < b;
        });

        int layer_swaps = 0;
        const size_t count = std::min(victims.size(), candidates.size());
        for (size_t i = 0; i < count; ++i) {
            const int cand = candidates[i];
            const int vic = victims[i];
            const float u_cand = usage_[layer_offset + static_cast<size_t>(cand)];
            const float u_vic = usage_[layer_offset + static_cast<size_t>(vic)];
            const float gain = u_cand - u_vic;

            // Hysteresis threshold Δ > 1.5 enforced to avoid thrashing
            if (gain <= config_.hysteresis_threshold) {
                break;
            }

            const int vic_slot = slot_table_[layer_offset + static_cast<size_t>(vic)];
            plans.push_back(AdaptiveSwapPlan{l, cand, vic, vic_slot, gain});
            layer_swaps++;
            if (max_swaps_per_layer > 0 && layer_swaps >= max_swaps_per_layer) {
                break;
            }
        }
    }

    // Sort all plans across layers by gain descending
    std::sort(plans.begin(), plans.end(), [](const AdaptiveSwapPlan& a, const AdaptiveSwapPlan& b) {
        if (a.gain != b.gain) return a.gain > b.gain;
        return a.layer < b.layer;
    });

    if (max_total_swaps > 0 && plans.size() > static_cast<size_t>(max_total_swaps)) {
        plans.resize(max_total_swaps);
    }

    return plans;
}

sycl::event Qwen4ExpAdaptiveCache::ExecuteSwapsAsync(
    Qwen4ExpHostArena& arena, sycl::queue& q, const std::vector<AdaptiveSwapPlan>& plans) {
    if (plans.empty()) return {};
    const auto begin = work_time_ns();
    if (!q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Qwen4Exp copy queue must be in order");
    // CPU exchanges reuse the same bounded staging banks; drain their H2D
    // readers before touching host bytes. GPU slot writes wait for consumers.
    for (int i=0;i<AdaptiveCacheConfig::kMaxStagingBanks;++i) {
        if (staging_pending_[i]) staging_copy_[i].wait_and_throw();
        staging_pending_[i]=false;
    }
    if (resident_pending_) resident_ready_.wait_and_throw();
    if (resident_read_) {
        q.submit([&](sycl::handler& cgh) { cgh.depends_on(resident_reader_); cgh.single_task([] {}); });
    }
    // A later plan may fail validation after earlier slots have changed.
    BumpResidentMappingGenerations();
    for (const auto& p:plans) {
        if (GetSlot(p.layer,p.victim_expert)!=p.victim_slot || GetSlot(p.layer,p.candidate_expert)>=0)
            throw std::logic_error("Stale adaptive swap plan");
        auto* source=arena.GetExpertPointer(p.layer,p.candidate_expert);
        auto* target=GetSlotDevicePtr(p.victim_slot);
        if (!source || !target) throw std::logic_error("Adaptive swap has no source/destination owner");
        std::memcpy(staging_host_[0],source,geom_.slot_bytes);
        if (!arena.GetExpertPointer(p.layer,p.victim_expert)) {
            // Preserve the outgoing expert's sole copy in the incoming RAM slot.
            q.memcpy(staging_host_[1],target,geom_.slot_bytes).wait_and_throw();
            std::memcpy(source,staging_host_[1],geom_.slot_bytes);
            arena.ExchangeOwner(p.layer,p.candidate_expert,p.victim_expert);
        }
        resident_ready_=q.memcpy(target,staging_host_[0],geom_.slot_bytes);
        resident_pending_=true;
        // This one-slot CPU buffer cannot be overwritten while DMA consumes it.
        resident_ready_.wait_and_throw();
        const size_t base=size_t(p.layer)*config_.num_experts_per_layer;
        slot_table_[base+p.victim_expert]=-1;
        slot_table_[base+p.candidate_expert]=p.victim_slot;
        slot_to_expert_[p.victim_slot]={p.layer,p.candidate_expert};
        stats_.swap_bytes_transferred+=geom_.slot_bytes; ++stats_.swaps_executed;
    }
    resident_ready_=SyncSlotTableToDevice(q);
    resident_pending_=true;
    stats_.swap_host_ns += work_time_ns() - begin;
    return resident_ready_;
}

StagingBatch Qwen4ExpAdaptiveCache::StageColdExpertsAsync(
    const Qwen4ExpHostArena& arena, sycl::queue& q, int layer, const int* experts, int count, int index,
    Qwen4ExpExpertExecutor* copy_executor, bool direct_arena) {
    if (!IsAllocated() || !experts || count<1 || count>config_.staging_slots_per_buffer ||
        layer<0 || layer>=config_.num_layers || arena.SlotBytes()!=geom_.slot_bytes ||
        !q.has_property<sycl::property::queue::in_order>())
        throw std::invalid_argument("Invalid Qwen4Exp cold staging batch");
    const int pp=(index%config_.staging_banks+config_.staging_banks)%config_.staging_banks;
    const auto wait_begin = work_time_ns();
    if (staging_pending_[pp]) staging_copy_[pp].wait_and_throw();
    const auto fill_begin = work_time_ns();
    stats_.staging_wait_ns += fill_begin - wait_begin;
    StagingBatch batch{};
    batch.ping_pong_index=pp; batch.layer=layer; batch.count=count;
    std::array<const uint8_t*,AdaptiveCacheConfig::kMaxStagingSlots> sources{};
    // Validate the complete batch before any copy task can start, so a missing
    // RAM owner cannot leave a partially refreshed staging bank.
    for (int i=0;i<count;++i) {
        sources[i]=arena.GetExpertPointer(layer,experts[i]);
        if (!sources[i]) throw std::logic_error("Cold expert is missing its RAM owner");
    }
    // Adopted default-on 2026-10-16: with the expert-XMX path this cut the
    // steady 4096-row block 17.0s -> 14.6s and the 14K TTFT 64.0s -> 54.1s
    // (fill 7.4s -> 2.3s, bank_wait 9ms -> 2.1s); confirmed by xpb-b1/b2.
    // AILA_Q4E_STAGE_COPY_PARALLEL=0 restores the serial host copy.
    static const bool parallel_host_copy =
        aila::env::read_flag("AILA_Q4E_STAGE_COPY_PARALLEL", true);
    if (!direct_arena) {
        if (parallel_host_copy && count>1 && copy_executor) {
            // The executor's parallel host copy is bounded to 16 slots per call;
            // deeper staging banks (AILA_Q4E_LAYER_MAJOR_STAGING_SLOTS) must split
            // into several calls instead of tripping its input validation.
            constexpr int kMaxHostCopySlots = 16;
            for (int offset = 0; offset < count; offset += kMaxHostCopySlots) {
                const int n = std::min(kMaxHostCopySlots, count - offset);
                copy_executor->CopyColdStagingHostSlots(
                    sources.data() + offset, staging_host_[pp] + size_t(offset) * geom_.slot_bytes,
                    geom_.slot_bytes, n, true);
            }
        } else {
            for (int i=0;i<count;++i)
                std::memcpy(staging_host_[pp]+size_t(i)*geom_.slot_bytes,
                            sources[i],geom_.slot_bytes);
        }
    }
    for (int i=0;i<count;++i) {
        staging_experts_[pp][i]=batch.expert_ids[i]=experts[i];
        batch.staging_slots[i]=i;
    }
    BumpDescriptorGeneration();
    stats_.staging_fill_ns += work_time_ns() - fill_begin;
    // The CPU can prepare the next host bank while the GPU still consumes the
    // device bank. Only the device write depends on its prior consumer.
    if (direct_arena) {
        // The large-page arena is a host USM source that the copy engine can
        // read directly (9.05-9.23 GB/s measured). Keep the same in-order
        // staging bank and prior-consumer dependency, but do not perform the
        // 5.45 GB/s coordinator memcpy that made the cap=4 split negative.
        //
        // Consecutive cold experts in id order are usually consecutive arena
        // slots (the arena is laid out layer-major/expert-major and only the
        // GPU-resident experts leave gaps), so a run of them is one contiguous
        // source range and one contiguous destination range. Collapsing those
        // runs into a single memcpy lifts each transfer from one 2.5 MB slot to
        // ~20 MB, which is where this platform's DMA stops losing efficiency
        // (probe: 2.4 MB chunks 8.69 GB/s, 64 MB 9.76 GB/s) and cuts the host
        // enqueue count per chunk from 16 to ~2.
        sycl::event last{};
        int merged_copies = 0;
        for (int i = 0; i < count;) {
            int run = 1;
            while (i + run < count &&
                   sources[i + run] == sources[i] + size_t(run) * geom_.slot_bytes)
                ++run;
            const size_t run_bytes = size_t(run) * geom_.slot_bytes;
            if (i == 0 && staging_read_[pp]) {
                last = q.memcpy(staging_dev_[pp], sources[0], run_bytes,
                                std::vector<sycl::event>{staging_reader_[pp]});
            } else {
                last = q.memcpy(staging_dev_[pp] + size_t(i) * geom_.slot_bytes,
                               sources[i], run_bytes);
            }
            ++merged_copies;
            i += run;
        }
        stats_.staging_copies_merged += static_cast<uint64_t>(merged_copies);
        batch.ready_event = last;
    } else {
        batch.ready_event=q.submit([&](sycl::handler& cgh) {
            if (staging_read_[pp]) cgh.depends_on(staging_reader_[pp]);
            cgh.memcpy(staging_dev_[pp],staging_host_[pp],size_t(count)*geom_.slot_bytes);
        });
    }
    staging_copy_[pp]=batch.ready_event; staging_pending_[pp]=true;
    stats_.cold_bytes_transferred+=size_t(count)*geom_.slot_bytes;
    stats_.cold_experts_staged+=count;
    return batch;
}
} // namespace aila
