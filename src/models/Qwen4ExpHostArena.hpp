#pragma once
#include "core/Context.hpp"
#include "models/Qwen4ExpExpertReader.hpp"
#include "ops/MoeExpertOps.hpp"
#include "utils/SafeTensors.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aila {
struct HostArenaConfig {
    int num_layers = 48;
    int num_experts_per_layer = 512;
    int hidden_size = 2560;
    int moe_intermediate_size = 640;
    int blocksize = 64;
    bool compact_nested = true;
    size_t chunk_bytes = 256ULL * 1024 * 1024;
};

struct LayerNestedMetadata {
    std::vector<float> gu_nested_absmax, dn_nested_absmax;
    std::vector<float> gu_nested_quant_map, dn_nested_quant_map;
    float gu_nested_offset = 0, dn_nested_offset = 0;
    int gu_nested_blocksize = 256, dn_nested_blocksize = 256;
    bool valid = false;
    const float* gu_nested_absmax_dev = nullptr;
    const float* dn_nested_absmax_dev = nullptr;
    const float* gu_nested_quant_map_dev = nullptr;
    const float* dn_nested_quant_map_dev = nullptr;
};

// Native large-page RAM complement of the GPU cache. Native pointers are CPU
// storage only; all DMA uses small, explicitly owned USM transfer buffers.
class Qwen4ExpHostArena {
public:
    Qwen4ExpHostArena() = default;
    ~Qwen4ExpHostArena();
    Qwen4ExpHostArena(const Qwen4ExpHostArena&) = delete;
    Qwen4ExpHostArena& operator=(const Qwen4ExpHostArena&) = delete;
    Qwen4ExpHostArena(Qwen4ExpHostArena&& other) noexcept;
    Qwen4ExpHostArena& operator=(Qwen4ExpHostArena&& other) noexcept;

    bool Allocate(Context& ctx, const HostArenaConfig& cfg = {}, std::string* error = nullptr,
                  const std::vector<uint8_t*>& gpu_owners = {});
    bool Allocate(sycl::queue& q, const HostArenaConfig& cfg = {}, std::string* error = nullptr,
                  const std::vector<uint8_t*>& gpu_owners = {});
    void Release() noexcept;
    bool IsAllocated() const noexcept { return total_experts_ && expert_ptrs_.size() == total_experts_; }
    size_t TotalAllocatedBytes() const noexcept { return allocated_bytes_; }
    size_t NumChunks() const noexcept { return chunks_.size(); }
    size_t ResidentExperts() const noexcept { return resident_experts_; }
    const ExpertSlotGeometry& Geometry() const noexcept { return geom_; }
    size_t SlotBytes() const noexcept { return geom_.slot_bytes; }
    int NumLayers() const noexcept { return config_.num_layers; }
    int NumExpertsPerLayer() const noexcept { return config_.num_experts_per_layer; }
    size_t TotalExperts() const noexcept { return total_experts_; }

    uint8_t* GetExpertPointer(int layer, int expert) const noexcept {
        if (layer < 0 || layer >= config_.num_layers || expert < 0 ||
            expert >= config_.num_experts_per_layer || expert_ptrs_.empty()) return nullptr;
        return expert_ptrs_[size_t(layer) * config_.num_experts_per_layer + expert];
    }
    uint8_t* GetExpertGateUpPacked(int l, int e) const noexcept { return GetExpertPointer(l,e); }
    uint8_t* GetExpertGateUpAbsmax(int l, int e) const noexcept {
        auto* p=GetExpertPointer(l,e); return p ? p+geom_.gu_absmax_offset : nullptr;
    }
    uint8_t* GetExpertDownPacked(int l, int e) const noexcept {
        auto* p=GetExpertPointer(l,e); return p ? p+geom_.dn_packed_offset : nullptr;
    }
    uint8_t* GetExpertDownAbsmax(int l, int e) const noexcept {
        auto* p=GetExpertPointer(l,e); return p ? p+geom_.dn_absmax_offset : nullptr;
    }
    sycl::event DmaTransferToGpu(sycl::queue& q, void* dst, int layer, int expert) const;
    void LoadLayerDirect(int layer, const Bnb4BitHostWeightRef& gu, const Bnb4BitHostWeightRef& dn);
    bool LoadFromModelWeights(const ModelWeights& weights, std::string* error = nullptr);
    ops::MoeExpertSlotHost GetExpertSlotHost(int layer, int expert) const noexcept;
    const LayerNestedMetadata& GetLayerNestedMetadata(int layer) const noexcept { return layer_nested_[layer]; }
    // Called only after the outgoing GPU payload has been copied into the
    // incoming expert's former RAM slot and its H2D source has been preserved.
    void ExchangeOwner(int layer, int incoming_expert, int outgoing_expert);

private:
    bool AllocateStorage(const HostArenaConfig&, const std::vector<uint8_t*>&, std::string*);
    sycl::queue& Queue() const { return ctx_ ? ctx_->queue() : *standalone_queue_; }
    HostArenaConfig config_{};
    ExpertSlotGeometry geom_{};
    size_t total_experts_=0, resident_experts_=0, allocated_bytes_=0;
    Context* ctx_=nullptr;
    std::unique_ptr<sycl::queue> standalone_queue_;
    std::vector<uint8_t*> chunks_, expert_ptrs_, initial_gpu_owners_;
    std::vector<size_t> chunk_sizes_;
    uint8_t* load_staging_=nullptr;
    mutable sycl::event load_copy_event_{};
    mutable bool load_copy_pending_=false;
    std::vector<LayerNestedMetadata> layer_nested_;
    std::vector<void*> layer_nested_dev_allocs_;
};
} // namespace aila
