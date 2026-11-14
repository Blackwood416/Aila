#pragma once

// Checkpoint-backed Qwen4Exp MTP (multi-token prediction) draft head.
// Semantics source: vLLM qwen4_exp/nvidia/mtp.py, residual_linear_shared,
// scheme A. Target and draft share the embedding table and LM head, but never
// KV state. The shared instance owns the draft weights, projections, its own
// 64-slot GPU expert cache and reusable decode scratch; each State owns only
// the draft KV and indexer tensors.

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/MoeExpertOps.hpp"
#include "../ops/Ops.hpp"
#include "../utils/SafeTensors.hpp"
#include "Qwen4ExpBnb4Backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class Qwen4ExpMtpDraft {
public:
    struct State {
        Qwen4ExpLayerCache cache;
    };

    Qwen4ExpMtpDraft(Context& ctx, Qwen4ExpBnb4Backend& target, ModelWeights& weights);
    ~Qwen4ExpMtpDraft();

    std::unique_ptr<State> CreateState(int capacity);

    void Step(State& state, const sycl::ext::oneapi::bfloat16* hidden, int token,
              int position, bool kvOnly = false,
              void (*onBeforeCpuCold)(void*) = nullptr, void* onBeforeCpuColdCtx = nullptr);
    void StepBatchKvOnly(State& state, const sycl::ext::oneapi::bfloat16* hidden,
                         const int* tokens, int count, int startPosition);

private:
    static constexpr int kMtpExperts = 512;
    static constexpr int kMtpExpertSlots = 64;
    static constexpr int kMtpExpertsPerStep = 10;
    static constexpr int kMtpHidden = 2560;
    static constexpr int kMtpIntermediate = 640;
    static constexpr int kMtpBlocksize = 64;
    static constexpr int kMtpNestedBlocksize = 256;
    static constexpr size_t kMtpExpertSlotBytes = 2534400;

    struct SlotLayout {
        size_t gu_packed_bytes = 0;
        size_t gu_absmax_bytes = 0;
        size_t dn_packed_bytes = 0;
        size_t dn_absmax_bytes = 0;
        size_t gu_packed_offset = 0;
        size_t gu_absmax_offset = 0;
        size_t dn_packed_offset = 0;
        size_t dn_absmax_offset = 0;
        size_t slot_bytes = 0;
    };

    struct SlotEntry {
        int expert = -1;
        uint64_t last_used = 0;
        sycl::event reader{};
        bool reader_valid = false;
    };

    static void Require(bool ok, const std::string& message);
    Tensor Allocate(int rows, int cols,
                    dnnl::memory::data_type type = dnnl::memory::data_type::bf16);
    Tensor* Norm(const std::string& name, bool addOneBf16 = false);
    void Quant(const std::string& name, Bnb4BitLinear& projection);
    void EraseFusedSource(const std::string& name);
    void Hc(const std::string& prefix, Qwen4ExpHyperConnectionLayer& hc, bool inject);

    void InitExpertCache();
    void ReleaseExpertCache() noexcept;
    sycl::event AcquireExperts(const int* expert_ids, int count, int* out_slots);
    void StageExpert(uint8_t* dst, int expert) const;
    void BuildSlotDev(int slot, int expert, ops::MoeExpertSlotDev& out) const;
    void RecordSlotReaders(const int* slots, int count, const sycl::event& consumer);

    Context& ctx_;
    Qwen4ExpBnb4Backend& target_;
    ModelWeights& weights_;

    Qwen4ExpLayer layer_;
    Qwen4ExpHyperConnectionLayer mixer_;
    Bnb4BitLinear fcEmbedding_;
    Bnb4BitLinear fcHidden_;
    Tensor* normEmbedding_ = nullptr;
    Tensor* normHidden_ = nullptr;
    Tensor embed_;
    Tensor embedNorm_;
    Tensor embedFc_;
    Tensor hiddenNorm_;
    Tensor draftH_;
    Tensor draftDown_;
    // USM host staging for one embedding row. A dedicated host block plus the
    // async H2D event keeps the row upload off the compute queue's critical
    // path: the blocking ctx_.memcpy_h2d would otherwise drain the queue on
    // every draft step.
    static constexpr int kMtpHostEmbedRing = 64;
    sycl::ext::oneapi::bfloat16* hostEmbed_ = nullptr;
    std::array<sycl::event, kMtpHostEmbedRing> hostEmbedEvents_{};
    uint64_t hostEmbedTick_ = 0;
    HostTensorView embeddingTable_;

    static constexpr int kMtpBatchEmbedTile = 512;
    sycl::ext::oneapi::bfloat16* batchEmbedHost_ = nullptr;
    sycl::event batchEmbedEvent_{};
    void FreeHostEmbed() noexcept;

    SlotLayout layout_{};
    Bnb4BitHostWeightRef gu_{};
    Bnb4BitHostWeightRef dn_{};
    uint8_t* slot_pool_dev_ = nullptr;
    uint8_t* slot_staging_host_ = nullptr;
    float* gu_nested_absmax_dev_ = nullptr;
    float* dn_nested_absmax_dev_ = nullptr;
    float* gu_nested_quant_map_dev_ = nullptr;
    float* dn_nested_quant_map_dev_ = nullptr;
    size_t gu_nested_per_expert_ = 0;
    size_t dn_nested_per_expert_ = 0;
    float gu_nested_offset_ = 0.0f;
    float dn_nested_offset_ = 0.0f;
    std::array<SlotEntry, kMtpExpertSlots> slot_pool_{};
    std::array<ops::MoeExpertSlotHost, kMtpExperts> host_slots_{};
    std::array<sycl::event, kMtpExpertsPerStep> staging_copy_{};
    std::array<bool, kMtpExpertsPerStep> staging_copy_valid_{};
    std::unique_ptr<sycl::queue> copy_q_;
    uint64_t tick_ = 0;

    std::array<ops::MoeExpertSlotDev, kMtpExpertsPerStep> slots_{};
    std::array<int, kMtpExpertsPerStep> slotIds_{};
    sycl::event slot_ready_{};
    bool slot_ready_valid_ = false;
};
