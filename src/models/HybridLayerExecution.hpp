#pragma once

#include "core/Context.hpp"

#include <array>
#include <chrono>
#include <vector>

class Tensor;
struct Bnb4BitLinearScratch;

namespace aila::models {

enum class HybridExecutionPhase { Prefill, Decode };
enum class HybridLayerKind { FullAttention, DeltaNet, LegacyLinearAttention };

// Bound after projection storage has reached its final address. The output of
// a layer is the residual stream plus the normalized input of the next layer
// (or final norm); execution must preserve that cross-layer fusion boundary.
template <typename Layer>
struct HybridLayerPlan {
    Layer* layer = nullptr;
    Tensor* postMlpNormWeight = nullptr;
    int cacheIndex = 0;
    HybridLayerKind kind = HybridLayerKind::FullAttention;
};

template <typename Layer>
struct HybridDecoderPlan {
    Tensor* initialNormWeight = nullptr;
    bool hasRecurrentLayers = false;
    std::vector<HybridLayerPlan<Layer>> layers;
};

enum class ProfileStage : int {
    EmbedNorm = 0,
    LinearProj,
    LinearDelta,
    LinearOProj,
    FullQkvProj,
    FullSplit,
    QkNormRope,
    KvCache,
    Attention,
    AttnGate,
    FullOProj,
    PostAttnNorm,
    FfnProj,
    FfnAct,
    DownProj,
    PostMlpNorm,
    LmHead,
    Count,
};

struct StageProfileTotals {
    std::array<double, static_cast<size_t>(ProfileStage::Count)> stage_ms{};
    int calls = 0;
    int tokens = 0;

    void reset() {
        stage_ms.fill(0.0);
        calls = 0;
        tokens = 0;
    }
};

struct StageProfiler {
    Context& context;
    std::array<double, static_cast<size_t>(ProfileStage::Count)>& stageMs;
    bool enabled;
    bool hostOnly;

    template <typename Fn>
    void operator()(ProfileStage stage, Fn&& fn) const {
        if (!enabled) {
            fn();
            return;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        if (!hostOnly) context.synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        stageMs[static_cast<size_t>(stage)] +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

struct Nf4ProjectionWorkspace {
    Bnb4BitLinearScratch& scratch;
    sycl::ext::oneapi::bfloat16* residualHidden;
};

struct Bf16ProjectionWorkspace {};

// One sequence's token rows, not a request batch. Cache and positional state
// belong to the Session; buffers/projection scratch belong to the execution
// slot. Neither view may escape this forward's serialized device lifetime.
template <typename Buffers, typename Cache, typename ProjectionWorkspace>
struct HybridLayerExecution {
    Context& context;
    Buffers& buffers;
    Cache& cache;
    HybridExecutionPhase phase;
    int startPosition;
    int tokenCount;
    int cachedLength;
    int cacheCapacity;
    int fullRotaryDimension;
    const int* mropePositionT;
    const int* mropePositionH;
    const int* mropePositionW;
    int mropePromptLength;
    int mropeTextPositionDelta;
    bool debugLayerStats;
    int debugLayerDetail;
    bool debugLinearCompare;
    int debugLinearCompareLayer;
    bool useHostGroupedLinear;
    const StageProfiler& profiler;
    ProjectionWorkspace projection;
};

enum class HybridProjectionSlot {
    LinearAll, LinearQkv, LinearGate, LinearOutput,
    FullQkv, FullOutput, FfnGateUp, FfnDown,
};

template <HybridProjectionSlot Slot, typename Layer>
auto& ProjectionFor(Layer& layer) {
    if constexpr (Slot == HybridProjectionSlot::LinearAll) return layer.linear_all_proj;
    else if constexpr (Slot == HybridProjectionSlot::LinearQkv) return layer.linear_qkv_proj;
    else if constexpr (Slot == HybridProjectionSlot::LinearGate) return layer.linear_z_proj;
    else if constexpr (Slot == HybridProjectionSlot::LinearOutput) return layer.linear_o_proj;
    else if constexpr (Slot == HybridProjectionSlot::FullQkv) return layer.qkv_proj;
    else if constexpr (Slot == HybridProjectionSlot::FullOutput) return layer.o_proj;
    else if constexpr (Slot == HybridProjectionSlot::FfnGateUp) return layer.gate_up_proj;
    else {
        static_assert(Slot == HybridProjectionSlot::FfnDown, "Unknown Hybrid projection slot");
        return layer.down_proj;
    }
}

struct HybridLayerExecutor;
struct HybridDecodeBatchExecutor;

} // namespace aila::models
