#pragma once

#include "inference/Session.hpp"

#include <string>

namespace aila::inference {

namespace detail {
struct DeviceState;
struct LoadedModelState;
}

struct MemorySnapshot {
    // Context-tracked USM only; oneDNN allocations are not a total VRAM meter.
    size_t currentBytes = 0;
    size_t peakBytes = 0;
};

class DeviceRuntime final {
public:
    // Handles share the active default-device execution domain. Additional
    // devices and concurrent independent legacy Engines are outside this API.
    static std::shared_ptr<DeviceRuntime> Create();
    std::string DeviceName() const;
    MemorySnapshot Memory() const;

private:
    friend class LoadedModel;
    explicit DeviceRuntime(std::shared_ptr<detail::DeviceState> state);
    std::shared_ptr<detail::DeviceState> state;
};

struct ModelOptions {
    int maxSequenceLength = 4096;
    size_t maxSessions = 4;
    // Zero uses the session-count bound alone. A nonzero limit reserves the
    // BF16 KV size before allocation. Shared sessions currently reject FP8 KV.
    size_t maxSessionCacheBytes = 0;
};

// Initial shared-model implementation supports Qwen3 NF4 text only. Other
// models continue through IModelBackend and the existing Engine facade.
// Weights and fused projections are loaded once. Their mutable oneDNN
// bindings/activations stay in one execution slot, guarded by DeviceRuntime.
class LoadedModel final {
public:
    static std::shared_ptr<LoadedModel> LoadQwen3Nf4(
        std::shared_ptr<DeviceRuntime> runtime,
        const std::string& modelDirectory,
        ModelOptions options = {});

    std::shared_ptr<SessionState> CreateSession();
    int VocabSize() const noexcept;
    int MaxSequenceLength() const noexcept;
    size_t ActiveSessions() const;
    size_t SessionReservationBytes() const noexcept;

private:
    explicit LoadedModel(std::shared_ptr<detail::LoadedModelState> state);
    std::shared_ptr<detail::LoadedModelState> state;
};

} // namespace aila::inference
