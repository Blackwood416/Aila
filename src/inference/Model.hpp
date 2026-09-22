#pragma once

#include "inference/Session.hpp"

#include <optional>
#include <string>

namespace aila::inference {

namespace detail {
struct DeviceState;
struct LoadedModelState;
}

struct MemorySnapshot {
    // Explicit Context-owned USM (including sampling and shared scratch).
    // Driver/library metadata and pageable host allocations are separate.
    size_t currentBytes = 0;
    size_t peakBytes = 0;
    size_t pinnedHostBytes = 0;
    size_t peakPinnedHostBytes = 0;
    size_t deviceAllocations = 0;
    size_t pinnedHostAllocations = 0;
    size_t deviceLimitBytes = 0;
    size_t pinnedHostLimitBytes = 0;
};

class DeviceRuntime final {
public:
    // Handles share the active default-device execution domain. Additional
    // devices and concurrent independent legacy Engines are outside this API.
    static std::shared_ptr<DeviceRuntime> Create();
    std::string DeviceName() const;
    MemorySnapshot Memory() const;
    // CPU ledger only: no queue access, submit or drain. Empty means the
    // execution lock is busy; a diagnostic caller must not wait for a step.
    std::optional<MemorySnapshot> TryMemory() const;
    // Set only with no live models. Zero disables the corresponding explicit
    // USM ceiling; this does not describe total WDDM usage or process RSS.
    void SetAllocationLimits(size_t deviceBytes, size_t pinnedHostBytes = 0);
    // First device failure, including one observed by a no-throw destructor
    // drain. Lifecycle/reaper callers forward it to Scheduler::ReportRuntimeFault.
    // Takes the execution mutex; do not call it on a transport reader thread.
    std::optional<std::string> FaultMessage() const;

private:
    friend class LoadedModel;
    explicit DeviceRuntime(std::shared_ptr<detail::DeviceState> state);
    MemorySnapshot MemoryLocked() const;
    std::shared_ptr<detail::DeviceState> state;
};

struct ModelOptions {
    int maxSequenceLength = 4096;
    size_t maxSessions = 4;
    // Zero uses the session-count bound alone. A nonzero limit reserves the
    // device cache/state size before allocation. Shared sessions reject FP8 KV.
    size_t maxSessionCacheBytes = 0;
    // Hybrid host mirrors plus bounded recurrent snapshot payloads. Zero uses
    // the session-count bound alone; STL/allocator overhead is not included.
    size_t maxSessionHostBytes = 0;
    // Qwen3.5 only: keep the latest N recurrent checkpoints; zero disables
    // snapshots. Truncate may restore an earlier checkpoint or reset to zero.
    size_t maxRecurrentSnapshots = 4;
};

// Shared-model text execution supports Qwen3 NF4 and Qwen3.5 Hybrid NF4/BF16. Other
// model/storage capabilities continue through the existing Engine facade.
// Weights and fused projections are loaded once. Their mutable oneDNN
// bindings/activations stay in one execution slot, guarded by DeviceRuntime.
class LoadedModel final {
public:
    static std::shared_ptr<LoadedModel> LoadQwen3Nf4(
        std::shared_ptr<DeviceRuntime> runtime,
        const std::string& modelDirectory,
        ModelOptions options = {});
    static std::shared_ptr<LoadedModel> LoadQwen35Nf4(
        std::shared_ptr<DeviceRuntime> runtime,
        const std::string& modelDirectory,
        ModelOptions options = {});
    static std::shared_ptr<LoadedModel> LoadQwen35Bf16(
        std::shared_ptr<DeviceRuntime> runtime,
        const std::string& modelDirectory,
        ModelOptions options = {});

    std::shared_ptr<SessionState> CreateSession();
    int VocabSize() const noexcept;
    int MaxSequenceLength() const noexcept;
    size_t ActiveSessions() const;
    size_t SessionReservationBytes() const noexcept;
    size_t SessionHostReservationBytes() const noexcept;

private:
    static std::shared_ptr<LoadedModel> Load(
        std::shared_ptr<DeviceRuntime> runtime,
        const std::string& modelDirectory,
        ModelOptions options, ModelFamily family, bool denseWeights = false);
    explicit LoadedModel(std::shared_ptr<detail::LoadedModelState> state);
    std::shared_ptr<detail::LoadedModelState> state;
};

} // namespace aila::inference
