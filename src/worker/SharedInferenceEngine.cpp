#include "worker/InferenceWorker.hpp"

#include "inference/Model.hpp"

#include <utility>

namespace aila::worker {
namespace {

class SharedModel final : public InferenceModel {
public:
    SharedModel(std::shared_ptr<inference::LoadedModel> model,
                std::optional<ipc::v2::ModelMemoryInfo> memory)
        : model(std::move(model)), memory(std::move(memory)) {}
    std::shared_ptr<inference::SessionState> CreateSession() override {
        return model->CreateSession();
    }
    int VocabSize() const noexcept override { return model->VocabSize(); }
    int MaxSequenceLength() const noexcept override { return model->MaxSequenceLength(); }
    size_t SessionReservationBytes() const noexcept override {
        return model->SessionReservationBytes();
    }
    std::optional<ipc::v2::ModelMemoryInfo> MemoryInfo() const override { return memory; }

private:
    std::shared_ptr<inference::LoadedModel> model;
    std::optional<ipc::v2::ModelMemoryInfo> memory;
};

class SharedFactory final : public InferenceModelFactory {
public:
    std::shared_ptr<InferenceModel> LoadQwen3(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, inference::LoadedModel::LoadQwen3Nf4);
    }
    std::shared_ptr<InferenceModel> LoadQwen35(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, inference::LoadedModel::LoadQwen35Nf4);
    }
    std::shared_ptr<InferenceModel> LoadQwen35Bf16(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, inference::LoadedModel::LoadQwen35Bf16);
    }
    std::shared_ptr<InferenceModel> LoadQwen4Exp(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, inference::LoadedModel::LoadQwen4ExpNf4);
    }

    std::string RuntimeFaultMessage() const override {
        return runtime ? runtime->FaultMessage().value_or(std::string{}) : std::string{};
    }

    ipc::v2::RuntimeMemoryInfo QueryRuntimeMemory() const override {
        if (!runtime) return {};
        const auto value = runtime->TryMemory();
        if (!value) return {ipc::v2::RuntimeMemoryState::Busy, std::nullopt};
        return {ipc::v2::RuntimeMemoryState::Ready, ipc::v2::RuntimeMemoryUsage{
            value->currentBytes, value->peakBytes, value->pinnedHostBytes, value->peakPinnedHostBytes,
            value->deviceAllocations, value->pinnedHostAllocations, value->deviceLimitBytes,
            value->pinnedHostLimitBytes}};
    }

private:
    using ModelLoader = decltype(&inference::LoadedModel::LoadQwen3Nf4);
    std::shared_ptr<InferenceModel> Load(
        const std::string& directory, InferenceModelOptions options, ModelLoader loader) {
        if (!runtime) runtime = inference::DeviceRuntime::Create();
        const auto limits = options.allocationLimits.value_or(ipc::v2::AllocationLimits{});
        runtime->SetAllocationLimits(static_cast<size_t>(limits.maxDeviceBytes),
                                     static_cast<size_t>(limits.maxPinnedHostBytes));
        inference::ModelOptions modelOptions;
        modelOptions.maxSequenceLength = options.maxSequenceLength;
        modelOptions.maxSessions = options.maxSessions;
        modelOptions.maxSessionCacheBytes = options.maxSessionCacheBytes;
        modelOptions.maxSessionHostBytes = static_cast<size_t>(limits.maxSessionHostBytes);
        // IPC has no rollback API. Keep the required Hybrid host mirrors but
        // allocate no recurrent snapshots. Host mirrors have a separate byte
        // limit when allocation_limits_v1 is requested.
        modelOptions.maxRecurrentSnapshots = 0;
        auto loaded = loader(runtime, directory, modelOptions);
        std::optional<ipc::v2::ModelMemoryInfo> memory;
        if (options.allocationLimits) {
            memory = ipc::v2::ModelMemoryInfo{limits, runtime->Memory().currentBytes,
                                             loaded->SessionHostReservationBytes()};
        }
        return std::make_shared<SharedModel>(std::move(loaded), std::move(memory));
    }

    std::shared_ptr<inference::DeviceRuntime> runtime;
};

} // namespace

std::unique_ptr<InferenceModelFactory> MakeSharedInferenceModelFactory() {
    return std::make_unique<SharedFactory>();
}

} // namespace aila::worker
