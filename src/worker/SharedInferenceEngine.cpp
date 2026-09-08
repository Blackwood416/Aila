#include "worker/InferenceWorker.hpp"

#include "inference/Model.hpp"

#include <utility>

namespace aila::worker {
namespace {

class SharedModel final : public InferenceModel {
public:
    explicit SharedModel(std::shared_ptr<inference::LoadedModel> model)
        : model(std::move(model)) {}
    std::shared_ptr<inference::SessionState> CreateSession() override {
        return model->CreateSession();
    }
    int VocabSize() const noexcept override { return model->VocabSize(); }
    int MaxSequenceLength() const noexcept override { return model->MaxSequenceLength(); }
    size_t SessionReservationBytes() const noexcept override {
        return model->SessionReservationBytes();
    }

private:
    std::shared_ptr<inference::LoadedModel> model;
};

class SharedFactory final : public InferenceModelFactory {
public:
    std::shared_ptr<InferenceModel> LoadQwen3(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, false);
    }
    std::shared_ptr<InferenceModel> LoadQwen35(
        const std::string& directory, InferenceModelOptions options) override {
        return Load(directory, options, true);
    }

    std::string RuntimeFaultMessage() const override {
        return runtime ? runtime->FaultMessage().value_or(std::string{}) : std::string{};
    }

private:
    std::shared_ptr<InferenceModel> Load(
        const std::string& directory, InferenceModelOptions options, bool hybrid) {
        if (!runtime) runtime = inference::DeviceRuntime::Create();
        inference::ModelOptions modelOptions;
        modelOptions.maxSequenceLength = options.maxSequenceLength;
        modelOptions.maxSessions = options.maxSessions;
        modelOptions.maxSessionCacheBytes = options.maxSessionCacheBytes;
        // IPC has no rollback API. Keep the required Hybrid host mirrors but
        // allocate no recurrent snapshots; host payload remains session-count
        // bounded (at most four), not covered by the device-cache byte limit.
        if (hybrid) modelOptions.maxRecurrentSnapshots = 0;
        auto loaded = hybrid
            ? inference::LoadedModel::LoadQwen35Nf4(runtime, directory, modelOptions)
            : inference::LoadedModel::LoadQwen3Nf4(runtime, directory, modelOptions);
        return std::make_shared<SharedModel>(std::move(loaded));
    }

    std::shared_ptr<inference::DeviceRuntime> runtime;
};

} // namespace

std::unique_ptr<InferenceModelFactory> MakeSharedInferenceModelFactory() {
    return std::make_unique<SharedFactory>();
}

} // namespace aila::worker
