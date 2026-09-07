#include "inference/Model.hpp"

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "models/Qwen3Bnb4Backend.hpp"
#include "ops/Ops.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace aila::inference {

namespace detail {

struct DeviceState {
    Context context;
    mutable std::mutex mutex;
    bool faulted = false;
    std::string fault;

    void CheckUsable() const {
        if (faulted) throw Error(ErrorCode::RuntimeFault, FaultMessage());
    }

    const char* FaultMessage() const noexcept {
        return fault.empty() ? "Device execution domain is unusable" : fault.c_str();
    }

    void RecordFaultNoThrow(const char* message) noexcept {
        // Keep the domain invalid even if recording the message runs out of
        // host memory while handling a device failure.
        faulted = true;
        try { if (fault.empty()) fault = message; } catch (...) {}
    }

    void DrainNoThrow() noexcept {
        try {
            context.synchronize();
        } catch (const std::exception& e) {
            // wait_and_throw drains submitted work. Destructors must not throw,
            // but other sessions in this execution domain must subsequently fail.
            RecordFaultNoThrow(e.what());
        } catch (...) {
            RecordFaultNoThrow("Unknown device failure");
        }
    }

    ~DeviceState() { DrainNoThrow(); }
};

// Shared by all sessions of the model. During migration the backend also
// retains its activations and mutable oneDNN bindings in this execution slot.
struct ExecutionWorkspace {
    Tensor tokenIds;
    Tensor sampledId;
    size_t tokenCapacity = 0;

    void Ensure(Context& context, size_t count) {
        if (count > tokenCapacity) {
            context.synchronize();
            tokenIds = Tensor::allocate(context, {static_cast<int64_t>(count)},
                                        dnnl::memory::data_type::s32);
            tokenCapacity = count;
        }
        if (!sampledId.valid()) {
            sampledId = Tensor::allocate(context, {1}, dnnl::memory::data_type::s32);
        }
    }
};

struct LoadedModelState {
    std::shared_ptr<DeviceState> device;
    ModelOptions options;
    ModelSpec spec;
    size_t sessionReservationBytes = 0;
    size_t activeSessions = 0; // Protected by device->mutex.
    std::unique_ptr<ModelWeights> weights;
    std::unique_ptr<Qwen3Bnb4Backend> backend;
    std::unique_ptr<ExecutionWorkspace> workspace;

    ~LoadedModelState() {
        if (!device) return;
        std::lock_guard<std::mutex> lock(device->mutex);
        device->DrainNoThrow();
        workspace.reset();
        backend.reset();
        weights.reset();
    }
};

class Qwen3SessionBackend final : public SessionBackend {
public:
    explicit Qwen3SessionBackend(std::shared_ptr<LoadedModelState> loadedModel)
        : model(std::move(loadedModel)) {}

    ~Qwen3SessionBackend() override {
        if (!cache) return;
        // Keep the model/runtime alive while every Tensor is freed, including
        // when the caller has already dropped its model and runtime handles.
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->DrainNoThrow();
        cache.reset();
        --model->activeSessions;
    }

    void Initialize() {
        aila::ScopedLogDeferral deferLogs;
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        const auto& options = model->options;
        if (model->activeSessions >= options.maxSessions ||
            (options.maxSessionCacheBytes != 0 &&
             model->activeSessions >= options.maxSessionCacheBytes / model->sessionReservationBytes)) {
            throw Error(ErrorCode::ResourceLimit, "Shared model session cache limit reached");
        }
        try {
            cache = model->backend->CreateSession(model->device->context);
            ++model->activeSessions;
        } catch (const std::invalid_argument& error) {
            model->device->DrainNoThrow();
            throw Error(ErrorCode::InvalidArgument, error.what());
        } catch (...) {
            model->device->DrainNoThrow();
            throw;
        }
    }

    int ContextLength() const override {
        std::lock_guard<std::mutex> lock(model->device->mutex);
        return cache->CurrentLength();
    }
    int Capacity() const override { return model->options.maxSequenceLength; }
    int VocabSize() const override { return model->spec.qwen3.vocab_size; }
    size_t CacheBytes() const override { return cache->CacheBytes(); }

    void Reset() override {
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        model->backend->Reset(*cache);
    }

    int Truncate(int length) override {
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        return model->backend->Truncate(*cache, length);
    }

    ExecutionResult Forward(const std::vector<int>& tokenIds,
                            const SamplingInput* sampling,
                            bool captureLogits) override {
        aila::ScopedLogDeferral deferLogs;
        auto& device = *model->device;
        std::lock_guard<std::mutex> lock(device.mutex);
        device.CheckUsable();
        auto& context = device.context;
        auto& slot = *model->workspace;
        try {
            slot.Ensure(context, tokenIds.size());
            context.memcpy_h2d_async(slot.tokenIds.data(), tokenIds.data(),
                                     tokenIds.size() * sizeof(int));
            Tensor& logits = model->backend->Forward(
                context, *cache, slot.tokenIds.data_as<int>(),
                static_cast<int>(tokenIds.size()));

            ExecutionResult result;
            if (sampling) {
                if (!sampling->config.do_sample) {
                    ops::argmax(context, logits, VocabSize(), slot.sampledId.data_as<int>());
                    context.memcpy_d2h(&result.tokenId, slot.sampledId.data(), sizeof(int));
                } else if (ops::can_use_device_sampling(VocabSize(), sampling->config)) {
                    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                    ops::sample_with_config_device(context, logits, VocabSize(), sampling->config,
                                                   uniform(sampling->rng), slot.sampledId.data_as<int>());
                    context.memcpy_d2h(&result.tokenId, slot.sampledId.data(), sizeof(int));
                } else {
                    result.tokenId = ops::sample_with_config(context, logits, VocabSize(),
                        sampling->config, sampling->generatedIds, sampling->rng);
                }
            }
            if (captureLogits) {
                using bf16 = sycl::ext::oneapi::bfloat16;
                std::vector<bf16> hostLogits(static_cast<size_t>(VocabSize()));
                context.memcpy_d2h(hostLogits.data(), logits.data(), hostLogits.size() * sizeof(bf16));
                result.logits.resize(hostLogits.size());
                std::transform(hostLogits.begin(), hostLogits.end(), result.logits.begin(),
                               [](bf16 value) { return static_cast<float>(value); });
            }
            // No device work may still refer to borrowed input or mutable
            // bindings when the caller destroys/switches the session.
            context.synchronize();
            return result;
        } catch (const std::exception& e) {
            device.faulted = true;
            // Drain borrowed host input before any error-message allocation.
            device.DrainNoThrow();
            device.RecordFaultNoThrow(e.what());
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        } catch (...) {
            device.faulted = true;
            device.DrainNoThrow();
            device.RecordFaultNoThrow("Unknown device execution failure");
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        }
    }

private:
    std::shared_ptr<LoadedModelState> model;
    std::unique_ptr<Qwen3Bnb4Backend::Session> cache;
};

} // namespace detail

namespace {

size_t checked_product(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) {
        throw Error(ErrorCode::InvalidArgument, "Session cache size overflows size_t");
    }
    return a * b;
}

size_t session_reservation_bytes(const Qwen3Config& config, int capacity) {
    size_t result = 2 * sizeof(sycl::ext::oneapi::bfloat16); // K and V.
    for (int dimension : {config.num_hidden_layers, config.num_key_value_heads,
                          config.head_dim, capacity}) {
        if (dimension <= 0) throw Error(ErrorCode::InvalidArgument, "Invalid model cache dimensions");
        result = checked_product(result, static_cast<size_t>(dimension));
    }
    return result;
}

} // namespace

DeviceRuntime::DeviceRuntime(std::shared_ptr<detail::DeviceState> value)
    : state(std::move(value)) {}

std::shared_ptr<DeviceRuntime> DeviceRuntime::Create() {
    // The first release has one default-device execution domain. Reusing it
    // also serializes legacy kernel caches that are still process-global.
    static std::mutex creationMutex;
    static std::weak_ptr<detail::DeviceState> activeDevice;
    std::lock_guard<std::mutex> lock(creationMutex);
    auto device = activeDevice.lock();
    if (!device) {
        device = std::make_shared<detail::DeviceState>();
        activeDevice = device;
    }
    return std::shared_ptr<DeviceRuntime>(new DeviceRuntime(std::move(device)));
}

std::string DeviceRuntime::DeviceName() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->context.queue().get_device().get_info<sycl::info::device::name>();
}

MemorySnapshot DeviceRuntime::Memory() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return {state->context.current_allocated_bytes(), state->context.peak_allocated_bytes()};
}

LoadedModel::LoadedModel(std::shared_ptr<detail::LoadedModelState> value)
    : state(std::move(value)) {}

std::shared_ptr<LoadedModel> LoadedModel::LoadQwen3Nf4(
    std::shared_ptr<DeviceRuntime> runtime,
    const std::string& modelDirectory,
    ModelOptions options) {
    aila::ScopedLogDeferral deferLogs;
    if (!runtime || modelDirectory.empty() || options.maxSequenceLength <= 0 || options.maxSessions == 0) {
        throw Error(ErrorCode::InvalidArgument, "Runtime, model directory and positive model limits are required");
    }
    if (aila::env::read_flag("AILA_KV_QUANT", false)) {
        throw Error(ErrorCode::InvalidArgument,
                    "Qwen3 shared sessions require BF16 KV; set AILA_KV_QUANT=0");
    }
    auto loaded = std::make_shared<detail::LoadedModelState>();
    loaded->device = runtime->state;
    std::string error;
    if (!aila::modelspec::load_from_dir(modelDirectory, loaded->spec, &error)) {
        throw Error(ErrorCode::InvalidArgument, error);
    }
    const auto& spec = loaded->spec;
    if (spec.family != ModelFamily::Qwen3Dense || !spec.is_bitsandbytes_4bit() ||
        spec.quantization.bnb_4bit_quant_type != "nf4") {
        throw Error(ErrorCode::UnsupportedModel, "Shared sessions currently require a Qwen3 NF4 text model");
    }
    if (spec.qwen3.vocab_size <= 0 || spec.qwen3.hidden_size <= 0 ||
        spec.qwen3.num_attention_heads <= 0 || spec.qwen3.num_key_value_heads <= 0 ||
        spec.qwen3.num_attention_heads % spec.qwen3.num_key_value_heads != 0 ||
        spec.qwen3.max_position_embeddings <= 0) {
        throw Error(ErrorCode::InvalidArgument, "Invalid Qwen3 model dimensions");
    }
    options.maxSequenceLength = std::min(options.maxSequenceLength, spec.qwen3.max_position_embeddings);
    loaded->options = options;
    loaded->sessionReservationBytes = session_reservation_bytes(spec.qwen3, options.maxSequenceLength);
    // Declare the lock after loaded: a failed load frees resources only after
    // releasing this lock (LoadedModelState's destructor acquires it again).
    std::unique_lock<std::mutex> lock(loaded->device->mutex);
    loaded->device->CheckUsable();
    auto& context = loaded->device->context;
    try {
        loaded->weights = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(modelDirectory, context));
        loaded->backend = std::make_unique<Qwen3Bnb4Backend>();
        if (!loaded->backend->load(context, *loaded->weights, spec, options.maxSequenceLength, &error)) {
            throw Error(ErrorCode::UnsupportedModel, error);
        }
        loaded->workspace = std::make_unique<detail::ExecutionWorkspace>();
        context.synchronize();
    } catch (...) {
        loaded->device->DrainNoThrow();
        throw;
    }
    lock.unlock();
    return std::shared_ptr<LoadedModel>(new LoadedModel(std::move(loaded)));
}

std::shared_ptr<SessionState> LoadedModel::CreateSession() {
    aila::ScopedLogDeferral deferLogs;
    auto backend = std::make_unique<detail::Qwen3SessionBackend>(state);
    backend->Initialize();
    return std::make_shared<SessionState>(std::move(backend));
}

int LoadedModel::VocabSize() const noexcept { return state->spec.qwen3.vocab_size; }
int LoadedModel::MaxSequenceLength() const noexcept { return state->options.maxSequenceLength; }
size_t LoadedModel::SessionReservationBytes() const noexcept { return state->sessionReservationBytes; }

size_t LoadedModel::ActiveSessions() const {
    std::lock_guard<std::mutex> lock(state->device->mutex);
    return state->activeSessions;
}

} // namespace aila::inference
