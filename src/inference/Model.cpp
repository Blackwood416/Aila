#include "inference/Model.hpp"
#include "inference/RuntimeFault.hpp"

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "memory/SessionMemory.hpp"
#include "models/Qwen3Bnb4Backend.hpp"
#include "models/Qwen35HybridBnb4Backend.hpp"
#include "models/Qwen35HybridTextBackend.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "models/Qwen4ExpMtpRuntime.hpp"
#include "models/Qwen4ExpStateDigest.hpp"
#include "ops/GreedyOps.hpp"
#include "ops/Ops.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace aila::inference {

using aila::diagnostics::ExecutionFields;
using aila::diagnostics::ExecutionPhase;
using aila::diagnostics::ExecutionScope;
using aila::diagnostics::ExecutionStage;
using aila::diagnostics::ExecutionTrace;

namespace detail {

// Worker-path device queue. Profiling is opt-in and independent of the decode
// stage-event gate: Qwen4ExpBnb4Backend can only read command_start/command_end
// when the queue carries sycl::property::queue::enable_profiling, so the
// enabler lives here and the default stays the plain in-order queue.
Context MakeDeviceContext(ExecutionTrace* trace) {
    static const bool s_queue_profiling =
        aila::env::read_flag("AILA_Q4E_DEVICE_QUEUE_PROFILING", false);
    if (!s_queue_profiling) return Context(trace);
    try {
        return Context(trace, true);
    } catch (const std::exception& error) {
        // Some backends reject the property at queue creation; keep the plain
        // in-order queue instead of failing the model load (warn once).
        static bool warned = false;
        if (!warned) {
            warned = true;
            AILA_LOG_WARN("[Q4E QueueProfiling] enable_profiling rejected (%s); "
                          "falling back to the default in-order queue", error.what());
        }
        return Context(trace);
    }
}

struct DeviceState {
    // First member dies last: trace_close follows native queue/stream teardown.
    std::unique_ptr<ExecutionTrace> trace = ExecutionTrace::FromEnvironment();
    // Device spans need the opt-in profiling queue; see MakeDeviceContext.
    Context context{MakeDeviceContext(trace.get())};
    mutable std::mutex mutex;
    RuntimeFaultRecord fault;
    size_t loadedModels = 0;

    void CheckUsable() {
        RecordTraceFailureNoThrow();
        fault.CheckUsable();
    }

    void RecordTraceFailureNoThrow() noexcept {
        if (trace && trace->Failed()) fault.Record("Execution trace write/flush failed; device work stopped");
    }

    const char* FaultMessage() const noexcept { return fault.Message(); }

    void RecordFaultNoThrow(const char* message) noexcept {
        const bool first = !fault.Faulted();
        fault.Record(message);
        if (first) {
            try { AILA_LOG_ERROR("[Aila RuntimeFault] %s", fault.Message()); } catch (...) {}
        }
    }

    void DrainNoThrow() noexcept {
        fault.DrainNoThrow([&] { context.synchronize(); });
    }

    ~DeviceState() {
        ExecutionScope stage(trace.get(), ExecutionStage::DeviceShutdown);
        DrainNoThrow();
        stage.Complete();
    }
};

// Shared by all sessions of the model. During migration the backend also
// retains its activations and mutable oneDNN bindings in this execution slot.
struct ExecutionWorkspace {
    Tensor tokenIds;
    Tensor sampledId;
    Tensor greedyPartials;
    Tensor greedyResult;
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
    int vocabSize = 0;
    size_t sessionReservationBytes = 0;
    size_t sessionHostReservationBytes = 0;
    size_t activeSessions = 0; // Protected by device->mutex.
    bool registeredModel = false;
    bool denseWeights = false;
    uint64_t traceModelId = 0;
    std::unique_ptr<ModelWeights> weights;
    std::unique_ptr<IModelBackend> backend;
    std::unique_ptr<ExecutionWorkspace> workspace;
    std::unique_ptr<Qwen4ExpMtpRuntime> mtp;

    ~LoadedModelState() {
        if (!device) return;
        std::lock_guard<std::mutex> lock(device->mutex);
        ExecutionScope stage(device->trace.get(), ExecutionStage::ModelDestroy, {traceModelId});
        device->DrainNoThrow();
        workspace.reset();
        mtp.reset();
        backend.reset();
        weights.reset();
        if (registeredModel) --device->loadedModels;
        if (!device->loadedModels) device->context.release_execution_resources();
        stage.Complete();
        device->RecordTraceFailureNoThrow();
    }
};

// Select the model implementation once when creating the session. Sampling,
// execution locking and lifetime rules are shared by both text backends.
template <typename Backend>
class ModelSessionBackend final : public SessionBackend {
public:
    explicit ModelSessionBackend(std::shared_ptr<LoadedModelState> loadedModel)
        : model(std::move(loadedModel)),
          traceSessionId(model->device->trace ? model->device->trace->NextObjectId() : 0) {}

    ~ModelSessionBackend() override {
        if (!cache) return;
        // Keep the model/runtime alive while every Tensor is freed, including
        // when the caller has already dropped its model and runtime handles.
        std::lock_guard<std::mutex> lock(model->device->mutex);
        ExecutionScope stage(model->device->trace.get(), ExecutionStage::SessionDestroy, TraceFields());
        model->device->DrainNoThrow();
        mtpState.reset();
        cache.reset();
        --model->activeSessions;
        stage.Complete();
        model->device->RecordTraceFailureNoThrow();
    }

    void Initialize() {
        aila::ScopedLogDeferral deferLogs;
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        const auto& options = model->options;
        if (model->activeSessions >= options.maxSessions ||
            (options.maxSessionCacheBytes != 0 &&
             model->activeSessions >= options.maxSessionCacheBytes / model->sessionReservationBytes) ||
            (options.maxSessionHostBytes != 0 && model->sessionHostReservationBytes != 0 &&
             model->activeSessions >= options.maxSessionHostBytes / model->sessionHostReservationBytes)) {
            throw Error(ErrorCode::ResourceLimit, "Shared model session memory limit reached");
        }
        std::unique_ptr<typename Backend::Session> initializedCache;
        ExecutionScope stage(model->device->trace.get(), ExecutionStage::SessionCreate, TraceFields());
        RunDeviceLifecycle<sycl::exception, dnnl::error>(model->device->fault, [&] {
            // Check the whole cache before its first allocation, so a rejected
            // session neither partially allocates KV nor raises the peak.
            model->device->context.check_device_allocation(model->sessionReservationBytes);
            if constexpr (!std::is_same_v<Backend, Qwen3Bnb4Backend>) {
                initializedCache = ModelBackend().CreateSession(
                    model->device->context, options.maxRecurrentSnapshots);
            } else {
                initializedCache = ModelBackend().CreateSession(model->device->context);
            }
            if (model->mtp) mtpState = model->mtp->CreateState();
        }, [&] { model->device->context.synchronize(); });
        // Count only a fully initialized, drained session. A failed final drain
        // must not leave a cache whose destructor decrements an unheld credit.
        cache = std::move(initializedCache);
        ++model->activeSessions;
        stage.Complete();
        model->device->CheckUsable();
    }

    int ContextLength() const override {
        // SessionState serializes reads and writes with operationMutex (or
        // this backend is still unpublished during construction). This is
        // private host metadata: querying an idle session must not wait on a
        // different session's GPU step while Scheduler::Submit admits it.
        return cache->CurrentLength();
    }
    int Capacity() const override { return model->options.maxSequenceLength; }
    int VocabSize() const override { return model->vocabSize; }
    size_t CacheBytes() const override {
        return cache->CacheBytes() + (model->mtp ? Qwen4ExpMtpRuntime::SessionBytes(model->mtp->Capacity()) : 0);
    }
    bool SupportsDecodeWindow() const noexcept override { return model->mtp != nullptr; }
    std::optional<DecodeWindowResult> ForwardWindow(int pending, const DecodeWindowSelector& selector,
                                                    int maxWidth = 0) override {
        if constexpr (!std::is_same_v<Backend, Qwen4ExpBnb4Backend>) return std::nullopt;
        else {
            auto& device = *model->device;
            std::lock_guard<std::mutex> lock(device.mutex);
            device.CheckUsable();
            if (!model->mtp || !mtpState->valid || mtpState->length != cache->CurrentLength() ||
                cache->CurrentLength() < 1 || cache->CurrentLength() + 2 > model->mtp->Capacity()) return std::nullopt;
            try {
                const int effWidth = maxWidth > 0
                    ? std::min(maxWidth, model->mtp->ConfiguredWidth())
                    : model->mtp->ConfiguredWidth();
                LogSessionBoundary(device.context, "win-pre", effWidth, -1, pending);
                auto window = model->mtp->ForwardWindow(*cache, *mtpState, pending, selector, effWidth);
                LogSessionBoundary(device.context, "win-post", effWidth, -1, window.tokenId);
                return window;
            } catch (const std::exception& error) {
                device.RecordFaultNoThrow(error.what()); device.DrainNoThrow();
                throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
            } catch (...) {
                device.RecordFaultNoThrow("Unknown MTP execution failure"); device.DrainNoThrow();
                throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
            }
        }
    }
    const void* DecodeBatchKey() const noexcept override {
        if (!ModelBackend().SupportsDecodeBatch()) return nullptr;
        return model.get();
    }

    void Reset() override {
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        ExecutionScope stage(model->device->trace.get(), ExecutionStage::SessionReset, TraceFields());
        MutateCache([&] { ModelBackend().Reset(*cache); });
        if (mtpState) { mtpState->length = 0; mtpState->valid = true; }
        stage.Complete();
        model->device->CheckUsable();
    }

    int Truncate(int length) override {
        std::lock_guard<std::mutex> lock(model->device->mutex);
        model->device->CheckUsable();
        auto fields = TraceFields();
        fields.targetPosition = length;
        ExecutionScope stage(model->device->trace.get(), ExecutionStage::SessionTruncate, fields);
        int restored = 0;
        MutateCache([&] { restored = ModelBackend().Truncate(*cache, length); });
        if (mtpState && restored != mtpState->length) {
            mtpState->length = restored; mtpState->valid = restored == 0;
        }
        stage.Complete();
        model->device->CheckUsable();
        return restored;
    }

    ExecutionResult Forward(const std::vector<int>& tokenIds,
                            const SamplingInput* sampling,
                            bool captureLogits) override {
        return ForwardImpl(tokenIds, sampling, captureLogits, -1);
    }

    // Prompt-phase entry point. MTP AfterForward needs complete target hidden
    // rows, so while a chunk lands inside the draft window the phase close-out
    // stays off and the existing path is used unchanged; a chunk that would
    // leave the window keeps the close-out and lets the draft invalidate itself.
    ExecutionResult ForwardPrefill(const std::vector<int>& tokenIds,
                                   const SamplingInput* sampling,
                                   bool captureLogits,
                                   int prefillRemaining,
                                   const int* nextTokens, int nextCount) override {
        const int remaining = (model->mtp && mtpState && mtpState->valid &&
                               cache->CurrentLength() + static_cast<int>(tokenIds.size()) <= model->mtp->Capacity())
            ? -1 : prefillRemaining;
        return ForwardImpl(tokenIds, sampling, captureLogits, remaining, nextTokens, nextCount);
    }

private:
    // B4-6 opt-in boundary dump: fingerprints the live session state (and the
    // MTP draft when one is loaded) so a T1 run and an MTP run can be compared
    // through the worker log. Purely diagnostic: inert unless
    // AILA_Q4E_MTP_BOUNDARY_DUMP is set, and it never touches device state.
    void LogSessionBoundary(Context& context, const char* phase, int count,
                            int prefillRemaining, int sampledToken) {
        static const bool enabled = aila::env::read_flag("AILA_Q4E_MTP_BOUNDARY_DUMP", false);
        if (!enabled) return;
        if constexpr (!std::is_same_v<Backend, Qwen4ExpBnb4Backend>) return;
        else {
            const int position = cache->CurrentLength();
            const auto digest = aila::q4ediag::DigestSession(context, *cache);
            std::string draft = "none";
            if (model->mtp && mtpState)
                draft = aila::q4ediag::DescribeMtpState(context, *mtpState, position);
            uint64_t slots = 0, routeQueries = 0, hits = 0, misses = 0;
            const auto& adaptive = ModelBackend().adaptive_cache();
            if (adaptive.IsAllocated()) {
                slots = aila::q4ediag::DigestBytes(1469598103934665603ull, adaptive.HostSlotTable(),
                    size_t(adaptive.NumLayers()) * size_t(adaptive.NumExpertsPerLayer()) * sizeof(int32_t));
                const auto& stats = adaptive.Stats();
                routeQueries = stats.route_queries;
                hits = stats.cache_hits;
                misses = stats.cache_misses;
            }
            const std::string sessionText = aila::q4ediag::DescribeSession(digest);
            AILA_LOG_INFO("[Q4E BND] phase=%s pos=%d n=%d rem=%d mtp=%d mtpLen=%d mtpValid=%d next=%d "
                          "slots=%s rq=%llu hits=%llu miss=%llu draft=%s %s",
                phase, position, count, prefillRemaining, model->mtp ? 1 : 0,
                mtpState ? mtpState->length : -1, (mtpState && mtpState->valid) ? 1 : 0, sampledToken,
                aila::q4ediag::Hex(slots).c_str(),
                static_cast<unsigned long long>(routeQueries),
                static_cast<unsigned long long>(hits),
                static_cast<unsigned long long>(misses),
                draft.c_str(), sessionText.c_str());
            // Per-layer hashes need their own lines: a single AILA_LOG_INFO
            // message is truncated near 1 KiB.
            AILA_LOG_INFO("[Q4E BNDL] phase=%s pos=%d layer=kv %s", phase, position,
                          aila::q4ediag::DescribeLayerList(digest.kvLayers).c_str());
            AILA_LOG_INFO("[Q4E BNDL] phase=%s pos=%d layer=gdn %s", phase, position,
                          aila::q4ediag::DescribeLayerList(digest.gdnLayers).c_str());
        }
    }

    ExecutionResult ForwardImpl(const std::vector<int>& tokenIds,
                                const SamplingInput* sampling,
                                bool captureLogits,
                                int prefillRemaining,
                                const int* nextTokens = nullptr, int nextCount = 0) {
        aila::ScopedLogDeferral deferLogs;
        auto& device = *model->device;
        std::lock_guard<std::mutex> lock(device.mutex);
        device.CheckUsable();
        auto& context = device.context;
        auto& slot = *model->workspace;
        static const bool workSummary = std::is_same_v<Backend, Qwen4ExpBnb4Backend> &&
            aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
        using WorkClock = std::chrono::steady_clock;
        const auto workStart = workSummary ? WorkClock::now() : WorkClock::time_point{};
        auto fields = TraceFields();
        fields.tokenCount = static_cast<int64_t>(tokenIds.size());
        fields.position = cache->CurrentLength();
        // This labels the backend kernel path; a one-token prompt also uses decode.
        fields.phase = tokenIds.size() == 1 ? ExecutionPhase::Decode : ExecutionPhase::Prefill;
        ExecutionScope stage(device.trace.get(), ExecutionStage::Forward, fields);
        try {
            LogSessionBoundary(context, "pre", static_cast<int>(tokenIds.size()), prefillRemaining, -1);
            slot.Ensure(context, tokenIds.size());
            context.memcpy_h2d_async(slot.tokenIds.data(), tokenIds.data(),
                                     tokenIds.size() * sizeof(int));
            Tensor* forwardedLogits = nullptr;
            const auto backendStart = workSummary ? WorkClock::now() : WorkClock::time_point{};
            {
                ExecutionScope submitted(device.trace.get(), ExecutionStage::BackendForward);
                if constexpr (!std::is_same_v<Backend, Qwen3Bnb4Backend>) {
                    if (sampling && sampling->legacyPrefillLogits) {
                        // The old Hybrid ABI samples row zero from its multi-row
                        // continuation output. Keep that compatibility choice
                        // local to the request; explicit ForwardLogits and new
                        // API defaults continue to return the last valid row.
                        forwardedLogits = &ModelBackend().ForwardLegacyLogits(
                            context, *cache, slot.tokenIds.data_as<int>(),
                            static_cast<int>(tokenIds.size()));
                    }
                }
                if (!forwardedLogits) {
                    if constexpr (std::is_same_v<Backend, Qwen4ExpBnb4Backend>) {
                        ForwardOptions options;
                        // Intermediate prompt chunks and the final pending
                        // token only update state; their LM head is never read.
                        options.need_logits = sampling != nullptr || captureLogits;
                        options.prefill_remaining = prefillRemaining;
                        // The ids are still in this std::vector, so the host
                        // embedding gather needs no device readback.
                        options.host_token_ids = tokenIds.data();
                        if (prefillRemaining >= 0) {
                            options.next_prefill_tokens = nextTokens;
                            options.next_prefill_count = nextCount;
                        }
                        forwardedLogits = &ModelBackend().Forward(context, *cache,
                            slot.tokenIds.data_as<int>(), static_cast<int>(tokenIds.size()), options);
                    } else {
                        forwardedLogits = &ModelBackend().Forward(
                            context, *cache, slot.tokenIds.data_as<int>(),
                            static_cast<int>(tokenIds.size()));
                    }
                }
                submitted.Complete(); // Host return, not proof that the GPU finished.
            }
            Tensor& logits = *forwardedLogits;
            const auto backendEnd = workSummary ? WorkClock::now() : WorkClock::time_point{};

            auto result = ReadOutput(logits, sampling, captureLogits);
            const auto sampleEnd = workSummary ? WorkClock::now() : WorkClock::time_point{};
            if (model->mtp) model->mtp->AfterForward(*mtpState,
                cache->CurrentLength() - static_cast<int>(tokenIds.size()), tokenIds);
            // No device work may still refer to borrowed input or mutable
            // bindings when the caller destroys/switches the session.
            context.synchronize();
            LogSessionBoundary(context, "post", static_cast<int>(tokenIds.size()), prefillRemaining,
                               result.tokenId);
            if (workSummary) {
                const auto end = WorkClock::now();
                const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b-a).count(); };
                AILA_LOG_INFO("[Q4E ForwardWall] pos=%d count=%d remaining=%d sample=%d total_ms=%.6f prepare_ms=%.6f backend_ms=%.6f sample_ms=%.6f drain_ms=%.6f",
                    cache->CurrentLength()-int(tokenIds.size()), int(tokenIds.size()), prefillRemaining, int(sampling!=nullptr),
                    ms(workStart,end), ms(workStart,backendStart), ms(backendStart,backendEnd), ms(backendEnd,sampleEnd), ms(sampleEnd,end));
            }
            stage.Complete();
            device.CheckUsable();
            return result;
        } catch (const aila::memory::AllocationLimitExceeded& error) {
            // A policy rejection is local. Earlier layers may already have
            // updated this session, so drain and reset it before publishing the
            // error; another session's state and the device domain stay usable.
            device.DrainNoThrow();
            device.CheckUsable();
            {
                ExecutionScope reset(device.trace.get(), ExecutionStage::SessionReset, TraceFields());
                MutateCache([&] { ModelBackend().Reset(*cache); });
                reset.Complete();
            }
            throw Error(ErrorCode::ResourceLimit, error.what());
        } catch (const std::exception& e) {
            // The record has fixed storage: preserve the first failure before
            // draining borrowed host input and allocating an Error message.
            device.RecordFaultNoThrow(e.what());
            device.DrainNoThrow();
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        } catch (...) {
            device.RecordFaultNoThrow("Unknown device execution failure");
            device.DrainNoThrow();
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        }
    }

public:
    std::vector<ExecutionResult> ForwardDecodeBatch(
        const std::vector<BackendDecodeInput>& rows) override {
        if (rows.empty() || rows.size() > MaxDecodeBatchSize) {
            throw Error(ErrorCode::InvalidArgument, "Decode batch must contain between one and four rows");
        }
        if (rows.size() == 1 && rows[0].backend == this) {
            return {Forward({rows[0].tokenId}, rows[0].sampling, rows[0].captureLogits)};
        }
        if (!DecodeBatchKey()) {
            throw Error(ErrorCode::UnsupportedModel, "Loaded model configuration does not support decode batching");
        }
        if constexpr (std::is_same_v<Backend, Qwen4ExpBnb4Backend>) {
            throw Error(ErrorCode::UnsupportedModel, "Qwen4Exp does not support cross-session decode batching");
        } else {
            std::vector<ModelSessionBackend*> members;
            std::vector<typename Backend::Session*> caches;
            std::vector<int> tokens;
            members.reserve(rows.size());
            caches.reserve(rows.size());
            tokens.reserve(rows.size());
            for (const auto& row : rows) {
                auto* member = dynamic_cast<ModelSessionBackend*>(row.backend);
                if (!member || member->model.get() != model.get() ||
                    std::find(members.begin(), members.end(), member) != members.end()) {
                    throw Error(ErrorCode::InvalidArgument, "Decode batch requires distinct sessions of the same loaded model");
                }
                if (row.tokenId < 0 || row.tokenId >= VocabSize() ||
                    member->cache->CurrentLength() >= member->Capacity()) {
                    throw Error(ErrorCode::InvalidArgument, "Invalid decode batch token or session capacity");
                }
                members.push_back(member);
                caches.push_back(member->cache.get());
                tokens.push_back(row.tokenId);
            }
            std::vector<ExecutionResult> results(rows.size());
            aila::ScopedLogDeferral deferLogs;
            auto& device = *model->device;
            std::lock_guard<std::mutex> lock(device.mutex);
            device.CheckUsable();
            auto& context = device.context;
            auto& slot = *model->workspace;
            ExecutionFields fields{model->traceModelId};
            fields.phase = ExecutionPhase::Decode;
            fields.tokenCount = static_cast<int64_t>(rows.size());
            ExecutionScope stage(device.trace.get(), ExecutionStage::DecodeBatch, fields);
            try {
                slot.Ensure(context, rows.size());
                context.memcpy_h2d_async(slot.tokenIds.data(), tokens.data(), tokens.size() * sizeof(int));
                Tensor* logits = nullptr;
                {
                    ExecutionScope submitted(device.trace.get(), ExecutionStage::BackendForward);
                    logits = &ModelBackend().ForwardDecodeBatch(context, caches, slot.tokenIds.data_as<int>());
                    submitted.Complete();
                }
                if (!logits->valid() || logits->dtype() != dnnl::memory::data_type::bf16 ||
                    logits->ndim() != 2 || logits->shape(0) != static_cast<int64_t>(rows.size()) ||
                    logits->shape(1) != VocabSize()) {
                    throw std::runtime_error("Backend returned an invalid dense decode batch logits layout");
                }
                using bf16 = sycl::ext::oneapi::bfloat16;
                for (size_t i = 0; i < rows.size(); ++i) {
                    Tensor rowLogits = Tensor::view(context,
                        logits->template data_as<bf16>() + i * static_cast<size_t>(VocabSize()),
                        {1, static_cast<int64_t>(VocabSize())}, logits->dtype());
                    auto rowFields = members[i]->TraceFields();
                    rowFields.phase = ExecutionPhase::Decode;
                    rowFields.position = members[i]->cache->CurrentLength() - 1;
                    rowFields.tokenCount = 1;
                    results[i] = ReadOutput(rowLogits, rows[i].sampling, rows[i].captureLogits, rowFields);
                }
                context.synchronize();
                stage.Complete();
                device.CheckUsable();
                return results;
            } catch (const aila::memory::AllocationLimitExceeded& error) {
                // A projection allocation can fail after some rows/layers updated
                // state. Reset every selected session only after the queue drains.
                // Nonmembers remain usable and keep their KV/recurrent history.
                device.DrainNoThrow();
                device.CheckUsable();
                for (auto* member : members) {
                    ExecutionScope reset(device.trace.get(), ExecutionStage::SessionReset, member->TraceFields());
                    member->MutateCache([&] { member->ModelBackend().Reset(*member->cache); });
                    reset.Complete();
                }
                device.CheckUsable();
                throw Error(ErrorCode::ResourceLimit, error.what());
            } catch (const std::exception& error) {
                device.RecordFaultNoThrow(error.what());
                device.DrainNoThrow();
                throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
            } catch (...) {
                device.RecordFaultNoThrow("Unknown batched device execution failure");
                device.DrainNoThrow();
                throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
            }
        }
    }

private:
    ExecutionFields TraceFields() const noexcept { return {model->traceModelId, traceSessionId}; }
    Backend& ModelBackend() const { return static_cast<Backend&>(*model->backend); }

    ExecutionResult ReadOutput(Tensor& logits, const SamplingInput* sampling,
                               bool captureLogits, ExecutionFields fields = {}) {
        auto& device = *model->device;
        auto& context = device.context;
        auto& slot = *model->workspace;
        ExecutionResult result;
        if (sampling) {
            ExecutionScope sampled(device.trace.get(), ExecutionStage::Sampling, fields);
            if (!sampling->config.do_sample) {
                if constexpr (std::is_same_v<Backend, Qwen4ExpBnb4Backend>) {
                    ops::greedy_bf16(context, logits.data_as<sycl::ext::oneapi::bfloat16>(),
                        VocabSize(), slot.greedyPartials.data_as<ops::GreedyResult>(),
                        slot.greedyResult.data_as<ops::GreedyResult>());
                    ops::GreedyResult selected{};
                    context.memcpy_d2h(&selected, slot.greedyResult.data(), sizeof(selected));
                    if (selected.nonfinite || selected.index < 0 || selected.index >= VocabSize())
                        throw std::runtime_error("Qwen4Exp returned nonfinite or invalid greedy logits");
                    result.tokenId = selected.index;
                } else {
                    ops::argmax(context, logits, VocabSize(), slot.sampledId.data_as<int>());
                    context.memcpy_d2h(&result.tokenId, slot.sampledId.data(), sizeof(int));
                }
            } else if (!sampling->forceHost && ops::can_use_device_sampling(VocabSize(), sampling->config)) {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                ops::sample_with_config_device(context, logits, VocabSize(), sampling->config,
                                               uniform(sampling->rng), slot.sampledId.data_as<int>());
                context.memcpy_d2h(&result.tokenId, slot.sampledId.data(), sizeof(int));
            } else {
                result.tokenId = ops::sample_with_config(context, logits, VocabSize(),
                    sampling->config, sampling->generatedIds, sampling->rng);
            }
            sampled.Complete();
        }
        if (captureLogits) {
            ExecutionScope captured(device.trace.get(), ExecutionStage::LogitsRead, fields);
            using bf16 = sycl::ext::oneapi::bfloat16;
            std::vector<bf16> hostLogits(static_cast<size_t>(VocabSize()));
            context.memcpy_d2h(hostLogits.data(), logits.data(), hostLogits.size() * sizeof(bf16));
            result.logits.resize(hostLogits.size());
            std::transform(hostLogits.begin(), hostLogits.end(), result.logits.begin(),
                           [](bf16 value) { return static_cast<float>(value); });
            captured.Complete();
        }
        return result;
    }

    // Reset/rollback can submit zeroing and host-to-device state restores.
    // Drain them before returning so borrowed host storage may be reused, and
    // preserve the same execution-domain failure semantics as Forward.
    template <typename Mutation>
    void MutateCache(Mutation&& mutate) {
        auto& device = *model->device;
        try {
            mutate();
            device.context.synchronize();
        } catch (const std::exception& error) {
            device.RecordFaultNoThrow(error.what());
            device.DrainNoThrow();
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        } catch (...) {
            device.RecordFaultNoThrow("Unknown session state mutation failure");
            device.DrainNoThrow();
            throw Error(ErrorCode::RuntimeFault, device.FaultMessage());
        }
    }

    std::shared_ptr<LoadedModelState> model;
    uint64_t traceSessionId = 0;
    std::unique_ptr<typename Backend::Session> cache;
    std::unique_ptr<Qwen4ExpMtpRuntime::State> mtpState;
};

} // namespace detail

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
        // There is no published domain to drain when Context construction
        // fails, but its device exception must still reach the Worker typed.
        detail::RuntimeFaultRecord creationFault;
        detail::RunDeviceLifecycle<sycl::exception, dnnl::error>(creationFault, [&] {
            device = std::make_shared<detail::DeviceState>();
        }, [] {});
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
    return MemoryLocked();
}

std::optional<MemorySnapshot> DeviceRuntime::TryMemory() const {
    std::unique_lock<std::mutex> lock(state->mutex, std::try_to_lock);
    if (!lock.owns_lock()) return std::nullopt;
    return MemoryLocked();
}

MemorySnapshot DeviceRuntime::MemoryLocked() const {
    return {state->context.current_allocated_bytes(), state->context.peak_allocated_bytes(),
            state->context.current_pinned_host_bytes(), state->context.peak_pinned_host_bytes(),
            state->context.device_allocation_count(), state->context.pinned_host_allocation_count(),
            state->context.device_allocation_limit(), state->context.pinned_host_allocation_limit()};
}

void DeviceRuntime::SetAllocationLimits(size_t deviceBytes, size_t pinnedHostBytes) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->CheckUsable();
    if (state->loadedModels) throw Error(ErrorCode::Busy, "Allocation limits cannot change with live models");
    try { state->context.set_allocation_limits(deviceBytes, pinnedHostBytes); }
    catch (const aila::memory::AllocationLimitExceeded& error) {
        throw Error(ErrorCode::ResourceLimit, error.what());
    }
}

std::optional<std::string> DeviceRuntime::FaultMessage() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->RecordTraceFailureNoThrow();
    if (!state->fault.Faulted()) return std::nullopt;
    return std::string(state->FaultMessage());
}

LoadedModel::LoadedModel(std::shared_ptr<detail::LoadedModelState> value)
    : state(std::move(value)) {}

std::shared_ptr<LoadedModel> LoadedModel::LoadQwen3Nf4(
    std::shared_ptr<DeviceRuntime> runtime,
    const std::string& modelDirectory,
    ModelOptions options) {
    return Load(std::move(runtime), modelDirectory, options, ModelFamily::Qwen3Dense);
}

std::shared_ptr<LoadedModel> LoadedModel::LoadQwen35Nf4(
    std::shared_ptr<DeviceRuntime> runtime,
    const std::string& modelDirectory,
    ModelOptions options) {
    return Load(std::move(runtime), modelDirectory, options, ModelFamily::Qwen35Hybrid);
}

std::shared_ptr<LoadedModel> LoadedModel::LoadQwen35Bf16(
    std::shared_ptr<DeviceRuntime> runtime, const std::string& modelDirectory, ModelOptions options) {
    return Load(std::move(runtime), modelDirectory, options, ModelFamily::Qwen35Hybrid, true);
}

std::shared_ptr<LoadedModel> LoadedModel::LoadQwen4ExpNf4(
    std::shared_ptr<DeviceRuntime> runtime, const std::string& modelDirectory, ModelOptions options) {
    return Load(std::move(runtime), modelDirectory, options, ModelFamily::Qwen4Exp);
}

std::shared_ptr<LoadedModel> LoadedModel::Load(
    std::shared_ptr<DeviceRuntime> runtime,
    const std::string& modelDirectory,
    ModelOptions options, ModelFamily family, bool denseWeights) {
    aila::ScopedLogDeferral deferLogs;
    if (!runtime || modelDirectory.empty() || options.maxSequenceLength <= 0 || options.maxSessions == 0) {
        throw Error(ErrorCode::InvalidArgument, "Runtime, model directory and positive model limits are required");
    }
    const bool hybrid = family == ModelFamily::Qwen35Hybrid;
    const bool qwen4exp = family == ModelFamily::Qwen4Exp;
    static const bool requestedMtp = aila::env::read_flag("AILA_Q4E_SERVER_MTP", false) ||
                                     aila::env::read_flag("AILA_Q4E_MTP_DRAFT", false);
    bool enableMtp = qwen4exp && requestedMtp;
    const bool quantizedCache = hybrid
        ? aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT")
        : aila::env::read_flag("AILA_KV_QUANT", false);
    if (quantizedCache && !(qwen4exp && options.allowQuantizedKv)) {
        throw Error(ErrorCode::InvalidArgument,
                    hybrid ? "Qwen3.5 shared sessions require BF16 KV; set AILA_VLM_KV_QUANT=0"
                           : qwen4exp ? "Qwen4Exp shared sessions require BF16 KV; set AILA_KV_QUANT=0"
                           : "Qwen3 shared sessions require BF16 KV; set AILA_KV_QUANT=0");
    }
    auto loaded = std::make_shared<detail::LoadedModelState>();
    loaded->device = runtime->state;
    loaded->traceModelId = loaded->device->trace ? loaded->device->trace->NextObjectId() : 0;
    loaded->denseWeights = denseWeights;
    std::string error;
    if (!aila::modelspec::load_from_dir(modelDirectory, loaded->spec, &error)) {
        throw Error(ErrorCode::InvalidArgument, error);
    }
    const auto& spec = loaded->spec;
    if (spec.family != family || (denseWeights ? spec.is_quantized() :
        (!spec.is_bitsandbytes_4bit() || spec.quantization.bnb_4bit_quant_type != "nf4"))) {
        throw Error(ErrorCode::UnsupportedModel,
                    denseWeights ? "This shared-model loader requires unquantized Qwen3.5 Hybrid BF16"
                    : hybrid ? "This shared-model loader requires Qwen3.5 Hybrid NF4"
                    : qwen4exp ? "This shared-model loader requires Qwen4Exp NF4"
                           : "This shared-model loader requires Qwen3 NF4");
    }
    aila::memory::SessionMemoryReservation reservation;
    try {
        if (hybrid) {
            const auto& config = spec.qwen35_text;
            if (!is_supported_qwen35_hybrid_text_spec(config)) {
                throw Error(ErrorCode::UnsupportedModel, "Unsupported Qwen3.5 Hybrid dimensions");
            }
            if (config.vocab_size <= 0 || config.max_position_embeddings <= 0) {
                throw Error(ErrorCode::InvalidArgument, "Invalid Qwen3.5 vocabulary or context capacity");
            }
            options.maxSequenceLength = std::min(options.maxSequenceLength, config.max_position_embeddings);
            reservation = aila::memory::qwen35_session_memory(
                config, options.maxSequenceLength, options.maxRecurrentSnapshots);
            loaded->vocabSize = config.vocab_size;
        } else if (qwen4exp) {
            const auto& config = spec.qwen4exp_text;
            if (!is_exact_qwen4exp_text_spec(config))
                throw Error(ErrorCode::UnsupportedModel, "Unsupported Qwen4Exp dimensions");
            if (config.vocab_size <= 0 || config.max_position_embeddings <= 0)
                throw Error(ErrorCode::InvalidArgument, "Invalid Qwen4Exp vocabulary or context capacity");
            options.maxSequenceLength = std::min(options.maxSequenceLength, config.max_position_embeddings);
            // The session builder allocates 1-byte K/V plus two F32 per-token
            // scales when int8 KV is opted in; the reservation must mirror the
            // same layout or the worker rejects the session at creation.
            const size_t kvElementBytes =
                (quantizedCache && aila::env::kv_quant_format_is_int8()) ? 1 : 2;
            reservation = aila::memory::qwen4exp_session_memory(
                config, options.maxSequenceLength, options.maxRecurrentSnapshots,
                kvElementBytes);
            loaded->vocabSize = config.vocab_size;
        } else {
            const auto& config = spec.qwen3;
            if (config.vocab_size <= 0 || config.hidden_size <= 0 ||
                config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
                config.num_attention_heads % config.num_key_value_heads != 0 ||
                config.max_position_embeddings <= 0) {
                throw Error(ErrorCode::InvalidArgument, "Invalid Qwen3 model dimensions");
            }
            options.maxSequenceLength = std::min(options.maxSequenceLength, config.max_position_embeddings);
            reservation = aila::memory::qwen3_session_memory(config, options.maxSequenceLength);
            loaded->vocabSize = config.vocab_size;
        }
    } catch (const std::invalid_argument& error) {
        throw Error(ErrorCode::InvalidArgument, error.what());
    }
    loaded->options = options;
    // The draft window is bounded by AILA_Q4E_MTP_MAX_CONTEXT, not by the
    // session budget, so admission and reservation both use the draft capacity.
    const int mtpCapacity = Qwen4ExpMtpRuntime::EffectiveCapacity(options.maxSequenceLength);
    enableMtp = enableMtp && options.maxSequenceLength >= 2 && mtpCapacity >= 2 &&
        mtpCapacity <= options.maxSequenceLength;
    if (enableMtp) {
        reservation.deviceBytes = aila::memory::checked_memory_sum(reservation.deviceBytes,
            Qwen4ExpMtpRuntime::SessionBytes(mtpCapacity));
    }
    loaded->sessionReservationBytes = reservation.deviceBytes;
    loaded->sessionHostReservationBytes = reservation.hostBytes;
    // Declare the lock after loaded: a failed load frees resources only after
    // releasing this lock (LoadedModelState's destructor acquires it again).
    std::unique_lock<std::mutex> lock(loaded->device->mutex);
    loaded->device->CheckUsable();
    auto& context = loaded->device->context;
    ExecutionScope stage(loaded->device->trace.get(), ExecutionStage::ModelLoad, {loaded->traceModelId});
    detail::RunDeviceLifecycle<sycl::exception, dnnl::error>(loaded->device->fault, [&] {
        // Q4E keeps MoE/PLE tables in the mmap; its backend uploads static
        // projections and populates the bounded expert cache on demand.
        loaded->weights = std::make_unique<ModelWeights>(qwen4exp
            ? LoadModelWeightsLazy(modelDirectory, context)
            : LoadModelWeightsFromDir(modelDirectory, context));
        if (denseWeights) {
            for (const auto& name : loaded->weights->names()) {
                const auto& tensor = loaded->weights->get(name);
                const bool textWeight = name.rfind("model.language_model.", 0) == 0 ||
                    name.rfind("model.layers.", 0) == 0 || name == "model.embed_tokens.weight" || name == "lm_head.weight";
                if (textWeight && tensor.ndim() >= 2 && tensor.dtype() != dnnl::memory::data_type::bf16)
                    throw Error(ErrorCode::UnsupportedModel, "Shared dense text projections require BF16 weights: " + name);
            }
            loaded->backend = std::make_unique<Qwen35HybridTextBackend>();
        } else if (hybrid) loaded->backend = std::make_unique<Qwen35HybridBnb4Backend>();
        else if (qwen4exp) loaded->backend = std::make_unique<Qwen4ExpBnb4Backend>();
        else loaded->backend = std::make_unique<Qwen3Bnb4Backend>();
        if (!loaded->backend->load(context, *loaded->weights, spec, options.maxSequenceLength, &error)) {
            throw Error(ErrorCode::UnsupportedModel, error);
        }
        if (enableMtp) loaded->mtp = std::make_unique<Qwen4ExpMtpRuntime>(context,
            static_cast<Qwen4ExpBnb4Backend&>(*loaded->backend), *loaded->weights, mtpCapacity);
        loaded->workspace = std::make_unique<detail::ExecutionWorkspace>();
        if (qwen4exp) {
            // Shared execution scratch is owned and budgeted once, before decode.
            loaded->workspace->Ensure(context, options.maxSequenceLength);
            loaded->workspace->greedyPartials = Tensor::allocate(context,
                {ops::GreedyGroups * static_cast<int64_t>(sizeof(ops::GreedyResult))},
                dnnl::memory::data_type::u8);
            loaded->workspace->greedyResult = Tensor::allocate(context,
                {static_cast<int64_t>(sizeof(ops::GreedyResult))}, dnnl::memory::data_type::u8);
        }
    }, [&] { context.synchronize(); });
    ++loaded->device->loadedModels;
    loaded->registeredModel = true;
    stage.Complete();
    loaded->device->CheckUsable();
    lock.unlock();
    auto* warmup_state = loaded.get();
    auto model = std::shared_ptr<LoadedModel>(new LoadedModel(std::move(loaded)));
    // Opt-in first-use warmup for the Qwen4Exp worker path. The worker loads
    // through ModelFactory::LoadQwen4Exp (this function), so the legacy
    // InferenceEngine::init warmup never runs and the first real prefill pays
    // ~80 s of first-touch kernel setup. One production-shaped chunk plus one
    // decode step moves that cost into the load phase. Skipped while a hotset
    // export is active so synthetic routes cannot pollute the profile.
    static const int warmup_mode = aila::env::read_int_raw("AILA_INIT_WARMUP", -1);
    if (qwen4exp && warmup_mode == 1 &&
        aila::env::read_string("AILA_Q4E_HOTSET_EXPORT", "").empty()) {
        auto* backend = dynamic_cast<Qwen4ExpBnb4Backend*>(warmup_state->backend.get());
        if (!backend) {
            AILA_LOG_WARN("[Q4E Warmup] Qwen4Exp backend unavailable; skipping warmup");
        } else {
            int warmup_rows = aila::env::read_int_raw("AILA_INIT_WARMUP_ROWS", 16);
            warmup_rows = std::clamp(warmup_rows, 1, warmup_state->options.maxSequenceLength);
            const auto warmup_start = std::chrono::steady_clock::now();
            backend->SetColdEpochEnabled(false);
            double prefill_ms = 0.0;
            double decode_ms = 0.0;
            std::string failure;
            try {
                auto session = model->CreateSession();
                // Varied ids: a constant dummy id leaves the PLE gather (and
                // part of routing) for the first real request, which costs
                // ~20 s on this model.
                const int warmup_vocab = std::max(1, warmup_state->vocabSize);
                std::vector<int> warmup_ids(static_cast<size_t>(warmup_rows));
                for (int row = 0; row < warmup_rows; ++row) {
                    warmup_ids[static_cast<size_t>(row)] = (row * 4099 + 12345) % warmup_vocab;
                }
                (void)session->ForwardLogits(warmup_ids);
                backend->WarmupPrefillKernels(warmup_state->device->context);
                const auto prefill_end = std::chrono::steady_clock::now();
                std::vector<int> decode_ids{warmup_ids.back()};
                (void)session->ForwardLogits(decode_ids);
                if (warmup_state->mtp) {
                    session->Reset();
                    std::vector<int> mtp_seed{warmup_ids[0], warmup_ids[1 % warmup_ids.size()]};
                    (void)session->ForwardLogits(mtp_seed);
                    (void)session->ExecuteWindow(warmup_ids[2 % warmup_ids.size()],
                                                 [](int) { return false; }, 2);
                    for (int w = 2; w <= warmup_state->mtp->ConfiguredWidth(); ++w) {
                        (void)session->ExecuteWindow(warmup_ids[(1 + w) % warmup_ids.size()],
                                                     [](int) { return true; }, w);
                    }
                    warmup_state->mtp->ResetStats();
                }
                const auto decode_end = std::chrono::steady_clock::now();
                prefill_ms = std::chrono::duration<double, std::milli>(
                    prefill_end - warmup_start).count();
                decode_ms = std::chrono::duration<double, std::milli>(
                    decode_end - prefill_end).count();
                session->Reset();
            } catch (const std::exception& error) {
                failure = error.what();
            } catch (...) {
                failure = "unknown error";
            }
            backend->SetColdEpochEnabled(true);
            const double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - warmup_start).count();
            if (failure.empty()) {
                AILA_LOG_INFO("[Q4E Warmup] rows=%d prefill_ms=%.1f decode_ms=%.1f total_ms=%.1f",
                              warmup_rows, prefill_ms, decode_ms, total_ms);
            } else {
                AILA_LOG_WARN("[Q4E Warmup] failed after %.1f ms: %s", total_ms, failure.c_str());
            }
            warmup_state->device->CheckUsable();
        }
    }
    return model;
}

std::shared_ptr<SessionState> LoadedModel::CreateSession() {
    aila::ScopedLogDeferral deferLogs;
    if (state->denseWeights) {
        auto backend = std::make_unique<detail::ModelSessionBackend<Qwen35HybridTextBackend>>(state);
        backend->Initialize();
        return std::make_shared<SessionState>(std::move(backend));
    } else if (state->spec.family == ModelFamily::Qwen35Hybrid) {
        auto backend = std::make_unique<detail::ModelSessionBackend<Qwen35HybridBnb4Backend>>(state);
        backend->Initialize();
        return std::make_shared<SessionState>(std::move(backend));
    } else if (state->spec.family == ModelFamily::Qwen4Exp) {
        auto backend = std::make_unique<detail::ModelSessionBackend<Qwen4ExpBnb4Backend>>(state);
        backend->Initialize();
        return std::make_shared<SessionState>(std::move(backend));
    } else {
        auto backend = std::make_unique<detail::ModelSessionBackend<Qwen3Bnb4Backend>>(state);
        backend->Initialize();
        return std::make_shared<SessionState>(std::move(backend));
    }
}

int LoadedModel::VocabSize() const noexcept { return state->vocabSize; }
int LoadedModel::MaxSequenceLength() const noexcept { return state->options.maxSequenceLength; }
size_t LoadedModel::SessionReservationBytes() const noexcept { return state->sessionReservationBytes; }
size_t LoadedModel::SessionHostReservationBytes() const noexcept { return state->sessionHostReservationBytes; }

size_t LoadedModel::ActiveSessions() const {
    std::lock_guard<std::mutex> lock(state->device->mutex);
    return state->activeSessions;
}

} // namespace aila::inference
