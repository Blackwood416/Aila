#include "worker/InferenceWorker.hpp"

#include "inference/Scheduler.hpp"
#include "ipc/InferenceProtocol.hpp"
#include "profile/Profiling.hpp"
#include "simdjson.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

namespace aila::worker {
namespace {

namespace wire = ipc::v2;
namespace core = inference;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

wire::ErrorCode WireError(core::ErrorCode value) noexcept {
    switch (value) {
        case core::ErrorCode::InvalidArgument: return wire::ErrorCode::InvalidArgument;
        case core::ErrorCode::Busy: return wire::ErrorCode::Busy;
        case core::ErrorCode::ContextOverflow: return wire::ErrorCode::ContextOverflow;
        case core::ErrorCode::UnsupportedModel: return wire::ErrorCode::UnsupportedModel;
        case core::ErrorCode::ResourceLimit: return wire::ErrorCode::ResourceLimit;
        case core::ErrorCode::RuntimeFault: return wire::ErrorCode::RuntimeFault;
        case core::ErrorCode::SchedulerStopped: return wire::ErrorCode::SchedulerStopped;
    }
    return wire::ErrorCode::RequestFailed;
}

wire::RequestStatus WireStatus(core::RequestStatus value) noexcept {
    switch (value) {
        case core::RequestStatus::Queued: return wire::RequestStatus::Queued;
        case core::RequestStatus::Prefill: return wire::RequestStatus::Prefill;
        case core::RequestStatus::Decode: return wire::RequestStatus::Decode;
        case core::RequestStatus::Finalizing: return wire::RequestStatus::Finalizing;
        case core::RequestStatus::Completed: return wire::RequestStatus::Completed;
        case core::RequestStatus::Cancelled: return wire::RequestStatus::Cancelled;
        case core::RequestStatus::Failed: return wire::RequestStatus::Failed;
    }
    return wire::RequestStatus::Failed;
}

wire::EndReason WireReason(core::RequestEndReason value) noexcept {
    switch (value) {
        case core::RequestEndReason::None: return wire::EndReason::None;
        case core::RequestEndReason::Length: return wire::EndReason::Length;
        case core::RequestEndReason::StopToken: return wire::EndReason::StopToken;
        case core::RequestEndReason::Cancelled: return wire::EndReason::Cancelled;
        case core::RequestEndReason::OutputBackpressure: return wire::EndReason::OutputBackpressure;
        case core::RequestEndReason::SchedulerStopped: return wire::EndReason::SchedulerStopped;
        case core::RequestEndReason::RuntimeFault: return wire::EndReason::RuntimeFault;
        case core::RequestEndReason::Failed: return wire::EndReason::Failed;
    }
    return wire::EndReason::Failed;
}

std::string BoundedMessage(std::string_view message) {
    size_t size = std::min(message.size(), wire::kMaxErrorBytes);
    if (size < message.size()) {
        while (size && (static_cast<unsigned char>(message[size]) & 0xc0) == 0x80) --size;
    }
    const auto text = message.substr(0, size);
    return simdjson::validate_utf8(text) ? std::string(text) : "Diagnostic was not valid UTF-8";
}

wire::Summary MakeSummary(const core::ScheduledRequestSnapshot& snapshot) {
    wire::Summary summary;
    summary.status = WireStatus(snapshot.status);
    summary.endReason = WireReason(snapshot.endReason);
    summary.promptTokensProcessed = snapshot.promptTokensProcessed;
    summary.generatedTokens = snapshot.generatedTokens;
    summary.eventCount = core::IsTerminal(snapshot.status) ? snapshot.generatedTokens + 1 : 0;
    if (snapshot.errorCode) summary.errorCode = WireError(*snapshot.errorCode);
    else if (snapshot.status == core::RequestStatus::Failed) summary.errorCode = wire::ErrorCode::RequestFailed;
    summary.message = BoundedMessage(snapshot.error);
    return summary;
}

core::RequestOptions CoreOptions(const wire::Command& command) {
    core::RequestOptions options;
    options.prefillChunkSize = static_cast<int>(command.prefillChunkSize);
    options.stopTokenIds.assign(command.stopTokenIds.begin(), command.stopTokenIds.end());
    options.thinkingBudget = command.thinkingBudget;
    options.prefillOnly = command.prefillOnly;
    options.forceHostSampling = command.forceHostSampling;
    options.legacyPrefillLogits = command.legacyPrefillLogits;
    auto& target = options.generation;
    const auto& source = command.generation;
    target.max_new_tokens = static_cast<int>(source.maxNewTokens);
    target.do_sample = source.doSample;
    target.temperature = static_cast<float>(source.temperature);
    target.top_k = static_cast<int>(source.topK);
    target.top_p = static_cast<float>(source.topP);
    target.use_fixed_seed = source.useFixedSeed;
    target.sampling_seed = source.samplingSeed;
    target.repetition_penalty = static_cast<float>(source.repetitionPenalty);
    target.presence_penalty = static_cast<float>(source.presencePenalty);
    target.frequency_penalty = static_cast<float>(source.frequencyPenalty);
    target.thinking_budget_tokens = -1;
    return options;
}

core::SchedulerOptions CoreLimits(const wire::Limits& limits, uint32_t maxDecodeBatchSize) {
    core::SchedulerOptions options;
    options.maxActiveRequests = limits.maxActiveRequests;
    options.maxDecodeBatchSize = maxDecodeBatchSize;
    options.maxOutstandingRequests = limits.maxOutstandingRequests;
    options.maxPromptTokens = limits.maxPromptTokens;
    options.maxGeneratedTokens = limits.maxGeneratedTokens;
    options.maxStopTokenIds = limits.maxStopTokenIds;
    options.maxOutstandingTokens = static_cast<size_t>(limits.maxOutstandingTokens);
    options.maxPrefillChunkSize = static_cast<int>(limits.maxPrefillChunkSize);
    options.maxBufferedTokensPerRequest = limits.maxBufferedTokensPerRequest;
    options.maxSessionCacheBytes = static_cast<size_t>(limits.maxSessionCacheBytes);
    return options;
}

bool IsControl(std::string_view method) noexcept {
    return method != "model.load" && method != "session.create" && method != "request.submit" &&
        method != "session.reset" && method != "session.truncate" && method != "worker.memory_query";
}

struct CommandFailure : std::runtime_error {
    CommandFailure(wire::ErrorCode code, const std::string& message)
        : std::runtime_error(message), code(code) {}
    wire::ErrorCode code;
};

struct OutputQueue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<InferenceWorkerOutput> frames;
    bool stopped = false;

    void Push(InferenceWorkerOutput output) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopped) throw std::runtime_error("Worker output queue is closed");
            frames.push_back(std::move(output));
        }
        changed.notify_one();
    }
    bool Read(InferenceWorkerOutput& output, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait_for(lock, timeout, [&] { return stopped || !frames.empty(); });
        if (frames.empty()) return false;
        output = std::move(frames.front());
        frames.pop_front();
        return true;
    }
    void Stop() noexcept {
        { std::lock_guard<std::mutex> lock(mutex); stopped = true; }
        changed.notify_all();
    }
};

} // namespace

struct InferenceWorker::State : std::enable_shared_from_this<State> {
    struct Input {
        std::weak_ptr<State> owner;
        ipc::Frame frame;
        uint64_t rpcId = 0;
        std::string method;
        uint64_t bytes = 0;
        bool normal = false;
        ~Input() {
            if (auto state = owner.lock()) {
                {
                    std::lock_guard<std::mutex> lock(state->inputMutex);
                    --state->pendingCommands;
                    if (normal) --state->pendingNormalCommands;
                    state->ingressBytes -= bytes;
                }
                state->changed.notify_one();
            }
        }
    };
    struct ModelRecord {
        uint64_t id = 0;
        std::string_view capability;
        bool closed = false;
        size_t sessionLimit = 0;
        size_t cacheLimit = 0;
        size_t sessions = 0; // Includes allocation/reaper jobs, not just handles.
        std::shared_ptr<InferenceModel> model;
        std::optional<wire::ModelMemoryInfo> memory;
    };
    struct SessionRecord {
        uint64_t id = 0;
        bool closed = false;
        bool contextPending = false; // Control-thread reservation spans queueing through publication.
        std::shared_ptr<ModelRecord> model;
        std::shared_ptr<core::SessionState> session;
    };
    struct RequestRecord {
        wire::Identity identity;
        std::shared_ptr<SessionRecord> executingSession;
        // Must survive release: ScheduledRequest destruction detaches its ring.
        std::shared_ptr<core::ScheduledRequest> scheduled;
        wire::Summary summary;
        uint64_t lastSent = 0;
        uint64_t sentTokens = 0;
        uint64_t acknowledged = 0;
        uint64_t tokenReservation = 0;
        uint64_t byteReservation = 0;
        std::atomic<bool> eventPending{false};
        std::atomic<bool> finalPending{false};
        bool finalQueued = false;
        bool terminalQueued = false;
        bool drained = false;
        bool closing = false;
    };
    struct LogMessage { int level; std::string message; uint64_t bytes; };

    State(std::string value, std::unique_ptr<InferenceModelFactory> modelFactory)
        : epoch(std::move(value)), factory(std::move(modelFactory)) {
        if (!wire::valid_epoch(epoch) || !factory) throw std::invalid_argument("Invalid inference Worker construction");
    }

    void Start() {
        auto self = shared_from_this();
        lifecycleThread = std::thread([self] { self->LifecycleLoop(); });
        try { controlThread = std::thread([self] { self->ControlLoop(); }); }
        catch (...) {
            { std::lock_guard<std::mutex> lock(lifecycleMutex); lifecycleStopping = true; }
            lifecycleChanged.notify_one();
            lifecycleThread.join();
            throw;
        }
    }

    bool Enqueue(ipc::Frame frame, std::string& error) {
        error.clear();
        if (frame.header.protocol != wire::kProtocolVersion ||
            frame.header.abi != wire::kPublicAbiVersion || frame.header.kind != "request" ||
            frame.header.request_id == 0) {
            error = "Invalid v2 command envelope";
            Fail(error);
            return false;
        }
        try {
            // Reserve raw, decoded and response/staging payloads together.
            // These are conservative payload reservations, not process RSS.
            const uint64_t bytes = 2048 + 3 * uint64_t(frame.header.payload_json.size()) +
                2 * uint64_t(frame.attachment.size());
            const bool normal = !IsControl(frame.header.method);
            std::lock_guard<std::mutex> lock(inputMutex);
            if (abort.load() || closed.load() || ingressClosed) {
                error = "Worker connection is closing";
                return false;
            }
            if (frame.header.request_id <= lastRpcId) {
                error = "RPC IDs must be strictly increasing and nonzero";
            } else if (pendingCommands >= limits.maxPendingCommands ||
                       (normal && pendingNormalCommands >= limits.maxPendingCommands - limits.reservedControlCommands) ||
                       bytes > limits.maxIngressBytes - ingressBytes) {
                error = "Peer exceeded negotiated command ingress limits";
            }
            if (!error.empty()) {
                Fail(error);
                return false;
            }
            auto input = std::make_shared<Input>();
            input->frame = std::move(frame);
            input->rpcId = input->frame.header.request_id;
            input->method = input->frame.header.method;
            input->bytes = bytes;
            input->normal = normal;
            inputs.push_back(input);
            lastRpcId = input->rpcId;
            ++pendingCommands;
            if (normal) ++pendingNormalCommands;
            ingressBytes += bytes;
            input->owner = shared_from_this();
            changed.notify_one();
            return true;
        } catch (const std::exception& exception) {
            error = exception.what();
            Fail(error);
            return false;
        }
    }

    void Fail(std::string_view message) noexcept {
        try {
            std::lock_guard<std::mutex> lock(failureMutex);
            if (failure.empty()) failure = BoundedMessage(message);
        } catch (...) {}
        abort.store(true);
        changed.notify_all();
        responses.Stop();
        events.Stop();
    }

    void Post(std::function<void()> completion) {
        {
            std::lock_guard<std::mutex> lock(inputMutex);
            completions.push_back(std::move(completion));
        }
        changed.notify_one();
    }

    void Lifecycle(std::function<void()> operation) {
        auto self = shared_from_this();
        {
            std::lock_guard<std::mutex> lock(lifecycleMutex);
            lifecycleJobs.push_back([self, operation = std::move(operation)]() mutable {
                operation();
                self->Post([self] { --self->lifecyclePending; });
            });
            // Admission rollback must not leave a phantom job if allocating
            // the closure or queue node failed before the job was published.
            ++lifecyclePending;
        }
        lifecycleChanged.notify_one();
    }

    void LifecycleLoop() noexcept {
        for (;;) {
            std::function<void()> operation;
            {
                std::unique_lock<std::mutex> lock(lifecycleMutex);
                lifecycleChanged.wait(lock, [&] { return lifecycleStopping || !lifecycleJobs.empty(); });
                if (lifecycleJobs.empty()) return;
                operation = std::move(lifecycleJobs.front());
                lifecycleJobs.pop_front();
            }
            try { operation(); }
            catch (const std::exception& exception) { Fail(exception.what()); }
            catch (...) { Fail("Unknown Worker lifecycle failure"); }
        }
    }

    void PublishFault(std::string_view message) noexcept {
        try {
            std::lock_guard<std::mutex> lock(faultMutex);
            if (!domainFaulted.load()) domainFault = BoundedMessage(message);
            domainFaulted.store(true);
        } catch (...) { domainFaulted.store(true); }
        if (scheduler) scheduler->ReportRuntimeFault(message);
        changed.notify_one();
    }

    void CheckLifecycleFault() {
        const auto message = factory ? factory->RuntimeFaultMessage() : std::string{};
        if (!message.empty()) PublishFault(message);
    }

    void RequireReady() const {
        if (domainFaulted.load()) throw CommandFailure(wire::ErrorCode::RuntimeFault, "Worker device domain has failed");
        if (stopping.load() || abort.load()) throw CommandFailure(wire::ErrorCode::SchedulerStopped, "Worker is stopping");
        if (!configured) throw CommandFailure(wire::ErrorCode::InvalidArgument, "Configure Worker before creating inference objects");
    }

    wire::Response ResponseFor(const Input& input) const {
        wire::Response response;
        response.rpcId = input.rpcId;
        response.epoch = epoch;
        if (!wire::parse_method(input.method, response.method)) {
            throw CommandFailure(wire::ErrorCode::InvalidArgument, "Unknown Worker command");
        }
        return response;
    }

    void SendResponse(wire::Response response, std::shared_ptr<Input> input = {},
                      std::function<void()> afterWrite = {}) {
        ipc::Frame frame;
        wire::ValidationError error;
        if (!wire::make_response(response, frame, error, limits)) throw std::runtime_error(error.message);
        responses.Push({std::move(frame), [input = std::move(input), afterWrite = std::move(afterWrite)]() mutable {
            input.reset();
            if (afterWrite) afterWrite();
        }});
    }

    void Reject(const std::shared_ptr<Input>& input, wire::ErrorCode code, std::string_view message) {
        wire::Method method;
        if (!wire::parse_method(input->method, method)) {
            // An unknown method in an otherwise valid envelope is a command
            // error. Preserve its exact name so its caller can correlate it.
            ipc::Frame frame;
            frame.header.protocol = wire::kProtocolVersion;
            frame.header.abi = wire::kPublicAbiVersion;
            frame.header.request_id = input->rpcId;
            frame.header.method = input->method;
            frame.header.kind = "error";
            frame.header.payload_json = "{\"epoch\":\"" + epoch +
                "\",\"errorCode\":\"InvalidArgument\",\"message\":\"Unknown Worker command\"}";
            responses.Push({std::move(frame), [reservation = input]() mutable { reservation.reset(); }});
            return;
        }
        auto response = ResponseFor(*input);
        response.kind = wire::ResponseKind::Error;
        response.errorCode = code;
        response.message = BoundedMessage(message);
        if (response.method == wire::Method::RequestSubmit) response.phase = wire::SubmissionPhase::Rejected;
        SendResponse(std::move(response), input);
    }

    void RejectException(const std::shared_ptr<Input>& input, std::exception_ptr exception) {
        try { std::rethrow_exception(exception); }
        catch (const CommandFailure& error) { Reject(input, error.code, error.what()); }
        catch (const core::Error& error) {
            if (error.Code() == core::ErrorCode::RuntimeFault) PublishFault(error.what());
            Reject(input, WireError(error.Code()), error.what());
        }
        catch (const std::bad_alloc&) { Reject(input, wire::ErrorCode::ResourceLimit, "Host allocation failed"); }
        catch (const std::invalid_argument& error) { Reject(input, wire::ErrorCode::InvalidArgument, error.what()); }
        catch (const std::exception& error) { Reject(input, wire::ErrorCode::RequestFailed, error.what()); }
        catch (...) { Reject(input, wire::ErrorCode::RequestFailed, "Unknown inference operation failure"); }
    }

    std::shared_ptr<ModelRecord> Model(uint64_t id) {
        const auto found = models.find(id);
        if (found == models.end() || found->second->closed)
            throw CommandFailure(wire::ErrorCode::InvalidHandle, "Model handle is closed or unknown");
        return found->second;
    }
    std::shared_ptr<SessionRecord> Session(uint64_t id) {
        const auto found = sessions.find(id);
        if (found == sessions.end() || found->second->closed)
            throw CommandFailure(wire::ErrorCode::InvalidHandle, "Session handle is closed or unknown");
        return found->second;
    }
    std::shared_ptr<RequestRecord> Request(uint64_t id, bool cleanup = false) {
        const auto found = requests.find(id);
        if (found == requests.end() || (!cleanup && found->second->closing))
            throw CommandFailure(wire::ErrorCode::InvalidHandle, "Request handle is closed or unknown");
        return found->second;
    }

    wire::WorkerInfo QueryWorker() {
        wire::WorkerInfo info;
        info.domainStatus = domainFaulted.load() ? wire::DomainStatus::Faulted :
            stopping.load() ? wire::DomainStatus::Stopping :
            configured ? wire::DomainStatus::Ready : wire::DomainStatus::Starting;
        if (scheduler) {
            const auto snapshot = scheduler->Snapshot();
            if (snapshot.faulted && !domainFaulted.load()) PublishFault(snapshot.faultMessage);
            if (snapshot.faulted) info.domainStatus = wire::DomainStatus::Faulted;
            info.accepting = snapshot.accepting && !stopping.load() && !domainFaulted.load();
            info.stopping = snapshot.stopping || stopping.load();
            info.faulted = snapshot.faulted || domainFaulted.load();
            info.outstandingRequests = snapshot.outstandingRequests;
            info.queuedRequests = snapshot.queuedRequests;
            info.activeRequests = snapshot.activeRequests;
            info.reservedTokens = snapshot.reservedTokens;
            info.reservedSessionCacheBytes = snapshot.reservedSessionCacheBytes;
        }
        info.ipcOutstandingRequests = ipcOutstanding;
        info.retainedRequestHandles = requests.size();
        info.transportBytes = transportBytes + uint64_t(requests.size()) * RetainedHandleBytes;
        info.droppedLogs = droppedLogs.load();
        {
            std::lock_guard<std::mutex> lock(inputMutex);
            info.pendingCommands = pendingCommands;
            info.ingressBytes = ingressBytes;
            info.transportBytes += pendingCommands * limits.maxFinalFrameBytes;
        }
        { std::lock_guard<std::mutex> lock(logMutex); info.transportBytes += logBytes; }
        return info;
    }

    void Configure(const std::shared_ptr<Input>& input, const wire::Command& command) {
        if (configured || stopping.load()) throw CommandFailure(wire::ErrorCode::InvalidArgument, "Worker limits are already fixed");
        wire::Limits selected;
        wire::ValidationError error;
        if (!wire::negotiate(command.limits, command.requiredCapabilities,
                             wire::configure_limits_ceiling(wire::hard_limits(), command.requiredCapabilities),
                             wire::supported_capabilities(), selected, error)) {
            throw CommandFailure(error.code, error.message);
        }
        const uint64_t reservedBase = uint64_t(selected.maxPendingCommands) * selected.maxFinalFrameBytes +
            selected.maxLogQueueBytes + uint64_t(selected.maxRetainedRequestHandles) * RetainedHandleBytes;
        const uint64_t oneRequest = uint64_t(selected.tokenWindow) * selected.maxTokenFrameBytes +
            selected.maxTerminalFrameBytes + selected.maxFinalFrameBytes + selected.maxTokenFrameBytes;
        if (reservedBase > selected.maxTransportBytes || oneRequest > selected.maxTransportBytes - reservedBase)
            throw CommandFailure(wire::ErrorCode::InvalidArgument, "Transport budget cannot reserve control, handle, log and request slots");
        auto newScheduler = std::make_unique<core::Scheduler>(CoreLimits(selected, command.maxDecodeBatchSize));
        {
            std::scoped_lock lock(inputMutex, logMutex);
            if (pendingCommands > selected.maxPendingCommands || ingressBytes > selected.maxIngressBytes)
                throw CommandFailure(wire::ErrorCode::ResourceLimit, "Configure exceeds selected ingress budget");
            if (logBytes > selected.maxLogQueueBytes)
                throw CommandFailure(wire::ErrorCode::ResourceLimit, "Configure cannot shrink below queued log bytes");
            limits = selected;
            logLimit.store(limits.maxLogQueueBytes);
        }
        scheduler = std::move(newScheduler);
        transportBase = reservedBase;
        configured = true;
        auto response = ResponseFor(*input);
        response.limits = limits;
        response.maxDecodeBatchSize = command.maxDecodeBatchSize;
        response.capabilities = wire::supported_capabilities();
        response.worker.domainStatus = wire::DomainStatus::Ready;
        SendResponse(std::move(response), input);
    }

    void QueryRuntimeMemory(const std::shared_ptr<Input>& input) {
        RequireReady();
        if (lifecyclePending) {
            // Load/create/release may be waiting for GPU work. Do not queue
            // observation behind that wait or accumulate diagnostic jobs.
            auto response = ResponseFor(*input);
            response.runtimeMemory = {wire::RuntimeMemoryState::Busy, std::nullopt};
            SendResponse(std::move(response), input);
            return;
        }
        auto self = shared_from_this();
        Lifecycle([self, input] {
            wire::RuntimeMemoryInfo memory;
            std::exception_ptr failure;
            try {
                self->RequireReady();
                memory = self->factory->QueryRuntimeMemory();
            } catch (...) { failure = std::current_exception(); }
            self->Post([self, input, memory, failure] {
                if (failure) { self->RejectException(input, failure); return; }
                auto response = self->ResponseFor(*input);
                response.runtimeMemory = memory;
                self->SendResponse(std::move(response), input);
            });
        });
    }

    void LoadModel(const std::shared_ptr<Input>& input, const wire::Command& command) {
        RequireReady();
        if (liveModels >= limits.maxLoadedModels) throw CommandFailure(wire::ErrorCode::ResourceLimit, "Loaded model limit reached");
        if (wire::model_capability(command.backend).empty())
            throw CommandFailure(wire::ErrorCode::UnsupportedModel,
                                 "Worker supports Qwen3 NF4, Qwen3.5 Hybrid NF4/BF16 and Qwen4Exp NF4");
        const auto capability = wire::model_capability(command.backend, command.kvType);
        if (capability.empty())
            throw CommandFailure(wire::ErrorCode::InvalidArgument, "Shared inference requires BF16 KV");
        if (command.maxSessions > limits.maxSessions)
            throw CommandFailure(wire::ErrorCode::ResourceLimit, "Model session limit exceeds negotiated limit");
        if (nextModelId == (std::numeric_limits<uint64_t>::max)())
            throw CommandFailure(wire::ErrorCode::ResourceLimit, "Model IDs exhausted; restart Worker");
        auto record = std::make_shared<ModelRecord>();
        record->id = nextModelId++;
        record->capability = capability;
        record->sessionLimit = command.maxSessions;
        record->cacheLimit = static_cast<size_t>(command.maxSessionCacheBytes);
        if (limits.maxSessionCacheBytes && (!record->cacheLimit || record->cacheLimit > limits.maxSessionCacheBytes))
            record->cacheLimit = static_cast<size_t>(limits.maxSessionCacheBytes);
        const InferenceModelOptions options{static_cast<int>(command.maxSequenceLength), record->sessionLimit,
                                             record->cacheLimit, command.allocationLimits, command.kvType};
        auto self = shared_from_this();
        ++liveModels;
        try {
            Lifecycle([self, input, record, directory = command.modelDirectory,
                       backend = command.backend, options] {
                std::exception_ptr failure;
                try {
                    self->RequireReady();
                    if (backend == "qwen3_nf4") record->model = self->factory->LoadQwen3(directory, options);
                    else if (backend == "qwen35_nf4") record->model = self->factory->LoadQwen35(directory, options);
                    else if (backend == "qwen35_bf16") record->model = self->factory->LoadQwen35Bf16(directory, options);
                    else if (backend == "qwen4exp_nf4") record->model = self->factory->LoadQwen4Exp(directory, options);
                    else throw CommandFailure(wire::ErrorCode::UnsupportedModel, "Unsupported model backend");
                    self->CheckLifecycleFault();
                    self->RequireReady();
                    if (!record->model || record->model->VocabSize() <= 0 || record->model->MaxSequenceLength() <= 0)
                        throw core::Error(core::ErrorCode::RuntimeFault, "Model factory returned invalid metadata");
                    record->memory = record->model->MemoryInfo();
                    if (record->memory.has_value() != options.allocationLimits.has_value() ||
                        (record->memory && !(record->memory->limits == *options.allocationLimits))) {
                        throw core::Error(core::ErrorCode::RuntimeFault, "Model factory did not enforce allocation limits");
                    }
                } catch (...) {
                    failure = std::current_exception();
                    record->model.reset();
                    self->CheckLifecycleFault();
                }
                self->Post([self, input, record, failure] {
                    if (failure) { --self->liveModels; self->RejectException(input, failure); return; }
                    self->models.emplace(record->id, record);
                    if (self->stopping.load() || self->domainFaulted.load()) {
                        record->closed = true;
                        self->Reject(input, self->domainFaulted.load() ? wire::ErrorCode::RuntimeFault : wire::ErrorCode::SchedulerStopped,
                                     "Worker stopped before model publication");
                        return;
                    }
                    auto response = self->ResponseFor(*input);
                    response.model = {record->id, static_cast<uint32_t>(record->model->VocabSize()),
                        static_cast<uint32_t>(record->model->MaxSequenceLength()),
                        record->model->SessionReservationBytes(),
                        {std::string(record->capability), "token_requests_v1", "session_lifecycle_v1"}};
                    if (record->memory) {
                        response.model.memory = record->memory;
                        response.model.capabilities.emplace_back("allocation_limits_v1");
                    }
                    self->SendResponse(std::move(response), input);
                });
            });
        } catch (...) { --liveModels; throw; }
    }

    void CreateSession(const std::shared_ptr<Input>& input, const wire::Command& command) {
        RequireReady();
        auto model = Model(command.modelId);
        const uint64_t reservation = model->model->SessionReservationBytes();
        if (liveSessions >= limits.maxSessions || model->sessions >= model->sessionLimit ||
            (model->cacheLimit && reservation > model->cacheLimit / (model->sessions + 1)) ||
            (limits.maxSessionCacheBytes && reservation > limits.maxSessionCacheBytes - sessionCacheBytes))
            throw CommandFailure(wire::ErrorCode::ResourceLimit, "Session count or cache reservation limit reached");
        if (nextSessionId == (std::numeric_limits<uint64_t>::max)())
            throw CommandFailure(wire::ErrorCode::ResourceLimit, "Session IDs exhausted; restart Worker");
        auto record = std::make_shared<SessionRecord>();
        record->id = nextSessionId++;
        record->model = model;
        auto self = shared_from_this();
        ++model->sessions;
        ++liveSessions;
        sessionCacheBytes += reservation;
        try {
            Lifecycle([self, input, record, reservation] {
                std::exception_ptr failure;
                try {
                    self->RequireReady();
                    record->session = record->model->model->CreateSession();
                    self->CheckLifecycleFault();
                    self->RequireReady();
                    if (!record->session || record->session->CacheBytes() != reservation)
                        throw core::Error(core::ErrorCode::RuntimeFault, "Session cache reservation did not match allocation");
                } catch (...) {
                    failure = std::current_exception();
                    record->session.reset();
                    self->CheckLifecycleFault();
                }
                self->Post([self, input, record, reservation, failure] {
                    if (failure) {
                        --self->liveSessions;
                        --record->model->sessions;
                        self->sessionCacheBytes -= reservation;
                        self->RejectException(input, failure);
                        return;
                    }
                    self->sessions.emplace(record->id, record);
                    if (self->stopping.load() || self->domainFaulted.load()) {
                        record->closed = true;
                        self->Reject(input, self->domainFaulted.load() ? wire::ErrorCode::RuntimeFault : wire::ErrorCode::SchedulerStopped,
                                     "Worker stopped before Session publication");
                        return;
                    }
                    auto response = self->ResponseFor(*input);
                    response.session = {record->id, record->model->id,
                        static_cast<uint32_t>(record->session->Capacity()), record->session->CacheBytes(), false, 0};
                    self->SendResponse(std::move(response), input);
                });
            });
        } catch (...) {
            --model->sessions; --liveSessions; sessionCacheBytes -= reservation; throw;
        }
    }

    void ChangeContext(const std::shared_ptr<Input>& input, const wire::Command& command) {
        RequireReady();
        auto record = Session(command.sessionId);
        if (record->contextPending || record->session->Busy())
            throw CommandFailure(wire::ErrorCode::Busy, "Session already has an active context operation or request");
        auto self = shared_from_this();
        record->contextPending = true;
        try {
            Lifecycle([self, input, record, method = command.method, length = command.contextLength] {
                std::exception_ptr failure;
                uint32_t restored = 0;
                try {
                    self->RequireReady();
                    if (method == wire::Method::SessionReset) record->session->Reset();
                    else restored = static_cast<uint32_t>(record->session->Truncate(static_cast<int>(length)));
                    self->CheckLifecycleFault();
                    self->RequireReady();
                } catch (...) {
                    failure = std::current_exception();
                    bool runtimeCause = false;
                    try { std::rethrow_exception(failure); }
                    catch (const core::Error& error) {
                        runtimeCause = error.Code() == core::ErrorCode::RuntimeFault;
                        // Stop subsequent lifecycle jobs before waiting for
                        // this operation's control-thread response publication.
                        if (runtimeCause) self->PublishFault(error.what());
                    } catch (...) {}
                    // Keep completion bookkeeping even if the final drain
                    // discovers a device fault after a local exception.
                    try {
                        self->CheckLifecycleFault();
                        if (!runtimeCause && self->domainFaulted.load()) self->RequireReady();
                    }
                    catch (...) { failure = std::current_exception(); }
                }
                self->Post([self, input, record, restored, failure] {
                    record->contextPending = false;
                    if (failure) { self->RejectException(input, failure); return; }
                    if (self->stopping.load() || self->domainFaulted.load()) {
                        self->Reject(input, self->domainFaulted.load() ? wire::ErrorCode::RuntimeFault : wire::ErrorCode::SchedulerStopped,
                                     "Worker stopped before context operation publication");
                        return;
                    }
                    auto response = self->ResponseFor(*input);
                    response.session = {record->id, record->model->id,
                        static_cast<uint32_t>(record->session->Capacity()), record->session->CacheBytes(), false, restored};
                    self->SendResponse(std::move(response), input);
                });
            });
        } catch (...) { record->contextPending = false; throw; }
    }

    void Submit(const std::shared_ptr<Input>& input, wire::Command command) {
        RequireReady();
        auto session = Session(command.sessionId);
        if (session->contextPending || session->session->Busy())
            throw CommandFailure(wire::ErrorCode::Busy, "Session already has an active context operation or request");
        const uint64_t tokens = command.promptIds.size() + uint64_t(command.generation.maxNewTokens);
        const uint64_t bytes = uint64_t(limits.tokenWindow) * limits.maxTokenFrameBytes +
            limits.maxTerminalFrameBytes + limits.maxFinalFrameBytes + limits.maxTokenFrameBytes;
        if (ipcOutstanding >= limits.maxOutstandingRequests || requests.size() >= limits.maxRetainedRequestHandles ||
            tokens > limits.maxOutstandingTokens - reservedTokens || bytes > limits.maxTransportBytes - transportBase - transportBytes)
            throw CommandFailure(wire::ErrorCode::ResourceLimit, "Inference transport reservation limit reached");
        auto record = std::make_shared<RequestRecord>();
        record->summary.message.reserve(wire::kMaxErrorBytes);
        record->identity = {input->rpcId, session->model->id, session->id};
        record->executingSession = session;
        record->tokenReservation = tokens;
        record->byteReservation = bytes;
        auto accepted = ResponseFor(*input);
        accepted.kind = wire::ResponseKind::Accepted;
        accepted.phase = wire::SubmissionPhase::Accepted;
        accepted.identity = record->identity;
        accepted.tokenWindow = limits.tokenWindow;
        // Allocate the registry node and output payload before Scheduler accepts.
        ipc::Frame acceptedFrame;
        wire::ValidationError error;
        if (!wire::make_response(accepted, acceptedFrame, error, limits)) throw std::runtime_error(error.message);
        auto options = CoreOptions(command);
        std::vector<int> prompt(command.promptIds.begin(), command.promptIds.end());
        requests.emplace(input->rpcId, record);
        ++ipcOutstanding;
        reservedTokens += tokens;
        transportBytes += bytes;
        try { record->scheduled = scheduler->Submit(session->session, std::move(prompt), std::move(options)); }
        catch (...) {
            --ipcOutstanding; reservedTokens -= tokens; transportBytes -= bytes;
            requests.erase(input->rpcId);
            throw;
        }
        try {
            responses.Push({std::move(acceptedFrame), [reservation = input]() mutable { reservation.reset(); }});
        } catch (...) {
            // Scheduler acceptance cannot be undone into a rejected response.
            // If the reserved sender itself fails, fail the connection instead.
            Fail("Could not publish an accepted inference request");
        }
    }

    void Drain(const std::shared_ptr<RequestRecord>& record) {
        if (record->drained || !record->terminalQueued || record->acknowledged != record->summary.eventCount ||
            record->eventPending.load() || record->finalPending.load()) return;
        record->drained = true;
        --ipcOutstanding;
        reservedTokens -= record->tokenReservation;
        transportBytes -= record->byteReservation;
        auto scheduled = std::move(record->scheduled);
        Lifecycle([scheduled = std::move(scheduled)]() mutable { scheduled.reset(); });
    }

    void Acknowledge(const std::shared_ptr<Input>& input, const wire::Command& command) {
        auto record = Request(command.inferenceId, true);
        if (command.consumedSequence > record->lastSent)
            throw CommandFailure(wire::ErrorCode::InvalidArgument, "ACK exceeds the submitted event prefix");
        if (command.consumedSequence > record->acknowledged) record->acknowledged = command.consumedSequence;
        Drain(record);
        auto response = ResponseFor(*input);
        response.acknowledgedSequence = record->acknowledged;
        SendResponse(std::move(response), input);
    }

    void Release(const std::shared_ptr<Input>& input, const wire::Command& command) {
        switch (command.objectType) {
            case wire::ObjectType::Model: {
                const auto found = models.find(command.objectId);
                if (found == models.end()) throw CommandFailure(wire::ErrorCode::InvalidHandle, "Unknown model handle");
                found->second->closed = true;
                break;
            }
            case wire::ObjectType::Session: {
                const auto found = sessions.find(command.objectId);
                if (found == sessions.end()) throw CommandFailure(wire::ErrorCode::InvalidHandle, "Unknown session handle");
                found->second->closed = true;
                break;
            }
            case wire::ObjectType::Request: {
                auto request = Request(command.objectId, true);
                request->closing = true;
                if (request->scheduled) request->scheduled->Cancel();
                break;
            }
        }
        SendResponse(ResponseFor(*input), input);
    }

    void BeginShutdown(const std::shared_ptr<Input>& input, const wire::Command& command) {
        if (stopping.exchange(true)) throw CommandFailure(wire::ErrorCode::SchedulerStopped, "Shutdown is already in progress");
        shutdownCommand = input;
        shutdownDeadline = Clock::now() + std::chrono::milliseconds(command.graceMs);
        for (auto& [id, model] : models) model->closed = true;
        for (auto& [id, session] : sessions) session->closed = true;
        auto self = shared_from_this();
        Lifecycle([self] {
            if (self->scheduler) self->scheduler->Shutdown();
            self->Post([self] { self->schedulerStopped = true; });
        });
    }

    void Dispatch(const std::shared_ptr<Input>& input) {
        try {
            wire::Command command;
            wire::ValidationError error;
            if (!wire::parse_command(input->frame, command, error, limits)) throw CommandFailure(error.code, error.message);
            input->frame = {};
            if (command.epoch != epoch) throw CommandFailure(wire::ErrorCode::StaleEpoch, "Command belongs to another Worker epoch");
            if (!configured && command.method != wire::Method::WorkerConfigure &&
                command.method != wire::Method::WorkerQuery && command.method != wire::Method::WorkerShutdown)
                throw CommandFailure(wire::ErrorCode::InvalidArgument, "Worker has not been configured");
            switch (command.method) {
                case wire::Method::WorkerConfigure: Configure(input, command); break;
                case wire::Method::WorkerQuery: {
                    auto response = ResponseFor(*input); response.worker = QueryWorker();
                    SendResponse(std::move(response), input); break;
                }
                case wire::Method::WorkerMemoryQuery: QueryRuntimeMemory(input); break;
                case wire::Method::WorkerSetLogLevel: {
                    // Profiling's threshold is synchronized independently of
                    // GPU execution. No lifecycle/GPU work or caller callback.
                    aila::set_log_level(static_cast<aila::LogLevel>(command.logLevel));
                    minimumLogLevel.store(command.logLevel);
                    auto response = ResponseFor(*input);
                    response.logLevel = command.logLevel;
                    SendResponse(std::move(response), input); break;
                }
                case wire::Method::ModelLoad: LoadModel(input, command); break;
                case wire::Method::SessionCreate: CreateSession(input, command); break;
                case wire::Method::SessionQuery: {
                    auto session = Session(command.sessionId);
                    auto response = ResponseFor(*input);
                    response.session = {session->id, session->model->id,
                        static_cast<uint32_t>(session->session->Capacity()), session->session->CacheBytes(),
                        session->contextPending || session->session->Busy(), std::nullopt};
                    if (!response.session.busy) response.session.contextLength = session->session->ContextLength();
                    SendResponse(std::move(response), input); break;
                }
                case wire::Method::SessionReset:
                case wire::Method::SessionTruncate: ChangeContext(input, command); break;
                case wire::Method::RequestSubmit: Submit(input, std::move(command)); break;
                case wire::Method::RequestCancel: {
                    auto record = Request(command.inferenceId);
                    auto response = ResponseFor(*input);
                    response.summary = record->finalQueued ? record->summary : MakeSummary(record->scheduled->Snapshot());
                    response.cancelRequested = !wire::is_terminal(response.summary.status);
                    if (response.cancelRequested) record->scheduled->Cancel();
                    SendResponse(std::move(response), input); break;
                }
                case wire::Method::RequestQuery: {
                    auto record = Request(command.inferenceId);
                    auto response = ResponseFor(*input);
                    if (record->scheduled) {
                        const auto snapshot = record->scheduled->Snapshot();
                        response.request.summary = MakeSummary(snapshot);
                        response.request.bufferedTokens = snapshot.bufferedTokens;
                    } else response.request.summary = record->summary;
                    response.request.lastSentSequence = record->lastSent;
                    response.request.lastAcknowledgedSequence = record->acknowledged;
                    response.request.completionReady = wire::is_terminal(response.request.summary.status);
                    SendResponse(std::move(response), input); break;
                }
                case wire::Method::RequestAck: Acknowledge(input, command); break;
                case wire::Method::HandleRelease: Release(input, command); break;
                case wire::Method::WorkerShutdown: BeginShutdown(input, command); break;
            }
        } catch (...) { RejectException(input, std::current_exception()); }
    }

    void SendEvent(wire::Event event, std::function<void()> afterWrite = {}) {
        if (lastEventOrdinal == (std::numeric_limits<uint64_t>::max)()) throw std::runtime_error("Event ordinals exhausted");
        event.epoch = epoch;
        event.eventOrdinal = ++lastEventOrdinal;
        ipc::Frame frame;
        wire::ValidationError error;
        if (!wire::make_event(event, frame, error, limits)) throw std::runtime_error(error.message);
        ++pendingEvents;
        auto weak = weak_from_this();
        events.Push({std::move(frame), [weak, afterWrite = std::move(afterWrite)] {
            if (afterWrite) afterWrite();
            if (auto self = weak.lock()) { --self->pendingEvents; self->changed.notify_one(); }
        }});
    }

    void FreezeFinal(const std::shared_ptr<RequestRecord>& record, const core::ScheduledRequestSnapshot& snapshot) {
        if (record->finalQueued || !core::IsTerminal(snapshot.status)) return;
        record->summary = MakeSummary(snapshot);
        wire::Response final;
        final.method = wire::Method::RequestSubmit;
        final.rpcId = record->identity.inferenceId;
        final.epoch = epoch;
        final.identity = record->identity;
        final.phase = wire::SubmissionPhase::Completed;
        final.kind = snapshot.status == core::RequestStatus::Failed ? wire::ResponseKind::Error : wire::ResponseKind::Result;
        final.summary = record->summary;
        record->finalPending.store(true);
        const std::weak_ptr<RequestRecord> weak = record;
        SendResponse(std::move(final), {}, [weak] { if (auto value = weak.lock()) value->finalPending.store(false); });
        record->finalQueued = true;
        // Registry still owns the parent. Its last GPU reference is moved to
        // the lifecycle reaper, never to a response/event writer.
        record->executingSession.reset();
    }

    void PumpRequests() {
        if (scheduler) {
            const auto snapshot = scheduler->Snapshot();
            if (snapshot.faulted && !domainFaulted.load()) PublishFault(snapshot.faultMessage);
        }
        if (domainFaulted.load() && !faultEventQueued) {
            wire::Event event;
            event.kind = wire::EventKind::DomainFault;
            event.errorCode = wire::ErrorCode::RuntimeFault;
            { std::lock_guard<std::mutex> lock(faultMutex); event.message = domainFault; }
            if (event.message.empty()) event.message = "Worker device domain has failed";
            SendEvent(std::move(event));
            faultEventQueued = true;
        }
        if (requests.empty()) return;
        auto first = requests.upper_bound(roundRobinAfter);
        if (first == requests.end()) first = requests.begin();
        auto iterator = first;
        const auto nextFirst = first->first;
        do {
            auto& record = iterator->second;
            Drain(record);
            if (record->scheduled && !record->drained) {
                auto snapshot = record->scheduled->Snapshot();
                FreezeFinal(record, snapshot);
                const uint64_t unacknowledged = record->sentTokens - std::min(record->acknowledged, record->sentTokens);
                const bool terminalNext = record->finalQueued && snapshot.bufferedTokens == 0 &&
                    record->sentTokens == record->summary.generatedTokens;
                if (!record->terminalQueued && !record->eventPending.load() &&
                    (unacknowledged < limits.tokenWindow || terminalNext)) {
                    // Each request has at most one staged/in-flight frame. Set
                    // the flag before Read so retries can never lose a token.
                    record->eventPending.store(true);
                    const auto read = record->scheduled->Read(0ms);
                    if (read.status == core::ReadStatus::Event) {
                        const auto& source = *read.event;
                        if (source.sequence != record->lastSent + 1) throw std::runtime_error("Scheduler event sequence mismatch");
                        wire::Event event;
                        event.identity = record->identity;
                        event.sequence = source.sequence;
                        if (source.kind == core::RequestEventKind::Token) {
                            event.kind = wire::EventKind::Token;
                            event.tokenId = source.tokenId;
                            event.summary.status = WireStatus(source.status);
                            event.summary.promptTokensProcessed = source.promptTokensProcessed;
                            event.summary.generatedTokens = source.generatedTokens;
                            ++record->sentTokens;
                        } else {
                            // A step can finish between Snapshot and Read.
                            // Refresh and freeze its summary before forwarding
                            // that terminal, preserving response-pipe ordering.
                            if (!record->finalQueued) FreezeFinal(record, record->scheduled->Snapshot());
                            if (!record->finalQueued) throw std::runtime_error("Core terminal arrived before its snapshot");
                            event.kind = wire::EventKind::Terminal;
                            event.summary = record->summary;
                            if (source.sequence != event.summary.eventCount || source.generatedTokens != event.summary.generatedTokens ||
                                source.promptTokensProcessed != event.summary.promptTokensProcessed ||
                                WireStatus(source.status) != event.summary.status || WireReason(source.endReason) != event.summary.endReason ||
                                (source.errorCode ? std::optional<wire::ErrorCode>(WireError(*source.errorCode)) :
                                    source.status == core::RequestStatus::Failed ? std::optional<wire::ErrorCode>(wire::ErrorCode::RequestFailed) : std::nullopt)
                                    != event.summary.errorCode || BoundedMessage(source.error) != event.summary.message)
                                throw std::runtime_error("Core terminal summary mismatch");
                            record->terminalQueued = true;
                        }
                        record->lastSent = source.sequence;
                        const std::weak_ptr<RequestRecord> weakRecord = record;
                        SendEvent(std::move(event), [weakRecord] {
                            if (auto value = weakRecord.lock()) value->eventPending.store(false);
                        });
                    } else {
                        record->eventPending.store(false);
                        if (read.status == core::ReadStatus::Closed) throw std::runtime_error("Core stream closed without terminal");
                    }
                }
            }
            if (++iterator == requests.end()) iterator = requests.begin();
        } while (iterator != first);
        roundRobinAfter = nextFirst;
    }

    void Reap() {
        for (auto iterator = requests.begin(); iterator != requests.end();) {
            const auto& record = iterator->second;
            if (record->drained && (record->closing || stopping.load()) && !record->eventPending.load())
                iterator = requests.erase(iterator);
            else ++iterator;
        }
        auto self = shared_from_this();
        for (auto iterator = sessions.begin(); iterator != sessions.end();) {
            auto& record = iterator->second;
            if (stopping.load()) record->closed = true;
            if (record->closed && record.use_count() == 1) {
                auto retired = std::move(record);
                iterator = sessions.erase(iterator);
                Lifecycle([self, retired = std::move(retired)]() mutable {
                    const auto reservation = retired->session->CacheBytes();
                    auto model = retired->model;
                    retired.reset();
                    self->CheckLifecycleFault();
                    self->Post([self, model = std::move(model), reservation] {
                        --self->liveSessions; --model->sessions; self->sessionCacheBytes -= reservation;
                    });
                });
            } else ++iterator;
        }
        for (auto iterator = models.begin(); iterator != models.end();) {
            auto& record = iterator->second;
            if (stopping.load()) record->closed = true;
            if (record->closed && record.use_count() == 1 && record->sessions == 0) {
                auto retired = std::move(record);
                iterator = models.erase(iterator);
                Lifecycle([self, retired = std::move(retired)]() mutable {
                    retired.reset();
                    self->CheckLifecycleFault();
                    self->Post([self] { --self->liveModels; });
                });
            } else ++iterator;
        }
    }

    void Log(int level, std::string_view message) noexcept {
        if (level < 0 || level > 3 || static_cast<uint32_t>(level) < minimumLogLevel.load() || !logsAccepting.load()) return;
        try {
            auto bounded = BoundedMessage(message);
            const uint64_t bytes = bounded.size() * 6 + 512;
            std::unique_lock<std::mutex> lock(logMutex, std::try_to_lock);
            const auto limit = logLimit.load();
            if (!lock.owns_lock() || !logsAccepting.load() || logBytes > limit || bytes > limit - logBytes) {
                ++droppedLogs; return;
            }
            logs.push_back({level, std::move(bounded), bytes});
            logBytes += bytes;
            changed.notify_one();
        } catch (...) { ++droppedLogs; }
    }

    void PumpLog() {
        std::optional<LogMessage> message;
        {
            std::lock_guard<std::mutex> lock(logMutex);
            if (!logs.empty()) { message = std::move(logs.front()); logs.pop_front(); }
        }
        if (!message) return;
        wire::Event event;
        event.kind = wire::EventKind::Log;
        event.logLevel = message->level;
        event.message = std::move(message->message);
        const auto weak = weak_from_this();
        SendEvent(std::move(event), [weak, bytes = message->bytes] {
            if (auto self = weak.lock()) { std::lock_guard<std::mutex> lock(self->logMutex); self->logBytes -= bytes; }
        });
    }

    void FinishShutdown() {
        if (!shutdownCommand || shutdownQueued) return;
        if (Clock::now() >= shutdownDeadline) { Fail("Worker shutdown grace period expired"); return; }
        if (!schedulerStopped || ipcOutstanding || !requests.empty() || liveModels || liveSessions || lifecyclePending) return;
        logsAccepting.store(false);
        {
            std::lock_guard<std::mutex> lock(logMutex);
            if (logBytes != 0 || !logs.empty()) return;
        }
        if (pendingEvents.load() != 0) return;
        {
            std::lock_guard<std::mutex> lock(inputMutex);
            if (pendingCommands != 1 || !inputs.empty() || !completions.empty()) return;
            ingressClosed = true;
        }
        auto response = ResponseFor(*shutdownCommand);
        response.worker.domainStatus = wire::DomainStatus::Closed;
        response.lastEventOrdinal = lastEventOrdinal;
        auto weak = weak_from_this();
        SendResponse(std::move(response), std::move(shutdownCommand), [weak] {
            if (auto self = weak.lock()) { self->closed.store(true); self->changed.notify_all(); }
        });
        shutdownQueued = true;
    }

    void ControlLoop() noexcept {
        try {
            while (!abort.load() && !closed.load()) {
                std::deque<std::function<void()>> ready;
                {
                    std::lock_guard<std::mutex> lock(inputMutex);
                    ready.swap(completions);
                }
                for (auto& operation : ready) operation();
                ready.clear();
                for (unsigned count = 0; count < 8; ++count) {
                    std::shared_ptr<Input> input;
                    {
                        std::lock_guard<std::mutex> lock(inputMutex);
                        if (inputs.empty()) break;
                        input = std::move(inputs.front()); inputs.pop_front();
                    }
                    Dispatch(input);
                }
                PumpRequests();
                PumpLog();
                Reap();
                FinishShutdown();
                std::unique_lock<std::mutex> lock(inputMutex);
                changed.wait_for(lock, 1ms, [&] { return abort.load() || closed.load() || !inputs.empty() || !completions.empty(); });
            }
        } catch (const std::exception& exception) { Fail(exception.what()); }
        catch (...) { Fail("Unknown Worker control failure"); }
        responses.Stop();
        events.Stop();
    }

    void Stop() noexcept {
        std::lock_guard<std::mutex> stopLock(stopMutex);
        if (joined) return;
        abort.store(true);
        stopping.store(true);
        logsAccepting.store(false);
        changed.notify_all();
        if (controlThread.joinable()) controlThread.join();
        if (scheduler) scheduler->Shutdown();
        // All remaining owners, including completion closures, are retired on
        // the lifecycle executor after the Scheduler has drained device work.
        {
            std::lock_guard<std::mutex> lock(lifecycleMutex);
            lifecycleJobs.push_back([this] {
                std::deque<std::function<void()>> retiredCompletions;
                std::deque<std::shared_ptr<Input>> retiredInputs;
                {
                    std::lock_guard<std::mutex> lock(inputMutex);
                    retiredCompletions.swap(completions);
                    retiredInputs.swap(inputs);
                }
                retiredCompletions.clear();
                requests.clear(); sessions.clear(); models.clear();
                factory.reset();
            });
            lifecycleStopping = true;
        }
        lifecycleChanged.notify_one();
        if (lifecycleThread.joinable()) lifecycleThread.join();
        shutdownCommand.reset();
        responses.Stop(); events.Stop();
        joined = true;
    }

    const std::string epoch;
    std::unique_ptr<InferenceModelFactory> factory;
    std::unique_ptr<core::Scheduler> scheduler;
    wire::Limits limits;
    bool configured = false;
    std::atomic<bool> stopping{false}, domainFaulted{false}, abort{false}, closed{false};
    mutable std::mutex failureMutex, faultMutex;
    std::string failure, domainFault;
    bool faultEventQueued = false;
    std::thread controlThread, lifecycleThread;
    std::mutex stopMutex;
    bool joined = false;

    std::mutex inputMutex;
    std::condition_variable changed;
    std::deque<std::shared_ptr<Input>> inputs;
    std::deque<std::function<void()>> completions;
    uint64_t lastRpcId = 0, pendingCommands = 0, pendingNormalCommands = 0, ingressBytes = 0;
    bool ingressClosed = false;
    OutputQueue responses, events;
    std::atomic<uint64_t> pendingEvents{0};

    std::mutex lifecycleMutex;
    std::condition_variable lifecycleChanged;
    std::deque<std::function<void()>> lifecycleJobs;
    size_t lifecyclePending = 0;
    bool lifecycleStopping = false;
    std::map<uint64_t, std::shared_ptr<ModelRecord>> models;
    std::map<uint64_t, std::shared_ptr<SessionRecord>> sessions;
    std::map<uint64_t, std::shared_ptr<RequestRecord>> requests;
    uint64_t nextModelId = 1, nextSessionId = 1, roundRobinAfter = 0;
    size_t liveModels = 0, liveSessions = 0;
    static constexpr uint64_t RetainedHandleBytes = 2048;
    uint64_t sessionCacheBytes = 0, ipcOutstanding = 0, reservedTokens = 0, transportBytes = 0, transportBase = 0;
    uint64_t lastEventOrdinal = 0;

    std::mutex logMutex;
    std::deque<LogMessage> logs;
    std::atomic<bool> logsAccepting{true};
    std::atomic<uint32_t> minimumLogLevel{0};
    std::atomic<uint64_t> logLimit{64 * 1024}, droppedLogs{0};
    uint64_t logBytes = 0;
    std::shared_ptr<Input> shutdownCommand;
    Clock::time_point shutdownDeadline;
    bool schedulerStopped = false, shutdownQueued = false;
};

InferenceWorker::InferenceWorker(std::string epoch, std::unique_ptr<InferenceModelFactory> factory)
    : state(std::make_shared<State>(std::move(epoch), std::move(factory))) { state->Start(); }
InferenceWorker::~InferenceWorker() { Stop(); }
bool InferenceWorker::Enqueue(ipc::Frame command, std::string& error) { return state->Enqueue(std::move(command), error); }
bool InferenceWorker::ReadResponse(InferenceWorkerOutput& output, std::chrono::milliseconds timeout) { return state->responses.Read(output, timeout); }
bool InferenceWorker::ReadEvent(InferenceWorkerOutput& output, std::chrono::milliseconds timeout) { return state->events.Read(output, timeout); }
void InferenceWorker::Log(int level, std::string_view message) noexcept { state->Log(level, message); }
void InferenceWorker::TransportFailed(std::string_view message) noexcept { state->Fail(message); }
bool InferenceWorker::Closed() const noexcept { return state->closed.load(); }
bool InferenceWorker::Failed() const noexcept { return state->abort.load() && !state->closed.load(); }
std::string InferenceWorker::FailureMessage() const { std::lock_guard<std::mutex> lock(state->failureMutex); return state->failure; }
ipc::v2::Limits InferenceWorker::Limits() const { std::lock_guard<std::mutex> lock(state->inputMutex); return state->limits; }
void InferenceWorker::Stop() noexcept { if (state) state->Stop(); }

} // namespace aila::worker
