#include "worker/InferenceWorker.hpp"
#include "profile/Profiling.hpp"
#include "ipc/InferenceProtocol.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace wire = aila::ipc::v2;
namespace core = aila::inference;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr char Epoch[] = "0123456789abcdef0123456789abcdef";

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool blocked = false;
    bool entered = false;
    void Block() { std::lock_guard<std::mutex> lock(mutex); blocked = true; entered = false; }
    void Open() { { std::lock_guard<std::mutex> lock(mutex); blocked = false; } changed.notify_all(); }
    void Enter() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true; changed.notify_all();
        if (!changed.wait_for(lock, 5s, [&] { return !blocked; })) throw std::runtime_error("CPU step gate timed out");
    }
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        Expect(changed.wait_for(lock, 5s, [&] { return entered; }), "CPU step gate was not entered");
    }
};

struct Counters {
    Gate step;
    Gate load;
    Gate context;
    Gate memory;
    std::atomic<bool> memoryInitialized{false}, memoryBusy{false};
    std::atomic<int> models{0}, sessions{0};
    std::atomic<int> qwen3Loads{0}, qwen35Loads{0}, qwen35Bf16Loads{0};
    std::atomic<bool> failSession{false};
    std::atomic<bool> faultSession{false};
    std::atomic<bool> faultContext{false};
    std::atomic<bool> failContext{false};
    std::atomic<bool> checkpointContext{false};
    std::atomic<int> contextCalls{0};
};

class FakeBackend final : public core::SessionBackend {
public:
    FakeBackend(std::shared_ptr<Counters> counters, int capacity, std::shared_ptr<void> model)
        : counters(std::move(counters)), capacity(capacity), model(std::move(model)) { ++this->counters->sessions; }
    ~FakeBackend() override { --counters->sessions; }
    int ContextLength() const override { return length; }
    int Capacity() const override { return capacity; }
    int VocabSize() const override { return 256; }
    size_t CacheBytes() const override { return static_cast<size_t>(capacity) * 16; }
    void Reset() override { ContextOperation(); length = 0; }
    int Truncate(int value) override {
        ContextOperation();
        length = counters->checkpointContext.load() ? value / 4 * 4 : value;
        return length;
    }
    core::ExecutionResult Forward(const std::vector<int>& ids, const core::SamplingInput* sampling, bool capture) override {
        counters->step.Enter();
        if (ids.front() == 254) throw core::Error(core::ErrorCode::RuntimeFault, "injected device failure");
        if (ids.front() == 253) throw std::runtime_error("injected request-only failure");
        length += static_cast<int>(ids.size());
        core::ExecutionResult result;
        if (sampling) result.tokenId = (ids.back() + length + 1) % 200;
        if (capture) result.logits.resize(256, 0.0f);
        return result;
    }
private:
    void ContextOperation() {
        ++counters->contextCalls;
        counters->context.Enter();
        if (counters->faultContext.exchange(false))
            throw core::Error(core::ErrorCode::RuntimeFault, "injected context device failure");
        if (counters->failContext.exchange(false)) throw std::invalid_argument("injected context-only failure");
    }
    std::shared_ptr<Counters> counters;
    int capacity;
    int length = 0;
    std::shared_ptr<void> model;
};

class FakeModel final : public aila::worker::InferenceModel, public std::enable_shared_from_this<FakeModel> {
public:
    FakeModel(std::shared_ptr<Counters> counters, int capacity,
              std::optional<wire::AllocationLimits> limits)
        : counters(std::move(counters)), capacity(capacity) {
        ++this->counters->models;
        if (limits) memory = wire::ModelMemoryInfo{*limits, 0, 0};
    }
    ~FakeModel() override { --counters->models; }
    std::shared_ptr<core::SessionState> CreateSession() override {
        if (counters->faultSession.exchange(false)) throw core::Error(core::ErrorCode::RuntimeFault, "injected Session allocation device failure");
        if (counters->failSession.exchange(false)) throw std::bad_alloc();
        return std::make_shared<core::SessionState>(std::make_unique<FakeBackend>(counters, capacity, shared_from_this()));
    }
    int VocabSize() const noexcept override { return 256; }
    int MaxSequenceLength() const noexcept override { return capacity; }
    size_t SessionReservationBytes() const noexcept override { return static_cast<size_t>(capacity) * 16; }
    std::optional<wire::ModelMemoryInfo> MemoryInfo() const override { return memory; }
private:
    std::shared_ptr<Counters> counters;
    int capacity;
    std::optional<wire::ModelMemoryInfo> memory;
};

class FakeFactory final : public aila::worker::InferenceModelFactory {
public:
    explicit FakeFactory(std::shared_ptr<Counters> counters) : counters(std::move(counters)) {}
    wire::RuntimeMemoryInfo QueryRuntimeMemory() const override {
        if (!counters->memoryInitialized.load()) return {};
        counters->memory.Enter();
        if (counters->memoryBusy.load()) return {wire::RuntimeMemoryState::Busy, std::nullopt};
        const auto count = static_cast<uint64_t>(counters->models.load() + counters->sessions.load());
        return {wire::RuntimeMemoryState::Ready, wire::RuntimeMemoryUsage{count * 1024, 8192, 0, 128, count, 0, 8192, 64}};
    }
    std::shared_ptr<aila::worker::InferenceModel> LoadQwen3(const std::string& directory,
                                                        aila::worker::InferenceModelOptions options) override {
        ++counters->qwen3Loads;
        return Load(directory, options);
    }
    std::shared_ptr<aila::worker::InferenceModel> LoadQwen35(const std::string& directory,
                                                         aila::worker::InferenceModelOptions options) override {
        ++counters->qwen35Loads;
        return Load(directory, options);
    }
    std::shared_ptr<aila::worker::InferenceModel> LoadQwen35Bf16(const std::string& directory,
                                                             aila::worker::InferenceModelOptions options) override {
        ++counters->qwen35Bf16Loads;
        return Load(directory, options);
    }
private:
    std::shared_ptr<aila::worker::InferenceModel> Load(const std::string& directory,
                                                    aila::worker::InferenceModelOptions options) {
        counters->load.Enter();
        if (directory == "bad-model") throw std::invalid_argument("injected file/configuration failure");
        counters->memoryInitialized.store(true);
        return std::make_shared<FakeModel>(counters, options.maxSequenceLength, options.allocationLimits);
    }
    std::shared_ptr<Counters> counters;
};

class Harness {
public:
    explicit Harness(std::string backend = "qwen3_nf4")
        : backend(std::move(backend)), counters(std::make_shared<Counters>()),
          worker(Epoch, std::make_unique<FakeFactory>(counters)) {}
    ~Harness() { counters->step.Open(); counters->load.Open(); counters->context.Open(); counters->memory.Open(); worker.Stop(); }
    uint64_t Send(wire::Command command) {
        command.rpcId = nextId + 1;
        if (command.epoch.empty()) command.epoch = Epoch;
        aila::ipc::Frame frame;
        wire::ValidationError error;
        Expect(wire::make_command(command, frame, error, limits), "make_command: " + error.message);
        return SendFrame(std::move(frame));
    }
    uint64_t SendFrame(aila::ipc::Frame frame) {
        frame.header.request_id = ++nextId;
        std::string transportError;
        Expect(worker.Enqueue(std::move(frame), transportError), "Enqueue: " + transportError);
        return nextId;
    }
    void Pump(bool readEvents = true) {
        aila::worker::InferenceWorkerOutput output;
        if (worker.ReadResponse(output, 1ms)) {
            wire::Response response;
            wire::ValidationError error;
            Expect(wire::parse_response(output.frame, response, error, limits), "parse_response: " + error.message + " payload=" + output.frame.header.payload_json);
            if (output.written) output.written();
            responses[response.rpcId].push_back(std::move(response));
        }
        if (readEvents && worker.ReadEvent(output, 0ms)) {
            wire::Event event;
            wire::ValidationError error;
            Expect(wire::parse_event(output.frame, event, error, limits), "parse_event: " + error.message);
            Expect(event.eventOrdinal == ++lastOrdinal, "global event ordinal gap");
            if (output.written) output.written();
            if (event.kind == wire::EventKind::DomainFault) { ++domainFaults; }
            else if (event.kind == wire::EventKind::Log) logs.push_back(std::move(event));
            else {
                auto& sequence = events[event.identity.inferenceId];
                Expect(event.sequence == sequence.size() + 1, "per-request event sequence gap");
                sequence.push_back(std::move(event));
            }
        }
        Expect(!worker.Failed(), "Worker failed: " + worker.FailureMessage());
    }
    wire::Response Wait(uint64_t id, bool readEvents = true) {
        const auto deadline = Clock::now() + 5s;
        while (Clock::now() < deadline) {
            auto found = responses.find(id);
            if (found != responses.end() && !found->second.empty()) {
                auto response = std::move(found->second.front()); found->second.pop_front(); return response;
            }
            Pump(readEvents);
        }
        throw std::runtime_error("Response timed out for RPC " + std::to_string(id));
    }
    wire::Response Call(wire::Command command) { return Wait(Send(std::move(command))); }
    void Configure(wire::Limits selected = {}, uint32_t maxDecodeBatchSize = 1) {
        wire::Command command;
        command.method = wire::Method::WorkerConfigure;
        command.limits = selected;
        command.maxDecodeBatchSize = maxDecodeBatchSize;
        command.requiredCapabilities = wire::default_capabilities();
        if (maxDecodeBatchSize > 1) command.requiredCapabilities.emplace_back("continuous_batching_v1");
        if (backend != "qwen3_nf4") command.requiredCapabilities.emplace_back(wire::model_capability(backend));
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result, "configure failed: " + response.message);
        Expect(response.capabilities == wire::supported_capabilities(), "configure omitted a supported backend");
        Expect(response.maxDecodeBatchSize == maxDecodeBatchSize, "configure changed the requested decode batch bound");
        limits = selected;
    }
    uint64_t Load(unsigned sessions = 4) {
        wire::Command command;
        command.method = wire::Method::ModelLoad;
        command.backend = backend;
        command.modelDirectory = "fake-model";
        command.maxSequenceLength = 512;
        command.maxSessions = sessions;
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result, "model load failed: " + response.message);
        const std::vector<std::string> capabilities{
            std::string(wire::model_capability(backend)), "token_requests_v1", "session_lifecycle_v1"};
        Expect(response.model.capabilities == capabilities, "loaded model advertised a different backend");
        return response.model.modelId;
    }
    uint64_t Create(uint64_t model) {
        wire::Command command;
        command.method = wire::Method::SessionCreate;
        command.modelId = model;
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result, "Session create failed: " + response.message);
        return response.session.sessionId;
    }
    uint64_t Submit(uint64_t session, uint32_t count, int32_t token = 1) {
        wire::Command command;
        command.method = wire::Method::RequestSubmit;
        command.sessionId = session;
        command.promptIds = {token, 2, 3};
        command.prefillChunkSize = 16;
        command.generation.maxNewTokens = count;
        command.generation.doSample = false;
        const auto id = Send(command);
        const auto accepted = Wait(id);
        Expect(accepted.kind == wire::ResponseKind::Accepted, "submit was not accepted: " + accepted.message);
        return id;
    }
    void Ack(uint64_t id, uint64_t sequence) {
        wire::Command command;
        command.method = wire::Method::RequestAck;
        command.inferenceId = id;
        command.consumedSequence = sequence;
        const auto result = Call(command);
        Expect(result.kind == wire::ResponseKind::Result && result.acknowledgedSequence >= sequence, "ACK failed");
    }
    wire::Response Final(uint64_t id) {
        wire::Response final;
        bool finalSeen = false;
        uint64_t ack = 0;
        const auto deadline = Clock::now() + 5s;
        while (Clock::now() < deadline) {
            Pump();
            auto found = responses.find(id);
            if (!finalSeen && found != responses.end() && !found->second.empty()) {
                final = std::move(found->second.front()); found->second.pop_front(); finalSeen = true;
            }
            const auto& sequence = events[id];
            if (!sequence.empty()) {
                const auto& last = sequence.back();
                const bool terminal = last.kind == wire::EventKind::Terminal;
                const auto consumable = terminal ? last.sequence - (finalSeen ? 0 : 1) : last.sequence;
                if (consumable > ack) { Ack(id, consumable); ack = consumable; }
                if (terminal && finalSeen) {
                    wire::ValidationError error;
                    Expect(wire::match_terminal_final(last, final, error, limits), "terminal/final mismatch: " + error.message);
                    return final;
                }
            }
        }
        throw std::runtime_error("Request drain timed out");
    }
    void Release(wire::ObjectType type, uint64_t id) {
        wire::Command command;
        command.method = wire::Method::HandleRelease;
        command.objectType = type;
        command.objectId = id;
        Expect(Call(command).kind == wire::ResponseKind::Result, "release failed");
    }
    wire::WorkerInfo Query() {
        wire::Command command;
        command.method = wire::Method::WorkerQuery;
        return Call(command).worker;
    }
    void WaitOutstanding(uint64_t target) {
        const auto deadline = Clock::now() + 5s;
        do { if (Query().ipcOutstandingRequests == target) return; } while (Clock::now() < deadline);
        throw std::runtime_error("Transport reservations were not reclaimed");
    }
    void Shutdown() {
        wire::Command command;
        command.method = wire::Method::WorkerShutdown;
        command.graceMs = 5000;
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result && response.worker.domainStatus == wire::DomainStatus::Closed,
               "shutdown did not succeed");
        Expect(lastOrdinal >= response.lastEventOrdinal, "shutdown overtook event writer barrier");
        Expect(counters->models.load() == 0 && counters->sessions.load() == 0, "shutdown retained resources");
    }

    const std::string backend;
    wire::Limits limits;
    std::shared_ptr<Counters> counters;
    aila::worker::InferenceWorker worker;
    uint64_t nextId = 0, lastOrdinal = 0;
    unsigned domainFaults = 0;
    std::vector<wire::Event> logs;
    std::map<uint64_t, std::deque<wire::Response>> responses;
    std::map<uint64_t, std::vector<wire::Event>> events;
};

void TestConcurrentControlAndParentLifetime(const std::string& backend, uint32_t batchSize = 1) {
    Harness h(backend);
    h.Configure({}, batchSize);
    const auto model = h.Load();
    const auto a = h.Create(model), b = h.Create(model);
    h.counters->step.Block();
    const auto requestA = h.Submit(a, 8);
    h.counters->step.Wait();
    const auto requestB = h.Submit(b, 4);
    wire::Command duplicate;
    duplicate.method = wire::Method::RequestSubmit;
    duplicate.sessionId = a;
    duplicate.promptIds = {1}; duplicate.generation.maxNewTokens = 3;
    Expect(h.Call(duplicate).errorCode == wire::ErrorCode::Busy, "same-Session submit was not Busy");
    wire::Command query;
    query.method = wire::Method::SessionQuery; query.sessionId = a;
    const auto busy = h.Call(query);
    Expect(busy.session.busy && !busy.session.contextLength, "busy Session exposed context length");
    h.Release(wire::ObjectType::Model, model);
    h.Release(wire::ObjectType::Session, a);
    h.Release(wire::ObjectType::Session, b);
    Expect(h.counters->models == 1 && h.counters->sessions == 2, "released parents destroyed live requests");
    wire::Command load;
    load.method = wire::Method::ModelLoad; load.modelDirectory = "fake-model";
    load.backend = backend;
    Expect(h.Call(load).errorCode == wire::ErrorCode::ResourceLimit, "parent release bypassed actual model count");
    wire::Command cancel;
    cancel.method = wire::Method::RequestCancel; cancel.inferenceId = requestA;
    Expect(h.Call(cancel).cancelRequested, "in-flight cancel did not respond before GPU boundary");
    h.counters->step.Open();
    Expect(h.Final(requestA).summary.status == wire::RequestStatus::Cancelled, "A was not cancelled");
    Expect(h.Final(requestB).summary.status == wire::RequestStatus::Completed, "A cancellation affected B");
    h.WaitOutstanding(0);
    h.Release(wire::ObjectType::Request, requestA);
    h.Release(wire::ObjectType::Request, requestB);
    h.Shutdown();
}

void TestBackpressureAndRetainedReservations(const std::string& backend, uint32_t batchSize = 1) {
    Harness h(backend);
    wire::Limits limits;
    limits.tokenWindow = 4;
    limits.maxBufferedTokensPerRequest = 8;
    limits.maxOutstandingRequests = 2;
    limits.maxActiveRequests = 2;
    h.Configure(limits, batchSize);
    const auto model = h.Load();
    const auto a = h.Create(model), b = h.Create(model);
    const auto requestA = h.Submit(a, 200);
    const auto requestB = h.Submit(b, 3);
    const auto finalB = h.Final(requestB);
    Expect(finalB.summary.status == wire::RequestStatus::Completed && finalB.summary.generatedTokens == 3,
           "slow A affected B");
    wire::Command query;
    query.method = wire::Method::RequestQuery; query.inferenceId = requestA;
    wire::Response state;
    const auto deadline = Clock::now() + 5s;
    do { state = h.Call(query); } while (!state.request.completionReady && Clock::now() < deadline);
    Expect(state.request.summary.endReason == wire::EndReason::OutputBackpressure, "long unread A was not locally backpressured");
    Expect(state.request.lastSentSequence <= limits.tokenWindow && state.request.bufferedTokens <= limits.maxBufferedTokensPerRequest,
           "backpressure exceeded bounded rings");
    Expect(h.Query().ipcOutstandingRequests == 1, "unconsumed terminal released transport quota");
    h.Release(wire::ObjectType::Request, requestA);
    const auto finalA = h.Final(requestA);
    Expect(finalA.summary.endReason == wire::EndReason::OutputBackpressure, "release changed original backpressure terminal");
    h.WaitOutstanding(0);
    h.Release(wire::ObjectType::Request, requestB);
    h.Shutdown();
}

void TestAdmissionRollbackAndStaleEpoch(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    wire::Command invalidLoad;
    invalidLoad.method = wire::Method::ModelLoad;
    invalidLoad.backend = backend;
    invalidLoad.modelDirectory = "bad-model";
    Expect(h.Call(invalidLoad).errorCode == wire::ErrorCode::InvalidArgument, "local load error was not preserved");
    Expect(h.Query().accepting, "local load error poisoned the domain");
    const auto model = h.Load(1);
    wire::Command create;
    create.method = wire::Method::SessionCreate; create.modelId = model;
    h.counters->failSession.store(true);
    Expect(h.Call(create).errorCode == wire::ErrorCode::ResourceLimit, "host Session allocation was not local");
    const auto session = h.Create(model);
    Expect(h.Call(create).errorCode == wire::ErrorCode::ResourceLimit, "Session count admission failed");
    wire::Command stale;
    stale.method = wire::Method::SessionQuery; stale.sessionId = session;
    stale.epoch = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    Expect(h.Call(stale).errorCode == wire::ErrorCode::StaleEpoch, "stale epoch was accepted");
    wire::Command invalid;
    invalid.method = wire::Method::RequestSubmit; invalid.sessionId = session;
    invalid.promptIds = {1000}; invalid.generation.maxNewTokens = 2;
    const auto rejection = h.Call(invalid);
    Expect(rejection.phase == wire::SubmissionPhase::Rejected && rejection.errorCode == wire::ErrorCode::InvalidArgument,
           "invalid token was accepted");
    Expect(h.Query().ipcOutstandingRequests == 0, "rejection leaked transport reservation");
    const auto request = h.Submit(session, 2);
    Expect(h.Final(request).summary.generatedTokens == 2, "post-rejection Session was not usable");
    wire::Command oldAck;
    oldAck.method = wire::Method::RequestAck; oldAck.inferenceId = request; oldAck.consumedSequence = 1;
    Expect(h.Call(oldAck).acknowledgedSequence == 3, "older ACK was not idempotent");
    oldAck.consumedSequence = 4;
    Expect(h.Call(oldAck).errorCode == wire::ErrorCode::InvalidArgument, "future ACK was accepted");
    h.Release(wire::ObjectType::Request, request);
    h.Shutdown();
}

void TestLifecycleFaultBroadcast(const std::string& backend, uint32_t batchSize = 1) {
    Harness h(backend);
    h.Configure({}, batchSize);
    const auto model = h.Load();
    const auto session = h.Create(model);
    h.counters->step.Block();
    const auto request = h.Submit(session, 10);
    h.counters->step.Wait();
    h.counters->faultSession.store(true);
    wire::Command create;
    create.method = wire::Method::SessionCreate; create.modelId = model;
    Expect(h.Call(create).errorCode == wire::ErrorCode::RuntimeFault, "lifecycle fault was not normalized");
    Expect(h.Query().faulted && !h.Query().accepting, "lifecycle fault did not stop the domain");
    h.counters->step.Open();
    Expect(h.Final(request).summary.endReason == wire::EndReason::RuntimeFault, "lifecycle failure did not terminate peer request");
    Expect(h.domainFaults == 1, "domain fault event was not exactly once");
    h.Release(wire::ObjectType::Request, request);
    h.Shutdown();
}

void TestLoadDoesNotBlockControl(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    h.counters->load.Block();
    wire::Command load;
    load.method = wire::Method::ModelLoad; load.modelDirectory = "fake-model";
    load.backend = backend;
    const auto loadId = h.Send(load);
    h.counters->load.Wait();
    Expect(h.Query().accepting, "worker.query waited on model loading");
    h.counters->load.Open();
    Expect(h.Wait(loadId).kind == wire::ResponseKind::Result, "gated model load failed");
    h.Shutdown();
}

void TestBackendValidationAndLoadRecovery(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    wire::Command load;
    load.method = wire::Method::ModelLoad;
    load.backend = backend;
    load.modelDirectory = "fake-model";
    load.maxSequenceLength = 512;
    load.rpcId = 1;
    load.epoch = Epoch;
    aila::ipc::Frame valid;
    wire::ValidationError validation;
    Expect(wire::make_command(load, valid, validation, h.limits), "valid load could not be encoded");
    for (const bool invalidBackend : {true, false}) {
        auto invalid = valid;
        const std::string from = invalidBackend ? "\"" + backend + "\"" : "\"bf16\"";
        const size_t position = invalid.header.payload_json.find(from);
        Expect(position != std::string::npos, "load fixture is missing a model option");
        invalid.header.payload_json.replace(position, from.size(), invalidBackend ? "\"future_nf4\"" : "\"fp8\"");
        const auto rejected = h.Wait(h.SendFrame(std::move(invalid)));
        Expect(rejected.kind == wire::ResponseKind::Error &&
               rejected.errorCode == (invalidBackend ? wire::ErrorCode::UnsupportedModel : wire::ErrorCode::InvalidArgument),
               "unsupported backend or KV type was not rejected locally");
        Expect(h.Query().accepting && h.counters->qwen3Loads == 0 && h.counters->qwen35Loads == 0 &&
               h.counters->qwen35Bf16Loads == 0,
               "invalid model options reached the factory or poisoned the Worker");
    }
    load.modelDirectory = "bad-model";
    Expect(h.Call(load).errorCode == wire::ErrorCode::InvalidArgument, "model configuration failure was not local");
    Expect(h.Query().accepting, "model configuration failure stopped the Worker");
    const auto model = h.Load();
    Expect(h.counters->qwen3Loads == (backend == "qwen3_nf4" ? 2 : 0) &&
           h.counters->qwen35Loads == (backend == "qwen35_nf4" ? 2 : 0) &&
           h.counters->qwen35Bf16Loads == (backend == "qwen35_bf16" ? 2 : 0),
           "model backend reached the wrong factory dispatch");
    const auto session = h.Create(model);
    const auto request = h.Submit(session, 3);
    Expect(h.Final(request).summary.generatedTokens == 3, "Worker was not usable after failed model loading");
    h.Release(wire::ObjectType::Request, request);
    h.Shutdown();
}

void TestSessionIsolationAndContinuation(const std::string& backend, uint32_t batchSize = 1) {
    Harness h(backend);
    h.Configure({}, batchSize);
    const auto model = h.Load();
    const auto reference = h.Create(model), a = h.Create(model), b = h.Create(model);
    const auto tokens = [&h](uint64_t request) {
        std::vector<int32_t> result;
        for (const auto& event : h.events.at(request)) {
            if (event.kind == wire::EventKind::Token) result.push_back(event.tokenId);
        }
        return result;
    };
    const auto baseline = h.Submit(reference, 4);
    Expect(h.Final(baseline).summary.status == wire::RequestStatus::Completed, "isolated reference failed");
    h.counters->step.Block();
    const auto requestA = h.Submit(a, 4);
    h.counters->step.Wait();
    const auto requestB = h.Submit(b, 4);
    h.counters->step.Open();
    Expect(h.Final(requestA).summary.status == wire::RequestStatus::Completed &&
           h.Final(requestB).summary.status == wire::RequestStatus::Completed, "interleaved Sessions failed");
    Expect(tokens(requestA) == tokens(baseline) && tokens(requestB) == tokens(baseline),
           "interleaved Sessions changed independent token results");

    const auto baselineContinuation = h.Submit(reference, 4);
    h.Final(baselineContinuation);
    const auto advanceA = h.Submit(a, 1);
    h.Final(advanceA);
    const auto continueB = h.Submit(b, 4);
    h.Final(continueB);
    Expect(tokens(continueB) == tokens(baselineContinuation), "advancing A changed B's continuation state");
    for (const auto request : {baseline, requestA, requestB, baselineContinuation, advanceA, continueB})
        h.Release(wire::ObjectType::Request, request);
    h.WaitOutstanding(0);
    h.Shutdown();
}

void TestThinkingBudgetForwarding(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    const auto model = h.Load();
    const auto session = h.Create(model);
    wire::Command command;
    command.method = wire::Method::RequestSubmit;
    command.sessionId = session;
    command.promptIds = {1, 2, 3};
    command.generation.maxNewTokens = 3;
    command.generation.doSample = false;
    command.thinkingBudget = wire::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}};
    const auto id = h.Send(command);
    Expect(h.Wait(id).kind == wire::ResponseKind::Accepted, "Budgeted request was not accepted");
    const auto result = h.Final(id);
    Expect(result.summary.endReason == wire::EndReason::Length && result.summary.generatedTokens == 2 &&
               h.events[id].size() == 3 && h.events[id][0].tokenId == 7 && h.events[id][1].tokenId == 12,
           "Worker did not forward the thinking budget or its early length reason");
    command.generation.maxNewTokens = 2;
    command.thinkingBudget->maxTokens = 0;
    const auto zeroId = h.Send(command);
    Expect(h.Wait(zeroId).kind == wire::ResponseKind::Accepted, "Zero-budget continuation failed");
    const auto zero = h.Final(zeroId);
    Expect(zero.summary.generatedTokens == 2 && h.events[zeroId][0].tokenId == 11 && h.events[zeroId][1].tokenId == 3,
           "Zero budget did not force tokenizer closing tokens through IPC");
    command.thinkingBudget->openTokenId = 256;
    const auto invalid = h.Call(command);
    Expect(invalid.kind == wire::ResponseKind::Error && invalid.errorCode == wire::ErrorCode::InvalidArgument,
           "Out-of-vocabulary thinking marker was accepted by the Worker");
    h.WaitOutstanding(0);
    h.Shutdown();
}

void TestAllocationLimitForwarding(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    wire::Command load;
    load.method = wire::Method::ModelLoad;
    load.backend = backend;
    load.modelDirectory = "fake-model";
    load.allocationLimits = wire::AllocationLimits{1024 * 1024, 65536, 8192};
    const auto response = h.Call(load);
    Expect(response.kind == wire::ResponseKind::Result && response.model.memory &&
           response.model.memory->limits == *load.allocationLimits,
           "Worker did not forward and acknowledge allocation limits through the lifecycle executor");
    const auto session = h.Create(response.model.modelId);
    h.Release(wire::ObjectType::Session, session);
    h.Release(wire::ObjectType::Model, response.model.modelId);
    h.Shutdown();
}

void TestContextReservationAndReleasedParents(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    const auto model = h.Load();
    const auto a = h.Create(model), b = h.Create(model), peer = h.Create(model);
    const auto initial = h.Submit(b, 5);
    Expect(h.Final(initial).summary.status == wire::RequestStatus::Completed, "Initial context did not populate");
    h.Release(wire::ObjectType::Request, initial);
    h.counters->checkpointContext = true;
    h.counters->context.Block();
    wire::Command change;
    change.method = wire::Method::SessionReset; change.sessionId = a;
    const auto reset = h.Send(change);
    h.counters->context.Wait();
    change.method = wire::Method::SessionTruncate; change.sessionId = b; change.contextLength = 5;
    const auto truncate = h.Send(change); // Queued behind A, before core Session::Acquire.
    wire::Command query;
    query.method = wire::Method::SessionQuery; query.sessionId = b;
    const auto pending = h.Call(query);
    Expect(pending.session.busy && !pending.session.contextLength,
           "Queued mutation did not reserve B, or query waited for the lifecycle executor");
    Expect(h.Call(change).errorCode == wire::ErrorCode::Busy, "Duplicate queued truncate was accepted");
    change.method = wire::Method::RequestSubmit; change.promptIds = {1}; change.generation.maxNewTokens = 1;
    Expect(h.Call(change).errorCode == wire::ErrorCode::Busy, "Submit overtook a queued context mutation");
    const auto other = h.Submit(peer, 2);
    Expect(h.Final(other).summary.status == wire::RequestStatus::Completed, "Context mutation blocked unrelated CPU peer");
    h.Release(wire::ObjectType::Request, other);
    h.Release(wire::ObjectType::Model, model);
    h.Release(wire::ObjectType::Session, a);
    h.Release(wire::ObjectType::Session, b);
    Expect(h.counters->models == 1 && h.counters->sessions == 3, "Release retired a pending context operation");
    h.counters->context.Open();
    const auto resetResult = h.Wait(reset), truncateResult = h.Wait(truncate);
    Expect(resetResult.kind == wire::ResponseKind::Result && resetResult.session.contextLength == 0,
           "Released-parent reset did not complete with zero context");
    Expect(truncateResult.kind == wire::ResponseKind::Result && truncateResult.session.contextLength == 4,
           "Truncate did not return the actual earlier checkpoint");
    h.Release(wire::ObjectType::Session, peer);
    h.Shutdown();
}

void TestContextBusyAndLocalRecovery(const std::string& backend) {
    Harness h(backend);
    h.Configure();
    const auto model = h.Load(), session = h.Create(model);
    h.counters->step.Block();
    const auto active = h.Submit(session, 3);
    h.counters->step.Wait();
    wire::Command command;
    command.sessionId = session;
    for (const auto method : {wire::Method::SessionReset, wire::Method::SessionTruncate}) {
        command.method = method;
        Expect(h.Call(command).errorCode == wire::ErrorCode::Busy, "Active Request permitted context mutation");
    }
    h.counters->step.Open();
    Expect(h.Final(active).summary.status == wire::RequestStatus::Completed, "Busy rejection damaged generation");
    h.Release(wire::ObjectType::Request, active);
    command.method = wire::Method::SessionTruncate; command.contextLength = 7;
    Expect(h.Call(command).errorCode == wire::ErrorCode::InvalidArgument, "Truncate grew the context");
    command.method = wire::Method::SessionQuery;
    Expect(h.Call(command).session.contextLength == 6 && h.Query().accepting, "Bad length changed context/domain");
    h.counters->failContext = true;
    command.method = wire::Method::SessionReset;
    Expect(h.Call(command).errorCode == wire::ErrorCode::InvalidArgument, "Local mutation failure was not local");
    command.method = wire::Method::SessionQuery;
    const auto after = h.Call(command);
    Expect(!after.session.busy && after.session.contextLength == 6, "Failed mutation leaked pending reservation");
    command.method = wire::Method::SessionReset;
    Expect(h.Call(command).session.contextLength == 0, "Mutation could not recover after a local error");
    const auto next = h.Submit(session, 2);
    Expect(h.Final(next).summary.status == wire::RequestStatus::Completed, "Session could not generate after reset");
    h.Release(wire::ObjectType::Request, next);
    h.Shutdown();
}

void TestContextShutdownAndDomainFault(const std::string& backend) {
    {
        Harness h(backend);
        h.Configure();
        const auto model = h.Load(), session = h.Create(model);
        h.counters->context.Block();
        wire::Command command;
        command.method = wire::Method::SessionReset; command.sessionId = session;
        const auto reset = h.Send(command);
        h.counters->context.Wait();
        command.method = wire::Method::WorkerShutdown; command.graceMs = 5000;
        const auto shutdown = h.Send(command);
        Expect(h.Query().stopping, "Shutdown waited on the context operation before stopping admission");
        h.counters->context.Open();
        Expect(h.Wait(reset).errorCode == wire::ErrorCode::SchedulerStopped,
               "Pending mutation was published after shutdown started");
        const auto result = h.Wait(shutdown);
        Expect(result.kind == wire::ResponseKind::Result && result.worker.domainStatus == wire::DomainStatus::Closed &&
               h.counters->models == 0 && h.counters->sessions == 0, "Shutdown leaked pending mutation resources");
    }
    {
        Harness h(backend);
        h.Configure();
        const auto model = h.Load(), a = h.Create(model), b = h.Create(model), queued = h.Create(model);
        h.counters->step.Block();
        const auto peer = h.Submit(b, 8);
        h.counters->step.Wait();
        h.counters->faultContext = true;
        h.counters->context.Block();
        wire::Command reset;
        reset.method = wire::Method::SessionReset; reset.sessionId = a;
        const auto faulting = h.Send(reset);
        h.counters->context.Wait();
        reset.sessionId = queued;
        const auto blocked = h.Send(reset);
        wire::Command query;
        query.method = wire::Method::SessionQuery; query.sessionId = queued;
        Expect(h.Call(query).session.busy, "Second context operation was not queued");
        h.counters->context.Open();
        Expect(h.Wait(faulting).errorCode == wire::ErrorCode::RuntimeFault, "Context device fault was not propagated");
        Expect(h.Wait(blocked).errorCode == wire::ErrorCode::RuntimeFault && h.counters->contextCalls == 1,
               "Queued context work executed after the lifecycle thread observed a device fault");
        Expect(h.Query().faulted && !h.Query().accepting, "Context fault did not stop admission");
        h.counters->step.Open();
        Expect(h.Final(peer).summary.endReason == wire::EndReason::RuntimeFault,
               "Context failure did not terminate the peer request");
        Expect(h.domainFaults == 1, "Context failure did not publish exactly one domain fault");
        h.Release(wire::ObjectType::Request, peer);
        h.Shutdown();
    }
}

void TestWorkerLogControl() {
    struct RestoreLevel {
        aila::LogLevel value = aila::get_log_level();
        ~RestoreLevel() { aila::set_log_level(value); }
    } restore;
    Harness h;
    h.Configure();
    wire::Command setting;
    setting.method = wire::Method::WorkerSetLogLevel;
    setting.logLevel = 2;
    Expect(h.Call(setting).logLevel == 2 && aila::get_log_level() == aila::LogLevel::Warning,
           "Worker did not apply the shared logging threshold");
    h.worker.Log(0, "suppressed debug");
    h.worker.Log(1, "suppressed info");
    Expect(h.Query().droppedLogs == 0 && h.logs.empty(), "Filtered logs consumed a queue or drop allowance");
    // Log enqueue is intentionally best-effort under lock contention. Several
    // short attempts exercise delivery without assuming each enqueue succeeds.
    for (unsigned i = 0; i < 8; ++i) {
        h.worker.Log(2, "warning"); h.Pump();
        h.worker.Log(3, "error"); h.Pump();
    }
    h.Query(); h.Pump();
    Expect(std::any_of(h.logs.begin(), h.logs.end(), [](const auto& log) { return log.logLevel == 2; }) &&
           std::any_of(h.logs.begin(), h.logs.end(), [](const auto& log) { return log.logLevel == 3; }) &&
           std::all_of(h.logs.begin(), h.logs.end(), [](const auto& log) { return log.logLevel >= 2; }),
           "Worker log threshold lost permitted levels or emitted suppressed levels");
    setting.logLevel = 0;
    Expect(h.Call(setting).logLevel == 0 && aila::get_log_level() == aila::LogLevel::Debug, "Debug level did not apply");
    for (unsigned i = 0; i < 8; ++i) { h.worker.Log(0, "debug enabled"); h.Pump(); }
    h.Shutdown();
    Expect(std::any_of(h.logs.begin(), h.logs.end(), [](const auto& log) { return log.logLevel == 0; }),
           "Lowering the threshold did not enable debug logging");
}

void TestRuntimeMemoryDoesNotBlockControl() {
    Harness h;
    h.Configure();
    wire::Command query;
    query.method = wire::Method::WorkerMemoryQuery;
    const auto empty = h.Call(query);
    Expect(empty.runtimeMemory.state == wire::RuntimeMemoryState::Uninitialized &&
           !empty.runtimeMemory.usage && h.counters->models.load() == 0,
           "Observation initialized a model or converted unavailable memory to zero");
    h.counters->load.Block();
    wire::Command loading;
    loading.method = wire::Method::ModelLoad;
    loading.modelDirectory = "fake-model";
    loading.maxSequenceLength = 512;
    const auto loadRpc = h.Send(loading);
    h.counters->load.Wait();
    const auto loadingMemory = h.Call(query);
    Expect(loadingMemory.runtimeMemory.state == wire::RuntimeMemoryState::Busy && !loadingMemory.runtimeMemory.usage,
           "Memory query waited behind blocked model initialization");
    h.counters->load.Open();
    const auto loaded = h.Wait(loadRpc);
    const auto model = loaded.model.modelId, session = h.Create(model);
    const auto ready = h.Call(query);
    Expect(ready.runtimeMemory.state == wire::RuntimeMemoryState::Ready &&
           ready.runtimeMemory.usage->deviceBytes == 2048, "Factory counters did not reach the response");
    h.counters->step.Block();
    const auto request = h.Submit(session, 8);
    h.counters->step.Wait();
    h.counters->memoryBusy.store(true);
    h.counters->memory.Block();
    const auto memoryRpc = h.Send(query);
    h.counters->memory.Wait();
    wire::Command cancel;
    cancel.method = wire::Method::RequestCancel;
    cancel.inferenceId = request;
    Expect(h.Call(cancel).cancelRequested && h.Query().accepting,
           "Memory query prevented cancellation or Worker control");
    h.counters->memory.Open();
    const auto busy = h.Wait(memoryRpc);
    Expect(busy.runtimeMemory.state == wire::RuntimeMemoryState::Busy && !busy.runtimeMemory.usage,
           "Busy factory snapshot was not preserved");
    h.counters->step.Open();
    Expect(h.Final(request).summary.status == wire::RequestStatus::Cancelled, "Cancelled request did not drain");
    h.Release(wire::ObjectType::Request, request);
    h.Shutdown();
}

void TestDecodeBatchingConfigurationRejectionAndRecovery() {
    Harness h;
    wire::Command configure;
    configure.method = wire::Method::WorkerConfigure;
    configure.rpcId = 1;
    configure.epoch = Epoch;
    configure.requiredCapabilities = wire::default_capabilities();
    for (const auto& item : {
            std::pair<const char*, wire::ErrorCode>{"0", wire::ErrorCode::InvalidArgument},
            {"5", wire::ErrorCode::InvalidArgument}, {"2", wire::ErrorCode::UnsupportedCapability}}) {
        aila::ipc::Frame frame;
        wire::ValidationError error;
        Expect(wire::make_command(configure, frame, error), "Could not encode ordinary configure");
        frame.header.payload_json.insert(1, std::string("\"max_decode_batch_size\":") + item.first + ",");
        const auto response = h.Wait(h.SendFrame(std::move(frame)));
        Expect(response.kind == wire::ResponseKind::Error && response.errorCode == item.second,
               "Invalid batching configuration was not a correlated local rejection");
        Expect(!h.worker.Failed() && h.counters->models == 0 && h.counters->sessions == 0,
               "Rejected batching configuration initialized resources or faulted the Worker");
    }
    h.Configure({}, 4);
    configure.maxDecodeBatchSize = 2;
    configure.requiredCapabilities.emplace_back("continuous_batching_v1");
    Expect(h.Call(configure).errorCode == wire::ErrorCode::InvalidArgument,
           "Batching configuration changed during the Worker lifetime");
    const auto model = h.Load();
    const auto session = h.Create(model);
    const auto request = h.Submit(session, 3);
    Expect(h.Final(request).summary.status == wire::RequestStatus::Completed,
           "Batching fallback could not run after a configuration rejection");
    h.Release(wire::ObjectType::Request, request);
    h.Shutdown();
}

void TestUnknownCommandAndLogLimitReduction() {
    Harness h;
    aila::ipc::Frame unknown;
    unknown.header.protocol = wire::kProtocolVersion;
    unknown.header.request_id = ++h.nextId;
    unknown.header.kind = "request";
    unknown.header.method = "future.command";
    unknown.header.payload_json = std::string("{\"epoch\":\"") + Epoch + "\"}";
    std::string error;
    Expect(h.worker.Enqueue(std::move(unknown), error), "unknown method broke command framing");
    aila::worker::InferenceWorkerOutput rejected;
    Expect(h.worker.ReadResponse(rejected, 5s), "unknown command had no response");
    Expect(rejected.frame.header.kind == "error" && rejected.frame.header.method == "future.command" &&
           rejected.frame.header.payload_json.find("InvalidArgument") != std::string::npos,
           "unknown command was not a correlated local rejection");
    if (rejected.written) rejected.written();
    Expect(!h.worker.Failed(), "unknown command failed healthy domain");

    h.worker.Log(1, std::string(1000, 'x'));
    wire::Limits small;
    small.maxLogQueueBytes = 512;
    wire::Command configure;
    configure.method = wire::Method::WorkerConfigure;
    configure.limits = small;
    configure.requiredCapabilities = wire::default_capabilities();
    const auto rejectedId = h.Send(configure);
    const auto result = h.Wait(rejectedId, false); // Keep the queued/in-flight log reserved.
    Expect(result.kind == wire::ResponseKind::Error && result.errorCode == wire::ErrorCode::ResourceLimit,
           "configure shrank log budget below an outstanding log");
    h.Pump();
    h.Configure(small);
    h.worker.Log(1, std::string(1000, 'x'));
    Expect(h.Query().droppedLogs >= 1, "lower log limit underflowed instead of dropping an oversized log");
    h.Shutdown();
}

} // namespace

int main() {
    try {
        for (const char* backend : {"qwen3_nf4", "qwen35_nf4", "qwen35_bf16"}) {
            TestConcurrentControlAndParentLifetime(backend);
            TestBackpressureAndRetainedReservations(backend);
            TestAdmissionRollbackAndStaleEpoch(backend);
            TestLifecycleFaultBroadcast(backend);
            TestLoadDoesNotBlockControl(backend);
            TestBackendValidationAndLoadRecovery(backend);
            TestSessionIsolationAndContinuation(backend);
            TestAllocationLimitForwarding(backend);
            TestThinkingBudgetForwarding(backend);
            TestContextReservationAndReleasedParents(backend);
            TestContextBusyAndLocalRecovery(backend);
            TestContextShutdownAndDomainFault(backend);
            // These CPU backends deliberately have no native batch executor.
            // Opt-in must preserve their serial fallback and service boundaries.
            TestConcurrentControlAndParentLifetime(backend, 4);
            TestBackpressureAndRetainedReservations(backend, 2);
            TestLifecycleFaultBroadcast(backend, 4);
            TestSessionIsolationAndContinuation(backend, 4);
            std::cout << "[PASS] " << backend << " dispatch, lifecycle, control, isolation and load recovery\n";
        }
        TestUnknownCommandAndLogLimitReduction();
        TestDecodeBatchingConfigurationRejectionAndRecovery();
        TestWorkerLogControl();
        TestRuntimeMemoryDoesNotBlockControl();
        std::cout << "[PASS] inference Worker lifecycle, routing, backpressure, faults and reservations\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
