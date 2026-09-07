#include "worker/InferenceWorker.hpp"
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
    std::atomic<int> models{0}, sessions{0};
    std::atomic<bool> failSession{false};
    std::atomic<bool> faultSession{false};
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
    void Reset() override { length = 0; }
    int Truncate(int value) override { length = value; return value; }
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
    std::shared_ptr<Counters> counters;
    int capacity;
    int length = 0;
    std::shared_ptr<void> model;
};

class FakeModel final : public aila::worker::InferenceModel, public std::enable_shared_from_this<FakeModel> {
public:
    FakeModel(std::shared_ptr<Counters> counters, int capacity)
        : counters(std::move(counters)), capacity(capacity) { ++this->counters->models; }
    ~FakeModel() override { --counters->models; }
    std::shared_ptr<core::SessionState> CreateSession() override {
        if (counters->faultSession.exchange(false)) throw core::Error(core::ErrorCode::RuntimeFault, "injected Session allocation device failure");
        if (counters->failSession.exchange(false)) throw std::bad_alloc();
        return std::make_shared<core::SessionState>(std::make_unique<FakeBackend>(counters, capacity, shared_from_this()));
    }
    int VocabSize() const noexcept override { return 256; }
    int MaxSequenceLength() const noexcept override { return capacity; }
    size_t SessionReservationBytes() const noexcept override { return static_cast<size_t>(capacity) * 16; }
private:
    std::shared_ptr<Counters> counters;
    int capacity;
};

class FakeFactory final : public aila::worker::InferenceModelFactory {
public:
    explicit FakeFactory(std::shared_ptr<Counters> counters) : counters(std::move(counters)) {}
    std::shared_ptr<aila::worker::InferenceModel> LoadQwen3(const std::string& directory,
                                                        aila::worker::InferenceModelOptions options) override {
        counters->load.Enter();
        if (directory == "bad-model") throw std::invalid_argument("injected file/configuration failure");
        return std::make_shared<FakeModel>(counters, options.maxSequenceLength);
    }
private:
    std::shared_ptr<Counters> counters;
};

class Harness {
public:
    Harness() : counters(std::make_shared<Counters>()), worker(Epoch, std::make_unique<FakeFactory>(counters)) {}
    ~Harness() { counters->step.Open(); counters->load.Open(); worker.Stop(); }
    uint64_t Send(wire::Command command) {
        command.rpcId = ++nextId;
        if (command.epoch.empty()) command.epoch = Epoch;
        aila::ipc::Frame frame;
        wire::ValidationError error;
        Expect(wire::make_command(command, frame, error, limits), "make_command: " + error.message);
        std::string transportError;
        Expect(worker.Enqueue(std::move(frame), transportError), "Enqueue: " + transportError);
        return command.rpcId;
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
            else if (event.kind != wire::EventKind::Log) {
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
    void Configure(wire::Limits selected = {}) {
        wire::Command command;
        command.method = wire::Method::WorkerConfigure;
        command.limits = selected;
        command.requiredCapabilities = wire::default_capabilities();
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result, "configure failed: " + response.message);
        limits = selected;
    }
    uint64_t Load(unsigned sessions = 4) {
        wire::Command command;
        command.method = wire::Method::ModelLoad;
        command.modelDirectory = "fake-model";
        command.maxSequenceLength = 512;
        command.maxSessions = sessions;
        const auto response = Call(command);
        Expect(response.kind == wire::ResponseKind::Result, "model load failed: " + response.message);
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

    wire::Limits limits;
    std::shared_ptr<Counters> counters;
    aila::worker::InferenceWorker worker;
    uint64_t nextId = 0, lastOrdinal = 0;
    unsigned domainFaults = 0;
    std::map<uint64_t, std::deque<wire::Response>> responses;
    std::map<uint64_t, std::vector<wire::Event>> events;
};

void TestConcurrentControlAndParentLifetime() {
    Harness h;
    h.Configure();
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

void TestBackpressureAndRetainedReservations() {
    Harness h;
    wire::Limits limits;
    limits.tokenWindow = 4;
    limits.maxBufferedTokensPerRequest = 8;
    limits.maxOutstandingRequests = 2;
    limits.maxActiveRequests = 2;
    h.Configure(limits);
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

void TestAdmissionRollbackAndStaleEpoch() {
    Harness h;
    h.Configure();
    wire::Command invalidLoad;
    invalidLoad.method = wire::Method::ModelLoad;
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

void TestLifecycleFaultBroadcast() {
    Harness h;
    h.Configure();
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

void TestLoadDoesNotBlockControl() {
    Harness h;
    h.Configure();
    h.counters->load.Block();
    wire::Command load;
    load.method = wire::Method::ModelLoad; load.modelDirectory = "fake-model";
    const auto loadId = h.Send(load);
    h.counters->load.Wait();
    Expect(h.Query().accepting, "worker.query waited on model loading");
    h.counters->load.Open();
    Expect(h.Wait(loadId).kind == wire::ResponseKind::Result, "gated model load failed");
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
        TestConcurrentControlAndParentLifetime();
        TestBackpressureAndRetainedReservations();
        TestAdmissionRollbackAndStaleEpoch();
        TestLifecycleFaultBroadcast();
        TestLoadDoesNotBlockControl();
        TestUnknownCommandAndLogLimitReduction();
        std::cout << "[PASS] inference Worker lifecycle, routing, backpressure, faults and reservations\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
