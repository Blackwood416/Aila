#include "runtime/InferenceClient.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace wire = aila::ipc::v2;
using namespace aila::runtime;
using namespace std::chrono_literals;
using aila::ipc::Frame;
using Clock = std::chrono::steady_clock;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void ExpectError(Function function, ClientErrorCode expected,
                 std::optional<wire::ErrorCode> remote = {}) {
    try { function(); }
    catch (const InferenceClientError& error) {
        Expect(error.Code() == expected, "Unexpected client error: " + std::string(error.what()));
        if (remote) Expect(error.RemoteCode() == remote, "Unexpected remote error code: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected InferenceClientError was not thrown");
}

template <typename Value>
Value Get(std::future<Value>& future) {
    Expect(future.wait_for(3s) == std::future_status::ready, "Timed out waiting for test operation");
    return future.get();
}

// Used only for public counters with no wait API. Every test transition waits
// for an observed state or a pipe gate, never for an assumed scheduling delay.
template <typename Predicate>
void Until(Predicate predicate, const char* message) {
    const auto deadline = Clock::now() + 3s;
    while (!predicate()) {
        Expect(Clock::now() < deadline, message);
        std::this_thread::yield();
    }
}

ExpectedInferenceIdentity ExpectedIdentity() {
    return {"fake-v2-build", "C:\\aila-v2\\AilaWorker.exe", "C:\\aila-v2", "C:\\aila-v2",
            "C:\\aila-v2;C:\\Windows\\System32;C:\\Windows"};
}

wire::Hello Hello() {
    const auto identity = ExpectedIdentity();
    wire::Hello hello;
    hello.epoch = "0123456789abcdef0123456789abcdef";
    hello.buildId = identity.buildId;
    hello.executable = identity.executable;
    hello.runtimeDirectory = identity.runtimeDirectory;
    hello.currentDirectory = identity.currentDirectory;
    hello.path = identity.path;
    hello.capabilities = wire::default_capabilities();
    return hello;
}

wire::Summary Summary(const wire::Command& submit, uint64_t count,
                      wire::RequestStatus status = wire::RequestStatus::Completed,
                      wire::EndReason reason = wire::EndReason::Length) {
    wire::Summary result;
    result.status = status;
    result.endReason = reason;
    result.promptTokensProcessed = submit.promptIds.size();
    result.generatedTokens = count;
    result.eventCount = count + 1;
    if (status == wire::RequestStatus::Failed) result.errorCode = wire::ErrorCode::RequestFailed;
    if (reason == wire::EndReason::SchedulerStopped) result.errorCode = wire::ErrorCode::SchedulerStopped;
    if (reason == wire::EndReason::RuntimeFault) result.errorCode = wire::ErrorCode::RuntimeFault;
    if (reason == wire::EndReason::OutputBackpressure) result.errorCode = wire::ErrorCode::ResourceLimit;
    return result;
}

class ControlledTransport final : public InferenceTransport {
public:
    explicit ControlledTransport(wire::Hello hello = Hello(), bool legacy = false) : hello_(std::move(hello)) {
        Frame frame;
        wire::ValidationError error;
        const bool valid = wire::make_hello(hello_, frame, error);
        Expect(valid, "Invalid fake hello: " + error.message);
        if (legacy) frame.header.protocol = 1;
        responses_.push_back(std::move(frame));
        responseSerial_ = 1;
        held_.insert(wire::Method::RequestSubmit);
    }

    bool ReadResponse(Frame& frame, std::string& error) override { return Read(false, frame, error); }
    bool ReadEvent(Frame& frame, std::string& error) override { return Read(true, frame, error); }

    bool WriteCommand(const Frame& frame, std::string& error) override {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopped_) { error = "Fake transport stopped"; return false; }
        wire::Command command;
        wire::ValidationError validation;
        if (!wire::parse_command(frame, command, validation, limits_)) {
            error = "Invalid host command: " + validation.message;
            return false;
        }
        Expect(command.rpcId > lastRpc_, "Command RPC IDs were reused or written out of order");
        Expect(command.epoch == hello_.epoch, "Command used the wrong epoch");
        lastRpc_ = command.rpcId;
        history_.push_back(command);
        observed_.push_back(command);
        changed_.notify_all();
        if (blockedMethod_ == command.method) {
            blockedMethod_.reset();
            blocked_ = true;
            changed_.notify_all();
            changed_.wait(lock, [&] { return !blocked_ || stopped_; });
            if (stopped_) { error = "Blocked fake writer stopped"; return false; }
        }
        auto response = Base(command);
        switch (command.method) {
        case wire::Method::WorkerConfigure:
            limits_ = command.limits;
            response.limits = limits_;
            response.maxDecodeBatchSize = command.maxDecodeBatchSize;
            response.capabilities = hello_.capabilities;
            response.worker.domainStatus = wire::DomainStatus::Ready;
            break;
        case wire::Method::WorkerQuery:
            response.worker.domainStatus = shuttingDown_ ? wire::DomainStatus::Stopping : wire::DomainStatus::Ready;
            response.worker.accepting = !shuttingDown_;
            response.worker.stopping = shuttingDown_;
            break;
        case wire::Method::WorkerSetLogLevel:
            response.logLevel = command.logLevel;
            break;
        case wire::Method::WorkerMemoryQuery:
            response.runtimeMemory = {wire::RuntimeMemoryState::Ready,
                wire::RuntimeMemoryUsage{4096, (uint64_t{1} << 53) + 1, 16, 128, 4, 1, 8192, 64}};
            break;
        case wire::Method::ModelLoad:
            response.model = {++modelId_, 2048, command.maxSequenceLength, 4096,
                              {std::string(wire::model_capability(command.backend, command.kvType)),
                               "token_requests_v1", "session_lifecycle_v1"}};
            if (command.allocationLimits) {
                response.model.memory = wire::ModelMemoryInfo{*command.allocationLimits, 0, 0};
                response.model.capabilities.emplace_back("allocation_limits_v1");
            }
            models_.emplace(response.model.modelId, response.model);
            break;
        case wire::Method::SessionCreate: {
            const auto& model = models_.at(command.modelId);
            response.session = {++sessionId_, model.modelId, model.maxSequenceLength,
                                model.sessionReservationBytes, false, 0};
            sessions_.emplace(response.session.sessionId, response.session);
            break;
        }
        case wire::Method::SessionQuery:
            response.session = sessions_.at(command.sessionId);
            break;
        case wire::Method::SessionReset:
        case wire::Method::SessionTruncate:
            response.session = sessions_.at(command.sessionId);
            response.session.contextLength = command.method == wire::Method::SessionReset ? 0 : command.contextLength;
            sessions_.at(command.sessionId) = response.session;
            break;
        case wire::Method::RequestSubmit: {
            Record record;
            record.submit = command;
            record.identity = {command.rpcId, sessions_.at(command.sessionId).modelId, command.sessionId};
            requests_.emplace(command.rpcId, std::move(record));
            response.kind = wire::ResponseKind::Accepted;
            response.phase = wire::SubmissionPhase::Accepted;
            response.identity = requests_.at(command.rpcId).identity;
            response.tokenWindow = limits_.tokenWindow;
            break;
        }
        case wire::Method::RequestCancel: {
            const auto& request = requests_.at(command.inferenceId);
            response.summary.status = request.summary.status;
            response.cancelRequested = !wire::is_terminal(request.summary.status);
            break;
        }
        case wire::Method::RequestQuery: {
            const auto& request = requests_.at(command.inferenceId);
            response.request.summary = request.summary;
            response.request.lastSentSequence = request.published + (request.terminal ? 1 : 0);
            response.request.lastAcknowledgedSequence = request.acknowledged;
            response.request.completionReady = wire::is_terminal(request.summary.status);
            break;
        }
        case wire::Method::RequestAck: {
            auto& request = requests_.at(command.inferenceId);
            const uint64_t sent = request.published + (request.terminal ? 1 : 0);
            Expect(command.consumedSequence >= request.acknowledged && command.consumedSequence <= sent,
                   "Host ACK did not describe a consumed contiguous prefix");
            request.acknowledged = command.consumedSequence;
            response.acknowledgedSequence = command.consumedSequence;
            break;
        }
        case wire::Method::HandleRelease:
            if (command.objectType == wire::ObjectType::Request) requests_.at(command.objectId).released = true;
            response.releaseAccepted = true;
            break;
        case wire::Method::WorkerShutdown:
            shuttingDown_ = true;
            shutdown_ = command;
            if (autoShutdown_) {
                for (auto& pair : requests_) {
                    auto& request = pair.second;
                    if (request.rejected) continue;
                    if (!request.accepted) {
                        auto rejection = Base(request.submit);
                        rejection.kind = wire::ResponseKind::Error;
                        rejection.phase = wire::SubmissionPhase::Rejected;
                        rejection.errorCode = wire::ErrorCode::SchedulerStopped;
                        rejection.message = "Fake shutdown rejected pending submit";
                        heldResponses_.erase(request.submit.rpcId);
                        request.rejected = true;
                        PushResponseLocked(std::move(rejection));
                        continue;
                    }
                    const auto summary = request.final || request.terminal ? request.summary
                        : Summary(request.submit, request.published, wire::RequestStatus::Cancelled,
                                  wire::EndReason::SchedulerStopped);
                    if (!request.final) FinalLocked(request, summary);
                    if (!request.terminal) TerminalLocked(request, summary);
                }
                MaybeShutdownLocked();
            }
            return true;
        }
        if (held_.count(command.method)) heldResponses_[command.rpcId] = std::move(response);
        else {
            if (command.method == wire::Method::RequestSubmit) requests_.at(command.rpcId).accepted = true;
            PushResponseLocked(std::move(response));
        }
        MaybeShutdownLocked();
        return true;
    }

    void SetLimits(const wire::Limits& value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        limits_ = value;
        limitsInstalledBeforeRead_ = limitsInstalledBeforeRead_ || (responseReadAttempts_ == 0 && eventReadAttempts_ == 0);
    }
    void Stop(bool force) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        forced_ = forced_ || force;
        blocked_ = false;
        changed_.notify_all();
    }
    bool WaitForExit(std::chrono::milliseconds, std::string& error) override {
        std::lock_guard<std::mutex> lock(mutex_);
        waitedForExit_ = true;
        if (throwOnExit_) throw std::runtime_error("Injected process exit failure");
        if (!exitSuccess_) { error = "Injected nonzero exit code"; return false; }
        return true;
    }

    void Hold(wire::Method method, bool hold = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (hold) held_.insert(method); else held_.erase(method);
    }
    void BlockNext(wire::Method method) { std::lock_guard<std::mutex> lock(mutex_); blockedMethod_ = method; }
    void Unblock() { std::lock_guard<std::mutex> lock(mutex_); blocked_ = false; changed_.notify_all(); }
    void AutoShutdown(bool enabled) { std::lock_guard<std::mutex> lock(mutex_); autoShutdown_ = enabled; }
    void BadExit(bool throwException) {
        std::lock_guard<std::mutex> lock(mutex_);
        exitSuccess_ = false;
        throwOnExit_ = throwException;
    }
    bool Forced() const { std::lock_guard<std::mutex> lock(mutex_); return forced_; }
    bool Stopped() const { std::lock_guard<std::mutex> lock(mutex_); return stopped_; }
    bool WaitedForExit() const { std::lock_guard<std::mutex> lock(mutex_); return waitedForExit_; }
    bool EarlyLimits() const { std::lock_guard<std::mutex> lock(mutex_); return limitsInstalledBeforeRead_; }
    void WaitStopped() {
        std::unique_lock<std::mutex> lock(mutex_);
        Expect(changed_.wait_for(lock, 3s, [&] { return stopped_; }), "Transport did not stop");
    }
    std::vector<wire::Command> History() const { std::lock_guard<std::mutex> lock(mutex_); return history_; }
    wire::Identity Identity(uint64_t id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.at(id).identity;
    }

    wire::Command Take(wire::Method method, uint64_t target = 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto find = [&] {
            return std::find_if(observed_.begin(), observed_.end(), [&](const auto& command) {
                return command.method == method && (target == 0 ||
                    (method == wire::Method::HandleRelease ? command.objectId : command.inferenceId) == target);
            });
        };
        Expect(changed_.wait_for(lock, 3s, [&] { return find() != observed_.end() || stopped_; }),
               "Command was not written: " + std::string(wire::to_string(method)));
        const auto item = find();
        Expect(item != observed_.end(), "Transport stopped before expected command: " + std::string(wire::to_string(method)));
        auto result = std::move(*item);
        observed_.erase(item);
        return result;
    }

    uint64_t ReplyHeld(const wire::Command& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = heldResponses_.find(command.rpcId);
        Expect(found != heldResponses_.end(), "No held response for this RPC");
        if (command.method == wire::Method::RequestSubmit) requests_.at(command.rpcId).accepted = true;
        auto response = std::move(found->second);
        heldResponses_.erase(found);
        const auto serial = PushResponseLocked(std::move(response));
        MaybeShutdownLocked();
        return serial;
    }
    void RejectSubmit(const wire::Command& command) {
        std::lock_guard<std::mutex> lock(mutex_);
        heldResponses_.erase(command.rpcId);
        requests_.at(command.rpcId).rejected = true;
        auto response = Base(command);
        response.kind = wire::ResponseKind::Error;
        response.phase = wire::SubmissionPhase::Rejected;
        response.errorCode = wire::ErrorCode::Busy;
        response.message = "Session is busy";
        PushResponseLocked(std::move(response));
    }
    uint64_t Reply(wire::Response response) {
        std::lock_guard<std::mutex> lock(mutex_);
        heldResponses_.erase(response.rpcId);
        return PushResponseLocked(std::move(response));
    }
    uint64_t Token(const wire::Command& submit, uint64_t sequence, int32_t token = 101) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& request = requests_.at(submit.rpcId);
        request.published = sequence;
        if (!request.final && !request.terminal) {
            request.summary.status = wire::RequestStatus::Decode;
            request.summary.promptTokensProcessed = submit.promptIds.size();
            request.summary.generatedTokens = sequence;
        }
        wire::Event event;
        event.kind = wire::EventKind::Token;
        event.identity = request.identity;
        event.sequence = sequence;
        event.tokenId = token;
        event.summary.status = wire::RequestStatus::Decode;
        event.summary.promptTokensProcessed = submit.promptIds.size();
        event.summary.generatedTokens = sequence;
        return PushEventLocked(std::move(event));
    }
    uint64_t Final(const wire::Command& submit, const wire::Summary& summary) {
        std::lock_guard<std::mutex> lock(mutex_);
        return FinalLocked(requests_.at(submit.rpcId), summary);
    }
    uint64_t Terminal(const wire::Command& submit, const wire::Summary& summary) {
        std::lock_guard<std::mutex> lock(mutex_);
        return TerminalLocked(requests_.at(submit.rpcId), summary);
    }
    uint64_t Event(wire::Event event) {
        std::lock_guard<std::mutex> lock(mutex_);
        return PushEventLocked(std::move(event));
    }
    uint64_t Log(const std::string& message) {
        wire::Event event;
        event.kind = wire::EventKind::Log;
        event.logLevel = 2;
        event.message = message;
        return Event(std::move(event));
    }
    void Finish(const wire::Command& submit, uint64_t count, bool finalFirst = true,
                wire::RequestStatus status = wire::RequestStatus::Completed,
                wire::EndReason reason = wire::EndReason::Length) {
        const auto summary = Summary(submit, count, status, reason);
        if (finalFirst) { Final(submit, summary); Terminal(submit, summary); }
        else { Terminal(submit, summary); Final(submit, summary); }
    }
    void ShutdownReply(const wire::Command& command, uint64_t ordinal) {
        auto response = Base(command);
        response.worker.domainStatus = wire::DomainStatus::Closed;
        response.lastEventOrdinal = ordinal;
        Reply(std::move(response));
    }
    void ClosePipe(bool event) {
        std::lock_guard<std::mutex> lock(mutex_);
        (event ? eventClosed_ : responseClosed_) = true;
        changed_.notify_all();
    }
    void WaitEventEof() {
        std::unique_lock<std::mutex> lock(mutex_);
        Expect(changed_.wait_for(lock, 3s, [&] { return eventEofRead_ || stopped_; }), "Event EOF was not read");
    }
    void ProcessedEvent(uint64_t serial) { Processed(true, serial); }
    void ProcessedResponse(uint64_t serial) { Processed(false, serial); }

    wire::Response Base(const wire::Command& command) const {
        wire::Response response;
        response.epoch = hello_.epoch;
        response.method = command.method;
        response.rpcId = command.rpcId;
        return response;
    }

private:
    struct Record {
        wire::Command submit;
        wire::Identity identity;
        wire::Summary summary;
        uint64_t published = 0;
        uint64_t acknowledged = 0;
        bool accepted = false;
        bool rejected = false;
        bool final = false;
        bool terminal = false;
        bool released = false;
    };

    bool Read(bool event, Frame& frame, std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        auto& attempts = event ? eventReadAttempts_ : responseReadAttempts_;
        auto& queue = event ? events_ : responses_;
        ++attempts;
        changed_.notify_all();
        changed_.wait(lock, [&] { return !queue.empty() || stopped_ || (event ? eventClosed_ : responseClosed_); });
        if (stopped_ || queue.empty()) {
            if (event && eventClosed_) eventEofRead_ = true;
            changed_.notify_all();
            error = "Fake pipe EOF";
            return false;
        }
        frame = std::move(queue.front());
        queue.pop_front();
        changed_.notify_all();
        return true;
    }
    void Processed(bool event, uint64_t serial) {
        std::unique_lock<std::mutex> lock(mutex_);
        Expect(changed_.wait_for(lock, 3s, [&] {
            return (event ? eventReadAttempts_ : responseReadAttempts_) > serial || stopped_;
        }), "Reader did not finish processing its frame");
        Expect(!stopped_, "Connection failed while awaiting valid frame processing");
    }
    uint64_t PushResponseLocked(wire::Response response) {
        Frame frame;
        wire::ValidationError error;
        const bool valid = wire::make_response(response, frame, error, limits_);
        Expect(valid, "Invalid fake response for " + std::string(wire::to_string(response.method)) + ": " + error.message);
        responses_.push_back(std::move(frame));
        ++responseSerial_;
        changed_.notify_all();
        return responseSerial_;
    }
    uint64_t PushEventLocked(wire::Event event) {
        if (event.epoch.empty()) event.epoch = hello_.epoch;
        if (!event.eventOrdinal) event.eventOrdinal = ++ordinal_;
        Frame frame;
        wire::ValidationError error;
        const bool valid = wire::make_event(event, frame, error, limits_);
        Expect(valid, "Invalid fake event: " + error.message);
        events_.push_back(std::move(frame));
        ++eventSerial_;
        changed_.notify_all();
        return eventSerial_;
    }
    uint64_t FinalLocked(Record& request, const wire::Summary& summary) {
        request.final = true;
        request.summary = summary;
        auto response = Base(request.submit);
        response.kind = summary.status == wire::RequestStatus::Failed ? wire::ResponseKind::Error : wire::ResponseKind::Result;
        response.phase = wire::SubmissionPhase::Completed;
        response.identity = request.identity;
        response.summary = summary;
        return PushResponseLocked(std::move(response));
    }
    uint64_t TerminalLocked(Record& request, const wire::Summary& summary) {
        request.terminal = true;
        request.summary = summary;
        wire::Event event;
        event.kind = wire::EventKind::Terminal;
        event.identity = request.identity;
        event.sequence = summary.eventCount;
        event.summary = summary;
        return PushEventLocked(std::move(event));
    }
    void MaybeShutdownLocked() {
        if (!autoShutdown_ || !shutdown_ || shutdownReplied_ || !heldResponses_.empty()) return;
        for (const auto& pair : requests_) {
            const auto& request = pair.second;
            if (!request.rejected && (!request.final || !request.terminal || request.acknowledged != request.summary.eventCount)) return;
        }
        auto response = Base(*shutdown_);
        response.worker.domainStatus = wire::DomainStatus::Closed;
        response.lastEventOrdinal = ordinal_;
        shutdownReplied_ = true;
        PushResponseLocked(std::move(response));
    }

    wire::Hello hello_;
    wire::Limits limits_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Frame> responses_;
    std::deque<Frame> events_;
    std::deque<wire::Command> observed_;
    std::vector<wire::Command> history_;
    std::set<wire::Method> held_;
    std::map<uint64_t, wire::Response> heldResponses_;
    std::map<uint64_t, wire::ModelInfo> models_;
    std::map<uint64_t, wire::SessionInfo> sessions_;
    std::map<uint64_t, Record> requests_;
    std::optional<wire::Method> blockedMethod_;
    std::optional<wire::Command> shutdown_;
    uint64_t lastRpc_ = 0;
    uint64_t modelId_ = 0;
    uint64_t sessionId_ = 0;
    uint64_t ordinal_ = 0;
    uint64_t responseSerial_ = 0;
    uint64_t eventSerial_ = 0;
    uint64_t responseReadAttempts_ = 0;
    uint64_t eventReadAttempts_ = 0;
    bool limitsInstalledBeforeRead_ = false;
    bool stopped_ = false;
    bool forced_ = false;
    bool blocked_ = false;
    bool responseClosed_ = false;
    bool eventClosed_ = false;
    bool eventEofRead_ = false;
    bool shuttingDown_ = false;
    bool autoShutdown_ = true;
    bool shutdownReplied_ = false;
    bool waitedForExit_ = false;
    bool exitSuccess_ = true;
    bool throwOnExit_ = false;
};

struct Fixture {
    std::shared_ptr<ControlledTransport> transport;
    std::shared_ptr<InferenceClient> client;
    std::shared_ptr<InferenceModel> model;
    std::vector<std::shared_ptr<InferenceSession>> sessions;

    explicit Fixture(InferenceClientOptions options = {}, size_t sessionCount = 2, wire::Hello hello = Hello())
        : transport(std::make_shared<ControlledTransport>(std::move(hello))) {
        options.commandTimeout = 1s;
        options.shutdownTimeout = 1s;
        client = InferenceClient::Connect(transport, ExpectedIdentity(), options);
        transport->Take(wire::Method::WorkerConfigure);
        Expect(transport->EarlyLimits(), "Limits were installed after the first read allocation");
        model = client->LoadQwen3("C:\\fake-qwen3");
        transport->Take(wire::Method::ModelLoad);
        for (size_t i = 0; i < sessionCount; ++i) {
            sessions.push_back(model->CreateSession());
            transport->Take(wire::Method::SessionCreate);
        }
    }
    ~Fixture() {
        if (!transport->Stopped()) transport->Stop(true);
        if (client) { try { client->Shutdown(1s); } catch (...) {} }
    }
    std::pair<std::shared_ptr<InferenceRequest>, wire::Command> Submit(size_t session = 0, uint32_t maxNewTokens = 32) {
        InferenceRequestOptions options;
        options.generation.maxNewTokens = maxNewTokens;
        auto future = std::async(std::launch::async, [&, options] { return sessions.at(session)->Submit({11, 12}, options); });
        auto command = transport->Take(wire::Method::RequestSubmit);
        transport->ReplyHeld(command);
        return {Get(future), std::move(command)};
    }
    void Acked(const std::shared_ptr<InferenceRequest>& request, uint64_t sequence) {
        Until([&] { return request->Snapshot().acknowledgedSequence == sequence || transport->Stopped(); }, "ACK was not processed");
        Expect(!transport->Stopped(), "Connection failed while draining ACK");
    }
    void Close() {
        client->Shutdown(2s);
        Expect(!transport->Forced(), "Graceful shutdown forcibly stopped the Worker");
        Expect(transport->WaitedForExit(), "Shutdown skipped process exit verification");
        const auto snapshot = client->Snapshot();
        Expect(snapshot.status == wire::DomainStatus::Closed && snapshot.pendingCommands == 0 &&
               snapshot.outstandingRequests == 0 && snapshot.ingressBytes == 0 && snapshot.transportBytes == 0,
               "Graceful shutdown leaked command/transport reservations");
    }
};

InferenceReadEvent ReadToken(const std::shared_ptr<InferenceRequest>& request, uint64_t sequence) {
    const auto result = request->Read(2s);
    Expect(result.status == InferenceReadStatus::Event && result.event && !result.event->terminal &&
           result.event->sequence == sequence, "Expected token sequence " + std::to_string(sequence));
    return *result.event;
}

InferenceReadEvent ReadTerminal(const std::shared_ptr<InferenceRequest>& request, bool localFailure = false) {
    const auto result = request->Read(2s);
    Expect(result.status == InferenceReadStatus::Event && result.event && result.event->terminal &&
           result.event->localFailure == localFailure, "Expected a single matching terminal");
    Expect(request->Read(0ms).status == InferenceReadStatus::Closed, "Request delivered more than one terminal");
    return *result.event;
}

size_t Count(const std::vector<wire::Command>& commands, wire::Method method) {
    return static_cast<size_t>(std::count_if(commands.begin(), commands.end(), [&](const auto& command) { return command.method == method; }));
}

void CheckShutdownOrder(const std::vector<wire::Command>& commands) {
    bool shutdown = false;
    for (const auto& command : commands) {
        if (command.method == wire::Method::WorkerShutdown) {
            Expect(!shutdown, "Concurrent callers sent duplicate shutdown commands");
            shutdown = true;
        } else if (shutdown) {
            Expect(command.method == wire::Method::RequestAck, "User/release RPC was sent after shutdown");
        }
    }
    Expect(shutdown, "Shutdown was never sent");
}

void ParallelRequestsAndCompletionOrder() {
    Fixture fixture;
    InferenceRequestOptions options;
    options.generation.maxNewTokens = 3;
    auto first = std::async(std::launch::async, [&] { return fixture.sessions[0]->Submit({11, 12}, options); });
    const auto a = fixture.transport->Take(wire::Method::RequestSubmit);
    auto second = std::async(std::launch::async, [&] { return fixture.sessions[1]->Submit({13, 14}, options); });
    const auto b = fixture.transport->Take(wire::Method::RequestSubmit);
    Expect(first.wait_for(0ms) == std::future_status::timeout && second.wait_for(0ms) == std::future_status::timeout,
           "Submit did not wait for its own accepted response");
    fixture.transport->ProcessedEvent(fixture.transport->Token(a, 1, 201));
    fixture.transport->ReplyHeld(b);
    auto requestB = Get(second);
    Expect(!fixture.transport->Stopped(), "Token before accepted broke the connection");
    fixture.transport->Token(b, 1, 301);
    const auto summaryB = Summary(b, 1, wire::RequestStatus::Completed, wire::EndReason::StopToken);
    fixture.transport->ProcessedEvent(fixture.transport->Terminal(b, summaryB));
    Expect(ReadToken(requestB, 1).tokenId == 301, "Cross-routed B token");
    Expect(requestB->Read(0ms).status == InferenceReadStatus::Timeout, "Terminal escaped before its final response");
    fixture.transport->Final(b, summaryB);
    ReadTerminal(requestB);
    Expect(first.wait_for(0ms) == std::future_status::timeout, "B completion incorrectly completed A submission");
    fixture.transport->ReplyHeld(a);
    auto requestA = Get(first);
    Expect(ReadToken(requestA, 1).tokenId == 201, "Pre-accepted A token was not retained");
    fixture.transport->ProcessedResponse(fixture.transport->Final(a, Summary(a, 3)));
    Expect(requestA->Read(0ms).status == InferenceReadStatus::Timeout, "Final response escaped before the event drain");
    fixture.transport->Token(a, 2, 202);
    fixture.transport->Token(a, 3, 203);
    fixture.transport->Terminal(a, Summary(a, 3));
    ReadToken(requestA, 2);
    ReadToken(requestA, 3);
    Expect(ReadTerminal(requestA).summary.generatedTokens == 3, "A final count changed");
    fixture.Acked(requestA, 4);
    fixture.Acked(requestB, 2);
    fixture.Close();
}

void AckWindowAndIndependentSlowReader() {
    InferenceClientOptions options;
    options.limits.tokenWindow = 2;
    Fixture fixture(options);
    auto [requestA, a] = fixture.Submit(0, 4);
    auto [requestB, b] = fixture.Submit(1, 1);
    fixture.transport->ProcessedEvent(fixture.transport->Token(a, 1));
    fixture.transport->ProcessedEvent(fixture.transport->Token(a, 2));
    fixture.transport->Token(b, 1);
    fixture.transport->Finish(b, 1);
    ReadToken(requestB, 1);
    ReadTerminal(requestB);
    fixture.Acked(requestB, 2);
    Expect(requestA->Snapshot().bufferedTokens == 2 && requestA->Snapshot().acknowledgedSequence == 0,
           "Host acknowledged A without consumption, or B stole A's ring");
    fixture.transport->Hold(wire::Method::RequestAck);
    ReadToken(requestA, 1);
    ReadToken(requestA, 2);
    const auto firstAck = fixture.transport->Take(wire::Method::RequestAck, a.rpcId);
    Expect(firstAck.consumedSequence > 0 && firstAck.consumedSequence <= 2, "Invalid cumulative ACK");
    // The Worker can refill credit before the separate ACK response pipe is read.
    fixture.transport->ProcessedEvent(fixture.transport->Token(a, 3));
    ReadToken(requestA, 3);
    Expect(!fixture.transport->Forced(), "Refill before ACK response was treated as a window violation");
    fixture.transport->Hold(wire::Method::RequestAck, false);
    fixture.transport->ReplyHeld(firstAck);
    Until([&fixture, requestA = requestA] { return requestA->Snapshot().acknowledgedSequence >= 2 || fixture.transport->Stopped(); },
          "Consumed prefix did not advance after delayed ACK reply");
    fixture.transport->Token(a, 4);
    fixture.transport->Finish(a, 4, false);
    ReadToken(requestA, 4);
    ReadTerminal(requestA);
    fixture.Acked(requestA, 5);
    fixture.Close();
}

void ReleaseDrainsAndReturnsReservations() {
    InferenceClientOptions options;
    options.limits.tokenWindow = 2;
    Fixture fixture(options);
    auto [requestA, a] = fixture.Submit(0, 4);
    auto [requestB, b] = fixture.Submit(1, 1);
    fixture.transport->Token(a, 1);
    fixture.transport->ProcessedEvent(fixture.transport->Token(a, 2));
    requestA->Release();
    const auto release = fixture.transport->Take(wire::Method::HandleRelease, a.rpcId);
    const auto ack = fixture.transport->Take(wire::Method::RequestAck, a.rpcId);
    Expect(release.rpcId < ack.rpcId, "ACK replenishment starved request release/cancellation");
    Expect(ack.consumedSequence == 2, "Release did not discard the already buffered contiguous prefix");
    fixture.transport->Token(a, 3);
    fixture.transport->Token(a, 4);
    fixture.transport->Finish(a, 4, true, wire::RequestStatus::Cancelled, wire::EndReason::Cancelled);
    fixture.transport->Token(b, 1);
    fixture.transport->Finish(b, 1);
    ReadToken(requestB, 1);
    ReadTerminal(requestB);
    fixture.Acked(requestA, 5);
    fixture.Acked(requestB, 2);
    fixture.client->Query();
    const auto state = requestA->Snapshot();
    Expect(state.released && state.deliveryAbandoned && state.deliveredTokens == 0 &&
           state.discardedPublishedTokens == 4 && state.completionReady && !state.localFailure,
           "Released in-flight stream did not privately drain every published token");
    Expect(fixture.client->Snapshot().outstandingRequests == 0 && fixture.client->Snapshot().retainedRequestHandles == 1,
           "Release returned transport quota without removing its retained route");
    ExpectError([requestA = requestA] { requestA->Read(0ms); }, ClientErrorCode::InvalidHandle);
    ExpectError([requestA = requestA] { requestA->Cancel(); }, ClientErrorCode::InvalidHandle);
    const auto before = Count(fixture.transport->History(), wire::Method::HandleRelease);
    requestA->Release();
    fixture.client->Query();
    Expect(Count(fixture.transport->History(), wire::Method::HandleRelease) == before, "Duplicate Release sent another RPC");
    fixture.Close();
}

void ParentOwnersAndReleaseOrdering() {
    Fixture fixture({}, 0);
    fixture.transport->BlockNext(wire::Method::WorkerQuery);
    auto query = std::async(std::launch::async, [&] { return fixture.client->Query(); });
    fixture.transport->Take(wire::Method::WorkerQuery);
    auto create = std::async(std::launch::async, [&] { return fixture.model->CreateSession(); });
    Until([&] { return fixture.client->Snapshot().queuedCommands == 1; }, "Create did not enter the blocked sender queue");
    fixture.model->Release();
    fixture.transport->Unblock();
    const auto createCommand = fixture.transport->Take(wire::Method::SessionCreate);
    const auto releaseCommand = fixture.transport->Take(wire::Method::HandleRelease, fixture.model->Info().modelId);
    Expect(createCommand.rpcId < releaseCommand.rpcId, "Parent release overtook an already queued create operation");
    Get(query);
    fixture.sessions.push_back(Get(create));
    auto [request, submit] = fixture.Submit(0, 1);
    fixture.sessions[0]->Release();
    fixture.transport->Take(wire::Method::HandleRelease, fixture.sessions[0]->Info().sessionId);
    fixture.client.reset();
    Expect(Count(fixture.transport->History(), wire::Method::WorkerShutdown) == 0,
           "A surviving request lost its connection when parents were released");
    fixture.transport->Token(submit, 1);
    fixture.transport->Finish(submit, 1);
    ReadToken(request, 1);
    ReadTerminal(request);
    fixture.Acked(request, 2);
    request->Release();
    fixture.transport->WaitStopped();
    Expect(!fixture.transport->Forced(), "Last child owner could not shut down gracefully");
    CheckShutdownOrder(fixture.transport->History());
}

void ReleasedParentWhileCreateIsPending() {
    Fixture fixture({}, 0);
    fixture.transport->Hold(wire::Method::SessionCreate);
    auto create = std::async(std::launch::async, [&] { return fixture.model->CreateSession(); });
    const auto command = fixture.transport->Take(wire::Method::SessionCreate);
    fixture.client.reset();
    fixture.model->Release();
    fixture.transport->Take(wire::Method::HandleRelease);
    Expect(Count(fixture.transport->History(), wire::Method::WorkerShutdown) == 0,
           "Pending child creation did not retain an operation owner");
    fixture.transport->ReplyHeld(command);
    auto session = Get(create);
    Expect(session->Query().modelId == fixture.model->Info().modelId, "Created child lost its released parent resources");
    session->Release();
    fixture.transport->WaitStopped();
    Expect(!fixture.transport->Forced(), "Child created during parent release did not shut down cleanly");
}

void RequestAndRpcTimeouts() {
    Fixture fixture;
    auto [request, submit] = fixture.Submit(0, 1);
    Expect(request->Read(1ms).status == InferenceReadStatus::Timeout, "An empty Read should time out locally");
    fixture.transport->Hold(wire::Method::RequestQuery);
    ExpectError([request = request] { request->Query(20ms); }, ClientErrorCode::Timeout);
    const auto query = fixture.transport->Take(wire::Method::RequestQuery, submit.rpcId);
    Expect(!request->Snapshot().deliveryAbandoned && !fixture.transport->Forced(), "Query timeout abandoned or killed its stream");
    fixture.transport->ProcessedResponse(fixture.transport->ReplyHeld(query));
    fixture.transport->Token(submit, 1);
    fixture.transport->Finish(submit, 1);
    ReadToken(request, 1);
    ReadTerminal(request);
    fixture.Acked(request, 2);

    InferenceRequestOptions options;
    options.generation.maxNewTokens = 2;
    ExpectError([&] { fixture.sessions[1]->Submit({21, 22}, options, 20ms); }, ClientErrorCode::Timeout);
    const auto late = fixture.transport->Take(wire::Method::RequestSubmit);
    Expect(fixture.client->Snapshot().outstandingRequests == 1, "Timed-out submit prematurely returned its reserved route");
    fixture.transport->ReplyHeld(late);
    fixture.transport->Take(wire::Method::HandleRelease, late.rpcId);
    fixture.transport->Token(late, 1);
    fixture.transport->Finish(late, 1, false, wire::RequestStatus::Cancelled, wire::EndReason::Cancelled);
    Until([&] { return fixture.client->Snapshot().outstandingRequests == 0 || fixture.transport->Stopped(); },
          "Late accepted request was not privately drained");
    Expect(!fixture.transport->Forced(), "Late response after a local timeout killed Worker");
    fixture.Close();
}

void LateModelCreationAndSubmitRejection() {
    auto transport = std::make_shared<ControlledTransport>();
    auto client = InferenceClient::Connect(transport, ExpectedIdentity());
    transport->Hold(wire::Method::ModelLoad);
    ExpectError([&] { client->LoadQwen3("C:\\fake-qwen3", {}, 20ms); }, ClientErrorCode::Timeout);
    const auto load = transport->Take(wire::Method::ModelLoad);
    transport->ReplyHeld(load);
    const auto release = transport->Take(wire::Method::HandleRelease);
    Expect(release.objectType == wire::ObjectType::Model, "Late created model was not released");
    client->Shutdown(2s);
    Expect(!transport->Forced(), "Late model release poisoned shutdown");

    Fixture fixture;
    InferenceRequestOptions options;
    options.generation.maxNewTokens = 1;
    auto submit = std::async(std::launch::async, [&] { return fixture.sessions[0]->Submit({1}, options); });
    const auto command = fixture.transport->Take(wire::Method::RequestSubmit);
    fixture.transport->RejectSubmit(command);
    ExpectError([&] { Get(submit); }, ClientErrorCode::RemoteError, wire::ErrorCode::Busy);
    const auto state = fixture.client->Snapshot();
    Expect(state.outstandingRequests == 0 && state.retainedRequestHandles == 0 && state.reservedTokens == 0 && state.transportBytes == 0,
           "Rejected submit did not roll back all reservations");
    fixture.Close();
}

void RetainedAndUnreadAdmissionLimits() {
    InferenceClientOptions options;
    options.limits.maxActiveRequests = 1;
    options.limits.maxOutstandingRequests = 1;
    options.limits.maxRetainedRequestHandles = 2;
    Fixture fixture(options);
    auto [requestA, a] = fixture.Submit(0, 1);
    fixture.transport->Token(a, 1);
    fixture.transport->Finish(a, 1);
    Until([requestA = requestA] { return requestA->Snapshot().completionReady; }, "Completed unread A did not reach completion");
    ExpectError([&] { fixture.sessions[1]->Submit({1}); }, ClientErrorCode::ResourceLimit);
    Expect(fixture.client->Snapshot().outstandingRequests == 1, "Completed unread stream returned its transport reservation early");
    ReadToken(requestA, 1);
    ReadTerminal(requestA);
    fixture.Acked(requestA, 2);
    auto [requestB, b] = fixture.Submit(1, 1);
    fixture.transport->Token(b, 1);
    fixture.transport->Finish(b, 1);
    ReadToken(requestB, 1);
    ReadTerminal(requestB);
    fixture.Acked(requestB, 2);
    Expect(fixture.client->Snapshot().retainedRequestHandles == 2, "Consumed terminal lost its retained handle reservation");
    ExpectError([&] { fixture.sessions[0]->Submit({1}); }, ClientErrorCode::ResourceLimit);
    requestA->Release();
    fixture.transport->Take(wire::Method::HandleRelease, a.rpcId);
    Until([&] { return fixture.client->Snapshot().retainedRequestHandles == 1; }, "Released completed handle was not collected");
    auto [requestC, c] = fixture.Submit(0, 1);
    fixture.transport->Token(c, 1);
    fixture.transport->Finish(c, 1);
    ReadToken(requestC, 1);
    ReadTerminal(requestC);
    fixture.Close();
}

void ByteAndControlAdmissionLimits() {
    for (const bool byteLimit : {false, true}) {
        InferenceClientOptions options;
        options.limits.maxPendingCommands = byteLimit ? 16 : 3;
        options.limits.reservedControlCommands = 1;
        if (byteLimit) {
            options.limits.maxHeaderBytes = 8192;
            options.limits.maxIngressBytes = 64 * 1024;
        }
        Fixture fixture(options, 0);
        fixture.transport->Hold(wire::Method::SessionCreate);
        auto first = std::async(std::launch::async, [&] { return fixture.model->CreateSession(); });
        const auto command1 = fixture.transport->Take(wire::Method::SessionCreate);
        std::future<std::shared_ptr<InferenceSession>> second;
        wire::Command command2;
        if (!byteLimit) {
            second = std::async(std::launch::async, [&] { return fixture.model->CreateSession(); });
            command2 = fixture.transport->Take(wire::Method::SessionCreate);
        }
        ExpectError([&] { fixture.model->CreateSession(); }, ClientErrorCode::ResourceLimit);
        Expect(fixture.client->Query().domainStatus == wire::DomainStatus::Ready, "Normal commands consumed the reserved control count/bytes");
        const auto state = fixture.client->Snapshot();
        Expect(state.peakIngressBytes <= options.limits.maxIngressBytes && state.pendingCommands == (byteLimit ? 1 : 2),
               "Pending command or ingress accounting exceeded its bound");
        fixture.transport->ReplyHeld(command1);
        fixture.sessions.push_back(Get(first));
        if (second.valid()) {
            fixture.transport->ReplyHeld(command2);
            fixture.sessions.push_back(Get(second));
        }
        fixture.Close();
    }

    for (const bool tokenBudget : {false, true}) {
        InferenceClientOptions options;
        options.limits.tokenWindow = 2;
        if (tokenBudget) options.limits.maxOutstandingTokens = 6;
        else options.limits.maxTransportBytes = 3 * options.limits.maxTokenFrameBytes +
            options.limits.maxTerminalFrameBytes + options.limits.maxFinalFrameBytes;
        Fixture fixture(options);
        auto [request, submit] = fixture.Submit(0, 4);
        InferenceRequestOptions requestOptions;
        requestOptions.generation.maxNewTokens = 1;
        ExpectError([&] { fixture.sessions[1]->Submit({1}, requestOptions); }, ClientErrorCode::ResourceLimit);
        const auto state = fixture.client->Snapshot();
        Expect(state.peakTransportBytes <= options.limits.maxTransportBytes && state.reservedTokens == 6,
               "Token/transport admission exceeded its reservation");
        fixture.Close();
        ReadTerminal(request);
    }
}

void ProtocolFaultsDeliverOneLocalTerminal() {
    for (int variant = 0; variant < 6; ++variant) {
        InferenceClientOptions options;
        options.limits.tokenWindow = 2;
        Fixture fixture(options);
        auto [requestA, a] = fixture.Submit(0, 4);
        auto [requestB, b] = fixture.Submit(1, 4);
        fixture.transport->ProcessedEvent(fixture.transport->Token(a, 1));
        ReadToken(requestA, 1);
        if (variant == 0) fixture.transport->Token(a, 1); // Repeated sequence.
        if (variant == 1) fixture.transport->Token(a, 3); // Skipped sequence.
        if (variant == 2) {
            const auto summary = Summary(a, 1, wire::RequestStatus::Completed, wire::EndReason::StopToken);
            fixture.transport->ProcessedEvent(fixture.transport->Terminal(a, summary));
            fixture.transport->Terminal(a, summary); // Duplicate terminal before final.
        }
        if (variant == 3) {
            fixture.transport->Final(a, Summary(a, 1, wire::RequestStatus::Completed, wire::EndReason::StopToken));
            fixture.transport->Terminal(a, Summary(a, 1, wire::RequestStatus::Cancelled, wire::EndReason::Cancelled));
        }
        if (variant == 4 || variant == 5) {
            wire::Event event;
            event.identity = fixture.transport->Identity(a.rpcId);
            event.sequence = 2;
            event.tokenId = 102;
            event.summary.status = wire::RequestStatus::Decode;
            event.summary.promptTokensProcessed = a.promptIds.size();
            event.summary.generatedTokens = 2;
            if (variant == 4) event.epoch = "fedcba9876543210fedcba9876543210";
            else ++event.identity.modelId;
            fixture.transport->Event(std::move(event));
        }
        fixture.transport->WaitStopped();
        Expect(fixture.transport->Forced(), "Malformed stream did not fail its connection");
        const auto first = ReadTerminal(requestA, true);
        const auto second = ReadTerminal(requestB, true);
        Expect(first.sequence == 0 && first.summary.eventCount == 0 && first.deliveredTokens == 1 &&
               second.sequence == 0 && second.summary.eventCount == 0 && second.deliveredTokens == 0,
               "Connection fault invented wire sequence/counts or forgot delivered tokens");
        Expect(fixture.client->Snapshot().status == wire::DomainStatus::Faulted &&
               fixture.client->Snapshot().transportBytes == 0 && fixture.client->Snapshot().ingressBytes == 0,
               "Protocol failure leaked host reservations");
    }
}

void WindowViolationAndAlreadyConsumedTerminal() {
    {
        InferenceClientOptions options;
        options.limits.tokenWindow = 2;
        Fixture fixture(options);
        auto [request, command] = fixture.Submit(0, 4);
        fixture.transport->Token(command, 1);
        fixture.transport->Token(command, 2);
        fixture.transport->Token(command, 3);
        fixture.transport->WaitStopped();
        const auto terminal = ReadTerminal(request, true);
        Expect(terminal.discardedPublishedTokens == 2 && terminal.deliveredTokens == 0,
               "Window violation silently expanded the fixed token ring");
    }
    {
        Fixture fixture;
        auto [requestA, a] = fixture.Submit(0, 1);
        auto [requestB, b] = fixture.Submit(1, 1);
        fixture.transport->Token(a, 1);
        fixture.transport->Finish(a, 1);
        ReadToken(requestA, 1);
        ReadTerminal(requestA);
        fixture.transport->Terminal(a, Summary(a, 1));
        fixture.transport->WaitStopped();
        Expect(requestA->Read(0ms).status == InferenceReadStatus::Closed, "Consumed normal completion gained a second local terminal");
        ReadTerminal(requestB, true);
    }
}

void DomainFaultAllowsDrainAndStopsAdmission() {
    Fixture fixture;
    auto [request, submit] = fixture.Submit(0, 1);
    wire::Event fault;
    fault.kind = wire::EventKind::DomainFault;
    fault.errorCode = wire::ErrorCode::RuntimeFault;
    fault.message = "Injected device fault";
    fixture.transport->ProcessedEvent(fixture.transport->Event(std::move(fault)));
    ExpectError([&] { fixture.sessions[1]->Submit({1}); }, ClientErrorCode::RemoteError, wire::ErrorCode::RuntimeFault);
    Expect(request->Query().summary.status == wire::RequestStatus::Queued, "Domain fault disabled cleanup/query routing");
    auto summary = Summary(submit, 0, wire::RequestStatus::Failed, wire::EndReason::RuntimeFault);
    summary.errorCode = wire::ErrorCode::RuntimeFault;
    summary.message = "Injected device fault";
    fixture.transport->Terminal(submit, summary);
    fixture.transport->Final(submit, summary);
    const auto terminal = ReadTerminal(request);
    Expect(terminal.summary.errorCode == wire::ErrorCode::RuntimeFault && !terminal.localFailure,
           "Domain fault replaced a valid Worker terminal with a fabricated local completion");
    fixture.Acked(request, 1);
    fixture.Close();
}

void ShutdownDiscardsAndHasOneSharedResult() {
    Fixture fixture({}, 3);
    auto [requestA, a] = fixture.Submit(0, 4);
    auto [requestB, b] = fixture.Submit(1, 2);
    auto [requestC, c] = fixture.Submit(2, 4);
    fixture.transport->Token(a, 1);
    fixture.transport->Token(b, 1);
    fixture.transport->Token(b, 2);
    fixture.transport->Finish(b, 2);
    Until([requestA = requestA, requestB = requestB] { return requestB->Snapshot().completionReady && requestA->Snapshot().bufferedTokens == 1; },
          "Initial unread streams were not routed");
    std::promise<void> start;
    auto gate = start.get_future().share();
    auto first = std::async(std::launch::async, [&] { gate.wait(); fixture.client->Shutdown(2s); });
    auto second = std::async(std::launch::async, [&] { gate.wait(); fixture.client->Shutdown(2s); });
    start.set_value();
    Get(first);
    Get(second);
    const auto completed = ReadTerminal(requestB);
    const auto active = ReadTerminal(requestA);
    const auto queued = ReadTerminal(requestC);
    Expect(completed.summary.status == wire::RequestStatus::Completed && completed.deliveryAbandoned &&
           completed.deliveredTokens == 0 && completed.discardedPublishedTokens == 2,
           "Shutdown rewrote Completed or claimed discarded tokens were delivered");
    Expect(active.summary.endReason == wire::EndReason::SchedulerStopped && active.discardedPublishedTokens == 1 &&
           queued.summary.endReason == wire::EndReason::SchedulerStopped && queued.discardedPublishedTokens == 0,
           "Shutdown did not drain queued and active requests independently");
    requestA->Release();
    fixture.model->Release();
    fixture.sessions[0]->Release();
    CheckShutdownOrder(fixture.transport->History());
    fixture.Close();
}

void ShutdownCrossPipeFenceAndEof() {
    for (bool responseFirst : {false, true}) {
        Fixture fixture;
        fixture.transport->AutoShutdown(false);
        auto close = std::async(std::launch::async, [&] { fixture.client->Shutdown(2s); });
        const auto shutdown = fixture.transport->Take(wire::Method::WorkerShutdown);
        if (responseFirst) {
            fixture.transport->ShutdownReply(shutdown, 2);
            fixture.transport->ClosePipe(false);
            fixture.transport->ProcessedEvent(fixture.transport->Log("first fence event"));
            Expect(close.wait_for(0ms) == std::future_status::timeout, "Shutdown ignored its final event ordinal");
            fixture.transport->Log("last fence event");
            fixture.transport->ClosePipe(true);
        } else {
            fixture.transport->ProcessedEvent(fixture.transport->Log("event before response EOF"));
            fixture.transport->ClosePipe(true);
            fixture.transport->WaitEventEof();
            fixture.transport->ShutdownReply(shutdown, 1);
        }
        Get(close);
        const auto state = fixture.client->Snapshot();
        Expect(state.receivedEventOrdinal == (responseFirst ? 2 : 1) && state.droppedLogs == 0 &&
               state.queuedLogs == state.receivedEventOrdinal,
               "Shutdown truncated its independent event pipe");
        for (uint64_t i = 1; i <= state.receivedEventOrdinal; ++i) {
            const auto log = fixture.client->ReadLog(0ms);
            Expect(log.status == InferenceReadStatus::Event && log.event->eventOrdinal == i,
                   "Shutdown lost a queued log before the close barrier");
        }
        Expect(fixture.client->ReadLog(0ms).status == InferenceReadStatus::Closed, "Closed log stream did not end");
        fixture.Close();
    }
}

void BoundedLogsAndIndependentInference() {
    InferenceClientOptions options;
    options.limits.maxLogQueueBytes = 2048;
    Fixture fixture(options);
    Expect(fixture.client->ReadLog(0ms).status == InferenceReadStatus::Timeout, "Empty live log stream did not time out");
    const std::string message = u8"层执行：开始 α\n";
    fixture.transport->ProcessedEvent(fixture.transport->Log(message));
    const auto first = fixture.client->ReadLog(0ms);
    Expect(first.status == InferenceReadStatus::Event && first.event->message == message && first.event->level == 2 &&
           first.event->eventOrdinal == 1 && first.event->epoch == fixture.client->Snapshot().epoch && !first.droppedLogs,
           "Log content, UTF-8, level or identity changed");
    auto [request, submit] = fixture.Submit(0, 2);
    uint64_t serial = 0;
    for (unsigned i = 0; i < 80; ++i) serial = fixture.transport->Log("burst-" + std::to_string(i));
    fixture.transport->ProcessedEvent(serial);
    const auto flooded = fixture.client->Snapshot();
    Expect(flooded.queuedLogs && flooded.droppedLogs && flooded.queuedLogs + flooded.droppedLogs == 80 &&
           flooded.logQueueBytes <= options.limits.maxLogQueueBytes &&
           flooded.peakLogQueueBytes <= options.limits.maxLogQueueBytes, "Log overflow was not bounded/accounted");
    fixture.transport->Token(submit, 1, 201);
    fixture.transport->Token(submit, 2, 202);
    fixture.transport->Finish(submit, 2);
    Expect(ReadToken(request, 1).tokenId == 201 && ReadToken(request, 2).tokenId == 202,
           "A full log queue lost or blocked token delivery");
    ReadTerminal(request);
    fixture.Acked(request, 3);
    Expect(fixture.client->Query().accepting, "Log overflow blocked control traffic");
    for (uint64_t i = 0; i < flooded.queuedLogs; ++i) {
        const auto log = fixture.client->ReadLog(0ms);
        Expect(log.status == InferenceReadStatus::Event && log.event->message == "burst-" + std::to_string(i) &&
               log.droppedLogs == flooded.droppedLogs, "Overflow replaced older logs or changed the drop counter");
    }
    Expect(fixture.client->Snapshot().logQueueBytes == 0, "Consumed log quota was not returned");
    std::promise<void> started;
    auto pending = std::async(std::launch::async, [&] { started.set_value(); return fixture.client->ReadLog(2s); });
    started.get_future().wait();
    fixture.client->CloseLogStream();
    Expect(Get(pending).status == InferenceReadStatus::Closed, "Log consumer release did not wake its reader");
    fixture.transport->ProcessedEvent(fixture.transport->Log("after log consumer release"));
    Expect(fixture.client->Snapshot().droppedLogs == flooded.droppedLogs + 1 && fixture.client->Query().accepting,
           "Released log consumer retained new logs or stopped the connection");
    fixture.Close();
}

void WorkerLogLevelNegotiationAndEcho() {
    {
        Fixture oldPeer;
        ExpectError([&] { oldPeer.client->SetLogLevel(0); }, ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(oldPeer.transport->History(), wire::Method::WorkerSetLogLevel) == 0 && oldPeer.client->Query().accepting,
               "Old peer received an unsupported log command or lost its connection");
        oldPeer.Close();
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    Fixture fixture({}, 1, hello);
    for (uint32_t level = 0; level <= 3; ++level) {
        fixture.client->SetLogLevel(level);
        Expect(fixture.transport->Take(wire::Method::WorkerSetLogLevel).logLevel == level, "Wrong log control dispatch");
    }
    ExpectError([&] { fixture.client->SetLogLevel(4); }, ClientErrorCode::RemoteError, wire::ErrorCode::InvalidArgument);
    Expect(Count(fixture.transport->History(), wire::Method::WorkerSetLogLevel) == 4, "Invalid log level emitted an RPC");
    fixture.transport->Hold(wire::Method::WorkerSetLogLevel);
    auto setting = std::async(std::launch::async, [&] { fixture.client->SetLogLevel(0); });
    const auto command = fixture.transport->Take(wire::Method::WorkerSetLogLevel);
    auto response = fixture.transport->Base(command);
    response.logLevel = 1;
    fixture.transport->Reply(std::move(response));
    ExpectError([&] { Get(setting); }, ClientErrorCode::ProtocolError);
    fixture.transport->WaitStopped();
}

void BrokenWriterAndProcessExitFailure() {
    {
        InferenceClientOptions options;
        options.limits.connectionWriteTimeoutMs = 100;
        Fixture fixture(options);
        auto [request, submit] = fixture.Submit(0, 4);
        fixture.transport->BlockNext(wire::Method::WorkerQuery);
        auto query = std::async(std::launch::async, [&] { return fixture.client->Query(); });
        fixture.transport->Take(wire::Method::WorkerQuery);
        ExpectError([&] { fixture.client->Shutdown(300ms); }, ClientErrorCode::Closed);
        ExpectError([&] { Get(query); }, ClientErrorCode::Closed);
        Expect(fixture.transport->Forced(), "A blocked connection writer was reported as graceful shutdown");
        ReadTerminal(request, true);
    }
    for (bool throws : {false, true}) {
        Fixture fixture;
        fixture.transport->BadExit(throws);
        ExpectError([&] { fixture.client->Shutdown(2s); }, ClientErrorCode::Closed, wire::ErrorCode::TransportError);
        Expect(fixture.transport->Forced() && fixture.transport->WaitedForExit() &&
               fixture.client->Snapshot().status == wire::DomainStatus::Faulted,
               "Exit failure/exception was swallowed after the close barrier");
    }
}

std::shared_ptr<InferenceModel> LoadModel(const std::shared_ptr<InferenceClient>& client,
                                         const std::string& backend, InferenceModelOptions options = {}) {
    if (backend == "qwen3_nf4") return client->LoadQwen3("C:\\fake-qwen3", options);
    if (backend == "qwen35_nf4") return client->LoadQwen35("C:\\fake-hybrid", options);
    if (backend == "qwen35_bf16") return client->LoadQwen35Bf16("C:\\fake-hybrid-bf16", options);
    throw std::invalid_argument("Unknown fixture backend");
}

void HybridModelNegotiationAndLoading(bool bf16) {
    const std::string backend = bf16 ? "qwen35_bf16" : "qwen35_nf4";
    const std::string capability = bf16 ? "qwen35_bf16_bf16_kv" : "qwen35_nf4_bf16_kv";
    {
        // The default handshake must still work with a Qwen3-only v2 peer.
        auto transport = std::make_shared<ControlledTransport>();
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        ExpectError([&] { LoadModel(client, backend); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::ModelLoad) == 0,
               "Unsupported Hybrid load consumed a wire command or model slot");
        auto model = client->LoadQwen3("C:\\fake-qwen3");
        Expect(client->Query().accepting && !transport->Forced(),
               "Missing optional model capability poisoned the connection");
        client->Shutdown(2s);
    }
    {
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        InferenceModelOptions options;
        options.maxSequenceLength = 512;
        options.maxSessions = 2;
        options.maxSessionCacheBytes = 8192;
        auto model = LoadModel(client, backend, options);
        const auto command = transport->Take(wire::Method::ModelLoad);
        Expect(command.backend == backend && command.kvType == "bf16" &&
                   command.maxSequenceLength == 512 && command.maxSessions == 2 && command.maxSessionCacheBytes == 8192,
               "Hybrid model load options changed in transit");
        const auto& capabilities = model->Info().capabilities;
        Expect(std::find(capabilities.begin(), capabilities.end(), capability) != capabilities.end() &&
                   std::find(capabilities.begin(), capabilities.end(), "qwen3_nf4_bf16_kv") == capabilities.end(),
               "Model capabilities describe a different backend");
        auto session = model->CreateSession();
        Expect(session->Query().contextLength == 0 && session->Info().capacity == 512,
               "Hybrid model did not create an independent empty session");
        client->Shutdown(2s);
        Expect(!transport->Forced(), "Hybrid client did not shut down gracefully");
    }
    {
        // Hello offers Hybrid, but configure may select a smaller feature set.
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto transport = std::make_shared<ControlledTransport>(hello);
        transport->Hold(wire::Method::WorkerConfigure);
        auto connected = std::async(std::launch::async, [&] { return InferenceClient::Connect(transport, ExpectedIdentity()); });
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        auto response = transport->Base(command);
        response.limits = command.limits;
        response.capabilities = wire::default_capabilities();
        response.worker.domainStatus = wire::DomainStatus::Ready;
        transport->Reply(std::move(response));
        auto client = Get(connected);
        ExpectError([&] { LoadModel(client, backend); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::ModelLoad) == 0,
               "Client used hello instead of the configured model capabilities");
        auto model = client->LoadQwen3("C:\\fake-qwen3");
        client->Shutdown(2s);
    }
    if (bf16) {
        // A Worker with Hybrid NF4 support still cannot load BF16 weights.
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        hello.capabilities.erase(std::remove(hello.capabilities.begin(), hello.capabilities.end(), capability),
                                 hello.capabilities.end());
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        ExpectError([&] { LoadModel(client, backend); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::ModelLoad) == 0,
               "BF16 load was sent to an NF4-only Worker");
        auto model = client->LoadQwen35("C:\\fake-hybrid");
        Expect(client->Query().accepting && !transport->Forced(), "Rejected BF16 load broke NF4 compatibility");
        client->Shutdown(2s);
    }
}

void ModelCapabilityMismatchAndConfigureConsistency() {
    for (const std::string backend : {"qwen3_nf4", "qwen35_nf4", "qwen35_bf16"}) {
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        transport->Hold(wire::Method::ModelLoad);
        auto loaded = std::async(std::launch::async, [&] {
            return LoadModel(client, backend);
        });
        const auto command = transport->Take(wire::Method::ModelLoad);
        auto response = transport->Base(command);
        response.model = {1, 2048, command.maxSequenceLength, 4096,
            {backend == "qwen35_nf4" ? "qwen35_bf16_bf16_kv" : "qwen35_nf4_bf16_kv",
             "token_requests_v1", "session_lifecycle_v1"}};
        transport->Reply(std::move(response));
        ExpectError([&] { Get(loaded); }, ClientErrorCode::ProtocolError);
        transport->WaitStopped();
        Expect(transport->Forced(), "Client accepted a model from the wrong backend");
    }
    {
        auto transport = std::make_shared<ControlledTransport>();
        transport->Hold(wire::Method::WorkerConfigure);
        auto connected = std::async(std::launch::async, [&] { return InferenceClient::Connect(transport, ExpectedIdentity()); });
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        auto response = transport->Base(command);
        response.limits = command.limits;
        response.capabilities = wire::supported_capabilities();
        response.worker.domainStatus = wire::DomainStatus::Ready;
        transport->Reply(std::move(response));
        ExpectError([&] { Get(connected); }, ClientErrorCode::ProtocolError);
        Expect(transport->Forced(), "Configure published a capability absent from hello");
    }
}

void ExtendedGenerationNegotiation() {
    InferenceClientOptions options;
    options.limits.maxGeneratedTokens = 2048;
    {
        auto transport = std::make_shared<ControlledTransport>();
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity(), options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::WorkerConfigure) == 0 && transport->Stopped(),
               "Extended configuration reached an old peer or leaked its connection");
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    for (const bool changeLimit : {false, true}) {
        auto transport = std::make_shared<ControlledTransport>(hello);
        transport->Hold(wire::Method::WorkerConfigure);
        auto connected = std::async(std::launch::async, [&] {
            return InferenceClient::Connect(transport, ExpectedIdentity(), options);
        });
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        Expect(command.limits.maxGeneratedTokens == 2048 &&
                   std::find(command.requiredCapabilities.begin(), command.requiredCapabilities.end(), "extended_generation_v1") != command.requiredCapabilities.end(),
               "Explicit generation limit lost its required capability");
        auto response = transport->Base(command);
        response.limits = command.limits;
        response.capabilities = hello.capabilities;
        response.worker.domainStatus = wire::DomainStatus::Ready;
        if (changeLimit) response.limits.maxGeneratedTokens = 1024;
        transport->Reply(std::move(response));
        if (changeLimit) {
            ExpectError([&] { Get(connected); }, ClientErrorCode::ProtocolError);
            Expect(transport->Stopped(), "Silently reduced limits did not close the connection");
        } else {
            auto client = Get(connected);
            client->Shutdown(2s);
        }
    }
    auto transport = std::make_shared<ControlledTransport>(hello);
    auto client = InferenceClient::Connect(transport, ExpectedIdentity());
    const auto ordinary = transport->Take(wire::Method::WorkerConfigure);
    Expect(ordinary.limits.maxGeneratedTokens == 1024 &&
               std::find(ordinary.requiredCapabilities.begin(), ordinary.requiredCapabilities.end(), "extended_generation_v1") == ordinary.requiredCapabilities.end(),
           "Ordinary callers changed their configuration contract");
    client->Shutdown(2s);
}

void DecodeBatchingNegotiationAndEcho() {
    for (uint32_t size : {uint32_t{0}, uint32_t{5}, uint32_t{0xffffffff}}) {
        auto transport = std::make_shared<ControlledTransport>();
        InferenceClientOptions options;
        options.maxDecodeBatchSize = size;
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity(), options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::InvalidArgument);
        Expect(transport->History().empty(), "Invalid decode batch size consumed a wire command");
    }
    {
        auto transport = std::make_shared<ControlledTransport>();
        InferenceClientOptions options;
        options.limits.maxActiveRequests = 1;
        options.maxDecodeBatchSize = 2;
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity(), options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::InvalidArgument);
        Expect(transport->History().empty(), "Decode batching bypassed the selected active bound");
    }
    {
        auto transport = std::make_shared<ControlledTransport>();
        InferenceClientOptions options;
        options.maxDecodeBatchSize = 2;
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity(), options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::WorkerConfigure) == 0 && transport->Stopped(),
               "Batching configuration reached an old Worker");
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    for (uint32_t size : {uint32_t{1}, uint32_t{2}, uint32_t{4}}) {
        auto transport = std::make_shared<ControlledTransport>(hello);
        InferenceClientOptions options;
        options.maxDecodeBatchSize = size;
        // Explicit callers must not receive a duplicate required capability.
        if (size == 4) options.requiredCapabilities.emplace_back("continuous_batching_v1");
        auto client = InferenceClient::Connect(transport, ExpectedIdentity(), options);
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        Expect(command.maxDecodeBatchSize == size &&
               std::count(command.requiredCapabilities.begin(), command.requiredCapabilities.end(),
                          "continuous_batching_v1") == (size > 1 ? 1 : 0),
               "Decode batching opt-in or its required capability changed");
        Expect(command.limits.maxActiveRequests == options.limits.maxActiveRequests &&
               command.limits.maxOutstandingTokens == options.limits.maxOutstandingTokens &&
               command.limits.maxSessions == options.limits.maxSessions,
               "Decode batching silently widened admission limits");
        client->Shutdown(2s);
        Expect(!transport->Forced(), "Batching configuration prevented graceful shutdown");
    }
    for (const auto sizes : {std::pair<uint32_t, uint32_t>{4, 1}, {2, 3}, {1, 2}}) {
        auto transport = std::make_shared<ControlledTransport>(hello);
        transport->Hold(wire::Method::WorkerConfigure);
        InferenceClientOptions options;
        options.maxDecodeBatchSize = sizes.first;
        auto connected = std::async(std::launch::async, [&] {
            return InferenceClient::Connect(transport, ExpectedIdentity(), options);
        });
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        auto response = transport->Base(command);
        response.limits = command.limits;
        response.capabilities = hello.capabilities;
        response.maxDecodeBatchSize = sizes.second;
        response.worker.domainStatus = wire::DomainStatus::Ready;
        transport->Reply(std::move(response));
        ExpectError([&] { Get(connected); }, ClientErrorCode::ProtocolError);
        Expect(transport->Stopped(), "Changed or unsolicited batch acknowledgement did not close transport");
    }
    {
        auto transport = std::make_shared<ControlledTransport>(hello);
        transport->Hold(wire::Method::WorkerConfigure);
        InferenceClientOptions options;
        options.maxDecodeBatchSize = 2;
        auto connected = std::async(std::launch::async, [&] {
            return InferenceClient::Connect(transport, ExpectedIdentity(), options);
        });
        const auto command = transport->Take(wire::Method::WorkerConfigure);
        auto response = transport->Base(command);
        response.limits = command.limits;
        response.capabilities = wire::default_capabilities();
        response.worker.domainStatus = wire::DomainStatus::Ready;
        transport->Reply(std::move(response));
        ExpectError([&] { Get(connected); }, ClientErrorCode::ProtocolError);
        Expect(transport->Stopped(), "Configure removed the required batching capability");
    }
}

void RequestPolicyNegotiation() {
    for (const std::string missing : {"host_sampling_v1", "thinking_budget_close_once_v1", "legacy_prefill_logits_v1"}) {
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto& capabilities = hello.capabilities;
        capabilities.erase(std::remove(capabilities.begin(), capabilities.end(), missing), capabilities.end());
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        auto model = client->LoadQwen3("C:\\fake-qwen3");
        auto session = model->CreateSession();
        InferenceRequestOptions options;
        options.forceHostSampling = true;
        options.legacyPrefillLogits = true;
        options.thinkingBudget = wire::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}, true};
        ExpectError([&] { session->Submit({1}, options); }, ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::RequestSubmit) == 0 && client->Query().accepting,
               "Unnegotiated policy reached the Worker or closed the connection");
        client->Shutdown(2s);
    }
}

void PrefillNegotiationAndRouting() {
    {
        Fixture oldPeer;
        ExpectError([&] { oldPeer.sessions[0]->Prefill({1, 2}, 2); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(oldPeer.transport->History(), wire::Method::RequestSubmit) == 0 && oldPeer.client->Query().accepting,
               "Unnegotiated replay reached the Worker or broke the connection");
        oldPeer.Close();
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    auto transport = std::make_shared<ControlledTransport>(hello);
    auto client = InferenceClient::Connect(transport, ExpectedIdentity());
    auto model = client->LoadQwen3("C:\\fake-qwen3");
    auto session = model->CreateSession();
    auto submitting = std::async(std::launch::async, [&] { return session->Prefill({1, 2, 3}, 2); });
    const auto command = transport->Take(wire::Method::RequestSubmit);
    Expect(command.prefillOnly && command.generation.maxNewTokens == 0 && command.prefillChunkSize == 2 &&
               command.promptIds == std::vector<int32_t>({1, 2, 3}), "Replay options changed before sending");
    transport->ReplyHeld(command);
    auto request = Get(submitting);
    transport->Finish(command, 0);
    const auto event = ReadTerminal(request);
    Expect(event.sequence == 1 && event.summary.generatedTokens == 0 &&
               event.summary.promptTokensProcessed == 3 && event.summary.endReason == wire::EndReason::Length,
           "Zero-token request did not deliver exactly its own terminal");
    client->Shutdown(2s);
    Expect(!transport->Forced(), "Replay prevented graceful shutdown");
}

void ThinkingBudgetNegotiationAndRouting() {
    InferenceRequestOptions options;
    options.generation.maxNewTokens = 3;
    options.thinkingBudget = wire::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}};
    {
        Fixture oldPeer;
        ExpectError([&] { oldPeer.sessions[0]->Submit({1}, options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(oldPeer.transport->History(), wire::Method::RequestSubmit) == 0 && oldPeer.client->Query().accepting,
               "Unnegotiated thinking budget reached the Worker or broke the connection");
        oldPeer.Close();
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    auto transport = std::make_shared<ControlledTransport>(hello);
    auto client = InferenceClient::Connect(transport, ExpectedIdentity());
    auto model = client->LoadQwen3("C:\\fake-qwen3");
    auto session = model->CreateSession();
    auto invalid = options;
    invalid.thinkingBudget->openTokenId = static_cast<int32_t>(model->Info().vocabSize);
    ExpectError([&] { session->Submit({1}, invalid); }, ClientErrorCode::RemoteError, wire::ErrorCode::InvalidArgument);
    Expect(Count(transport->History(), wire::Method::RequestSubmit) == 0, "Invalid thinking token reached the Worker");
    auto submitting = std::async(std::launch::async, [&] { return session->Submit({1}, options); });
    const auto command = transport->Take(wire::Method::RequestSubmit);
    Expect(command.thinkingBudget && command.thinkingBudget->maxTokens == 2 &&
               command.thinkingBudget->promptOpensThink && command.thinkingBudget->closeTokenIds == std::vector<int32_t>({11, 3}),
           "Thinking metadata was dropped or changed before sending");
    transport->ReplyHeld(command);
    auto request = Get(submitting);
    transport->Token(command, 1, 101);
    transport->Finish(command, 1); // Length may finish early when a complete close cannot fit.
    Expect(ReadToken(request, 1).tokenId == 101 && ReadTerminal(request).summary.endReason == wire::EndReason::Length,
           "Early budget length completion lost its own terminal");
    client->Shutdown(2s);
    Expect(!transport->Forced(), "Budget extension prevented graceful shutdown");
}

void AllocationLimitsNegotiationAndAcknowledgement() {
    InferenceModelOptions options;
    options.allocationLimits = wire::AllocationLimits{1024 * 1024, 65536, 8192};
    {
        auto transport = std::make_shared<ControlledTransport>();
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        ExpectError([&] { client->LoadQwen3("C:\\fake-qwen3", options); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(Count(transport->History(), wire::Method::ModelLoad) == 0 && client->Query().accepting,
               "Unnegotiated allocation limits reached the Worker or broke the connection");
        client->Shutdown(2s);
    }
    for (const std::string backend : {"qwen3_nf4", "qwen35_nf4", "qwen35_bf16"}) {
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        auto model = LoadModel(client, backend, options);
        const auto command = transport->Take(wire::Method::ModelLoad);
        Expect(command.allocationLimits && *command.allocationLimits == *options.allocationLimits &&
               model->Info().memory && model->Info().memory->limits == *options.allocationLimits,
               "Allocation limits changed across the command/result route");
        auto session = model->CreateSession();
        Expect(session->Query().contextLength == 0, "Budgeted model could not create a session");
        client->Shutdown(2s);
    }
    for (bool omit : {false, true}) {
        auto hello = Hello();
        hello.capabilities = wire::supported_capabilities();
        auto transport = std::make_shared<ControlledTransport>(hello);
        auto client = InferenceClient::Connect(transport, ExpectedIdentity());
        transport->Hold(wire::Method::ModelLoad);
        auto loading = std::async(std::launch::async, [&] { return client->LoadQwen3("C:\\fake-qwen3", options); });
        const auto command = transport->Take(wire::Method::ModelLoad);
        auto response = transport->Base(command);
        response.model = {1, 2048, command.maxSequenceLength, 4096,
            {"qwen3_nf4_bf16_kv", "token_requests_v1", "session_lifecycle_v1", "allocation_limits_v1"}};
        if (!omit) {
            response.model.memory = wire::ModelMemoryInfo{*options.allocationLimits, 0, 0};
            ++response.model.memory->limits.maxDeviceBytes;
        }
        transport->Reply(std::move(response));
        ExpectError([&] { Get(loading); }, ClientErrorCode::ProtocolError);
        transport->WaitStopped();
        Expect(transport->Forced(), "Worker silently omitted or changed requested allocation limits");
    }
}

void SessionContextNegotiationAndResponseValidation() {
    {
        Fixture oldPeer;
        const auto before = oldPeer.transport->History().size();
        ExpectError([&] { oldPeer.sessions[0]->Reset(); }, ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        ExpectError([&] { oldPeer.sessions[0]->Truncate(0); }, ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(oldPeer.transport->History().size() == before, "Unsupported context operations emitted RPCs");
        Expect(oldPeer.sessions[0]->Query().contextLength == 0, "Capability rejection broke old query");
        oldPeer.Close();
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    {
        Fixture fixture({}, 1, hello);
        auto session = fixture.sessions[0];
        ExpectError([&] { session->Truncate(UINT32_MAX); }, ClientErrorCode::RemoteError, wire::ErrorCode::InvalidArgument);
        Expect(Count(fixture.transport->History(), wire::Method::SessionTruncate) == 0,
               "Unrepresentable truncate target reached the peer");
        fixture.transport->Hold(wire::Method::SessionTruncate);
        auto truncate = std::async(std::launch::async, [&] { return session->Truncate(9); });
        auto command = fixture.transport->Take(wire::Method::SessionTruncate);
        Expect(command.sessionId == session->Info().sessionId && command.contextLength == 9,
               "Truncate changed Session or requested length");
        auto response = fixture.transport->Base(command);
        response.session = session->Info();
        response.session.contextLength = 4; // Independent peer checkpoint, below the target.
        fixture.transport->Reply(response);
        Expect(Get(truncate).contextLength == 4, "Client replaced actual checkpoint with requested length");
        Expect(session->Reset().contextLength == 0 && session->Query().contextLength == 0, "Reset/query disagreed");
        fixture.Close();
    }
    // Payloads are individually valid; only the correlation with the command
    // reveals the malicious identity, cache metadata or restoration length.
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
        Fixture fixture({}, 1, hello);
        auto session = fixture.sessions[0];
        fixture.transport->Hold(wire::Method::SessionTruncate);
        auto truncate = std::async(std::launch::async, [&] { return session->Truncate(4); });
        const auto command = fixture.transport->Take(wire::Method::SessionTruncate);
        auto response = fixture.transport->Base(command);
        response.session = session->Info();
        response.session.contextLength = 4;
        switch (mutation) {
        case 0: ++response.session.sessionId; break;
        case 1: ++response.session.modelId; break;
        case 2: ++response.session.capacity; break;
        case 3: ++response.session.cacheBytes; break;
        case 4: response.session.contextLength = 5; break;
        }
        fixture.transport->Reply(response);
        ExpectError([&] { Get(truncate); }, ClientErrorCode::ProtocolError);
        fixture.transport->WaitStopped();
        Expect(fixture.transport->Forced(), "Invalid context response left the connection usable");
    }
}

void SessionContextTimeoutAndRelease() {
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    {
        Fixture fixture({}, 2, hello);
        fixture.transport->Hold(wire::Method::SessionReset);
        ExpectError([&] { fixture.sessions[0]->Reset(20ms); }, ClientErrorCode::Timeout);
        const auto reset = fixture.transport->Take(wire::Method::SessionReset);
        auto [peer, command] = fixture.Submit(1, 1);
        fixture.transport->Token(command, 1);
        fixture.transport->Finish(command, 1);
        ReadToken(peer, 1);
        ReadTerminal(peer);
        fixture.Acked(peer, 2);
        fixture.transport->ProcessedResponse(fixture.transport->ReplyHeld(reset));
        Expect(fixture.sessions[0]->Query().contextLength == 0 && !fixture.transport->Stopped(),
               "Late reset response poisoned independent requests or query");
        fixture.Close();
    }
    {
        Fixture fixture({}, 1, hello);
        fixture.transport->Hold(wire::Method::SessionReset);
        auto session = fixture.sessions[0];
        auto resetting = std::async(std::launch::async, [&] { return session->Reset(); });
        const auto reset = fixture.transport->Take(wire::Method::SessionReset);
        session->Release();
        fixture.model->Release();
        fixture.client.reset();
        Expect(Count(fixture.transport->History(), wire::Method::WorkerShutdown) == 0,
               "Last-parent release dropped the pending context operation owner");
        fixture.transport->ReplyHeld(reset);
        Expect(Get(resetting).contextLength == 0, "Pending reset failed when its parent was released");
        fixture.transport->WaitStopped();
        Expect(!fixture.transport->Forced(), "Pending reset prevented graceful last-owner shutdown");
        ExpectError([&] { session->Reset(); }, ClientErrorCode::InvalidHandle);
    }
}

void SessionContextPreservesControlReserve() {
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    InferenceClientOptions options;
    options.limits.maxPendingCommands = 3;
    options.limits.reservedControlCommands = 1;
    Fixture fixture(options, 2, hello);
    fixture.transport->Hold(wire::Method::SessionReset);
    fixture.transport->Hold(wire::Method::SessionTruncate);
    auto first = std::async(std::launch::async, [&] { return fixture.sessions[0]->Reset(); });
    const auto reset = fixture.transport->Take(wire::Method::SessionReset);
    auto second = std::async(std::launch::async, [&] { return fixture.sessions[1]->Truncate(0); });
    const auto truncate = fixture.transport->Take(wire::Method::SessionTruncate);
    ExpectError([&] { fixture.sessions[0]->Reset(); }, ClientErrorCode::ResourceLimit);
    Expect(fixture.sessions[0]->Query().sessionId == fixture.sessions[0]->Info().sessionId,
           "Queued mutations consumed the reserved control capacity");
    fixture.transport->ReplyHeld(reset);
    fixture.transport->ReplyHeld(truncate);
    Expect(Get(first).contextLength == 0 && Get(second).contextLength == 0, "Admitted context calls did not complete");
    fixture.Close();
}

void RuntimeMemoryCapabilityAndControlReserve() {
    {
        Fixture legacy;
        ExpectError([&] { legacy.client->QueryMemory(); }, ClientErrorCode::RemoteError,
                    wire::ErrorCode::UnsupportedCapability);
        Expect(legacy.client->Query().accepting, "Optional memory query broke an old Worker");
        legacy.Close();
    }
    auto hello = Hello();
    hello.capabilities = wire::supported_capabilities();
    InferenceClientOptions options;
    options.limits.maxPendingCommands = 3;
    options.limits.reservedControlCommands = 1;
    Fixture fixture(options, 1, hello);
    fixture.transport->Hold(wire::Method::WorkerMemoryQuery);
    auto first = std::async(std::launch::async, [&] { return fixture.client->QueryMemory(); });
    const auto firstCommand = fixture.transport->Take(wire::Method::WorkerMemoryQuery);
    auto second = std::async(std::launch::async, [&] { return fixture.client->QueryMemory(); });
    const auto secondCommand = fixture.transport->Take(wire::Method::WorkerMemoryQuery);
    ExpectError([&] { fixture.client->QueryMemory(); }, ClientErrorCode::ResourceLimit);
    Expect(fixture.client->Query().accepting, "Memory queries consumed reserved control capacity");
    fixture.transport->ReplyHeld(firstCommand);
    auto busy = fixture.transport->Base(secondCommand);
    busy.runtimeMemory = {wire::RuntimeMemoryState::Busy, std::nullopt};
    fixture.transport->Reply(std::move(busy));
    const auto ready = Get(first), unavailable = Get(second);
    Expect(ready.state == wire::RuntimeMemoryState::Ready && ready.usage &&
           ready.usage->peakDeviceBytes == (uint64_t{1} << 53) + 1, "Memory RPC lost its exact counters");
    Expect(unavailable.state == wire::RuntimeMemoryState::Busy && !unavailable.usage,
           "Busy RPC published a zero or stale snapshot");
    fixture.Close();
}

void HandshakeAndReusedObjectIds() {
    {
        auto hello = Hello();
        hello.buildId = "wrong-build";
        auto transport = std::make_shared<ControlledTransport>(hello);
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity()); }, ClientErrorCode::ProtocolError);
        Expect(transport->Forced(), "Identity mismatch left the Worker running");
    }
    {
        auto hello = Hello();
        hello.capabilities.pop_back();
        auto transport = std::make_shared<ControlledTransport>(hello);
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity()); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedCapability);
        Expect(transport->Forced(), "Unsupported capability left the Worker running");
    }
    {
        auto transport = std::make_shared<ControlledTransport>(Hello(), true);
        ExpectError([&] { InferenceClient::Connect(transport, ExpectedIdentity()); },
                    ClientErrorCode::RemoteError, wire::ErrorCode::UnsupportedProtocol);
        Expect(transport->Forced(), "v1 Worker was silently accepted for v2");
    }
    {
        Fixture fixture;
        fixture.transport->Hold(wire::Method::SessionCreate);
        auto create = std::async(std::launch::async, [&] { return fixture.model->CreateSession(); });
        const auto command = fixture.transport->Take(wire::Method::SessionCreate);
        auto response = fixture.transport->Base(command);
        response.session = fixture.sessions[0]->Info();
        fixture.transport->Reply(std::move(response));
        ExpectError([&] { Get(create); }, ClientErrorCode::ProtocolError);
        fixture.transport->WaitStopped();
        Expect(fixture.transport->Forced(), "Reused session ID was accepted within an epoch");
    }
}

} // namespace

int main() {
    const std::pair<const char*, void (*)()> tests[] = {
        {"parallel requests and cross-pipe completion", ParallelRequestsAndCompletionOrder},
        {"ACK credit and independent slow reader", AckWindowAndIndependentSlowReader},
        {"request release drain and quota return", ReleaseDrainsAndReturnsReservations},
        {"parent owners and release write order", ParentOwnersAndReleaseOrdering},
        {"pending create retains owner", ReleasedParentWhileCreateIsPending},
        {"request and RPC timeouts", RequestAndRpcTimeouts},
        {"late model and rejected submit cleanup", LateModelCreationAndSubmitRejection},
        {"completed unread and retained admission", RetainedAndUnreadAdmissionLimits},
        {"count byte and control admission", ByteAndControlAdmissionLimits},
        {"malformed streams publish one local terminal", ProtocolFaultsDeliverOneLocalTerminal},
        {"window fault and consumed terminal", WindowViolationAndAlreadyConsumedTerminal},
        {"domain fault preserves drain", DomainFaultAllowsDrainAndStopsAdmission},
        {"shared shutdown and abandoned delivery", ShutdownDiscardsAndHasOneSharedResult},
        {"shutdown event fence and EOF races", ShutdownCrossPipeFenceAndEof},
        {"bounded log consumption and independent inference", BoundedLogsAndIndependentInference},
        {"Worker log capability and exact level acknowledgement", WorkerLogLevelNegotiationAndEcho},
        {"blocked writer and abnormal process exit", BrokenWriterAndProcessExitFailure},
        {"Hybrid NF4 model negotiation and loading", [] { HybridModelNegotiationAndLoading(false); }},
        {"Hybrid BF16 model negotiation and loading", [] { HybridModelNegotiationAndLoading(true); }},
        {"model capability mismatch and configure consistency", ModelCapabilityMismatchAndConfigureConsistency},
        {"allocation limits negotiation and acknowledgement", AllocationLimitsNegotiationAndAcknowledgement},
        {"thinking budget negotiation and routing", ThinkingBudgetNegotiationAndRouting},
        {"request policy capability rejection before RPC", RequestPolicyNegotiation},
        {"extended generation old-peer rejection and exact configure echo", ExtendedGenerationNegotiation},
        {"decode batching capability, admission bound and exact configure echo", DecodeBatchingNegotiationAndEcho},
        {"prefill-only negotiation and zero-token terminal routing", PrefillNegotiationAndRouting},
        {"Session context capability and correlated response validation", SessionContextNegotiationAndResponseValidation},
        {"Session context late response and pending owner release", SessionContextTimeoutAndRelease},
        {"Session mutations preserve reserved control admission", SessionContextPreservesControlReserve},
        {"runtime memory capability, exact response and reserved control", RuntimeMemoryCapabilityAndControlReserve},
        {"handshake rejection and stale object IDs", HandshakeAndReusedObjectIds},
    };
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
