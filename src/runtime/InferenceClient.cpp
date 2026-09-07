#include "runtime/InferenceClient.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

namespace aila::runtime {
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

[[noreturn]] void protocol_error(const std::string& message) {
    throw InferenceClientError(ClientErrorCode::ProtocolError, message, wire::ErrorCode::ProtocolError);
}

void require(bool condition, const std::string& message) {
    if (!condition) protocol_error(message);
}

bool control_method(wire::Method method) {
    return method != wire::Method::ModelLoad && method != wire::Method::SessionCreate &&
        method != wire::Method::RequestSubmit;
}

bool identity_equal(const wire::Identity& a, const wire::Identity& b) {
    return a.inferenceId == b.inferenceId && a.modelId == b.modelId && a.sessionId == b.sessionId;
}

void throw_remote(const wire::Response& response) {
    if (response.kind == wire::ResponseKind::Error) {
        throw InferenceClientError(ClientErrorCode::RemoteError, response.message, response.errorCode);
    }
}
}

namespace detail {

struct BufferedToken {
    uint64_t sequence = 0;
    int32_t id = -1;
    wire::RequestStatus status = wire::RequestStatus::Queued;
    uint64_t promptTokens = 0;
    uint64_t generatedTokens = 0;
};

struct InferenceRoute {
    std::string epoch;
    wire::Identity identity;
    uint32_t vocabSize = 0;
    uint64_t promptCount = 0;
    uint64_t maxGenerated = 0;
    uint64_t reservedTokens = 0;
    uint64_t reservedBytes = 0;
    std::vector<BufferedToken> ring;
    size_t head = 0;
    size_t count = 0;
    uint64_t received = 0;
    uint64_t consumed = 0;
    uint64_t acknowledged = 0;
    uint64_t ackSent = 0;
    uint64_t deliveredTokens = 0;
    uint64_t discardedTokens = 0;
    uint64_t releaseAfterTicket = 0;
    bool accepted = false;
    bool rejected = false;
    bool finalSeen = false;
    bool terminalSeen = false;
    bool complete = false;
    bool userTerminalConsumed = false;
    bool wireTerminalConsumed = false;
    bool releaseWanted = false;
    bool releaseSent = false;
    bool releaseAcknowledged = false;
    bool ackInFlight = false;
    bool abandoned = false;
    bool localFailure = false;
    bool reservationHeld = true;
    Clock::time_point ackDue = Clock::now();
    wire::Event terminal;
    wire::Response final;
    wire::Summary localSummary;
};

struct PendingCommand {
    wire::Command command;
    std::shared_ptr<InferenceRoute> route;
    wire::Response response;
    bool done = false;
    bool detached = false;
    bool internal = false;
    bool reserved = true;
    uint64_t bytes = 0;
    uint64_t enqueueTicket = 0;
};

struct ReleaseMarker {
    wire::ObjectType type;
    uint64_t id;
    uint64_t afterTicket;
};

struct InferenceConnection : std::enable_shared_from_this<InferenceConnection> {
    std::shared_ptr<InferenceTransport> transport;
    ExpectedInferenceIdentity expected;
    InferenceClientOptions options;
    wire::Limits limits;
    mutable std::mutex mutex;
    std::condition_variable changed;
    wire::Hello hello;
    bool helloSeen = false;
    bool configured = false;
    bool dead = false;
    bool graceful = false;
    bool shutdownWanted = false;
    bool shutdownSent = false;
    bool shutdownResponseSeen = false;
    bool writing = false;
    bool markerTurn = true;
    size_t owners = 0;
    size_t threadsRunning = 0;
    wire::DomainStatus status = wire::DomainStatus::Starting;
    std::string failure;
    ClientErrorCode failureCode = ClientErrorCode::Closed;
    std::optional<wire::ErrorCode> failureRemoteCode;
    uint64_t nextRpc = 1;
    uint64_t lastEnqueuedTicket = 0;
    uint64_t lastSentTicket = 0;
    uint64_t lastModelId = 0;
    uint64_t lastSessionId = 0;
    uint64_t lastOrdinal = 0;
    uint64_t shutdownOrdinal = 0;
    uint64_t pendingCount = 0;
    uint64_t normalPendingCount = 0;
    uint64_t outstanding = 0;
    uint64_t reservedTokens = 0;
    uint64_t ingressBytes = 0;
    uint64_t transportBytes = 0;
    uint64_t peakIngress = 0;
    uint64_t peakTransport = 0;
    uint64_t droppedLogs = 0;
    Clock::time_point writeStarted;
    Clock::time_point shutdownDeadline;
    uint32_t shutdownGraceMs = 0;
    std::deque<std::shared_ptr<PendingCommand>> queued;
    std::map<uint64_t, std::shared_ptr<PendingCommand>> pending;
    std::vector<std::shared_ptr<InferenceRoute>> routes;
    std::vector<ReleaseMarker> releaseMarks;
    size_t routeCursor = 0;

    InferenceConnection(std::shared_ptr<InferenceTransport> value, ExpectedInferenceIdentity identity,
                        InferenceClientOptions settings)
        : transport(std::move(value)), expected(std::move(identity)), options(std::move(settings)),
          limits(options.limits) {
        failure.reserve(wire::kMaxErrorBytes);
        routes.reserve(limits.maxRetainedRequestHandles);
        releaseMarks.reserve(static_cast<size_t>(limits.maxPendingCommands) + limits.maxSessions + limits.maxLoadedModels);
    }

    std::chrono::milliseconds Timeout(std::chrono::milliseconds timeout) const {
        if (timeout == 0ms) timeout = options.commandTimeout;
        if (timeout <= 0ms) throw InferenceClientError(ClientErrorCode::Timeout, "RPC timeout must be positive");
        return timeout;
    }

    void CheckUser(bool execution) const {
        if (dead) throw InferenceClientError(failureCode, failure.empty() ? "Inference connection is closed" : failure, failureRemoteCode);
        if (shutdownWanted) throw InferenceClientError(ClientErrorCode::Closed, "Inference connection is stopping");
        if (!configured) throw InferenceClientError(ClientErrorCode::Closed, "Inference connection is not ready");
        if (execution && status == wire::DomainStatus::Faulted) {
            throw InferenceClientError(ClientErrorCode::RemoteError, "Inference device domain is faulted", wire::ErrorCode::RuntimeFault);
        }
    }

    void AddOwner() {
        std::lock_guard<std::mutex> lock(mutex);
        ++owners;
    }

    void AbandonLocked(InferenceRoute& route) {
        route.abandoned = true;
        route.discardedTokens += route.count;
        route.count = 0;
        route.head = 0;
        route.consumed = route.terminalSeen ? route.received - 1 : route.received;
        if (route.complete && !route.localFailure) {
            route.consumed = route.received;
            route.wireTerminalConsumed = true;
        }
        route.ackDue = Clock::now();
    }

    void RequestShutdownLocked(std::chrono::milliseconds timeout) {
        if (shutdownWanted || dead) return;
        shutdownWanted = true;
        status = wire::DomainStatus::Stopping;
        const auto count = std::min<uint64_t>(static_cast<uint64_t>(timeout.count()), limits.maxShutdownGraceMs);
        shutdownGraceMs = static_cast<uint32_t>(count);
        shutdownDeadline = Clock::now() + timeout;
        // Worker shutdown releases its registry; do not send late release RPCs
        // after it has already acknowledged the final close barrier.
        releaseMarks.clear();
        for (auto& route : routes) AbandonLocked(*route);
        changed.notify_all();
    }

    void ReleaseOwner() noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (owners) --owners;
            if (owners == 0 && configured) RequestShutdownLocked(options.shutdownTimeout);
        } catch (...) { Fail(ClientErrorCode::Closed, "Could not close last inference owner"); }
    }

    void ReleaseReservationLocked(InferenceRoute& route) {
        if (!route.reservationHeld) return;
        // Free the reserved ring before returning credit to another Submit.
        std::vector<BufferedToken>().swap(route.ring);
        route.count = 0;
        route.head = 0;
        route.reservationHeld = false;
        --outstanding;
        reservedTokens -= route.reservedTokens;
        transportBytes -= route.reservedBytes;
    }

    void CollectRoutesLocked() {
        routes.erase(std::remove_if(routes.begin(), routes.end(), [&](const auto& route) {
            const bool released = route->rejected ||
                (route->releaseWanted && route->releaseAcknowledged && route->complete &&
                 route->wireTerminalConsumed && route->acknowledged == route->received);
            if (released) ReleaseReservationLocked(*route);
            return released;
        }), routes.end());
        if (routeCursor >= routes.size()) routeCursor = 0;
    }

    void Fail(ClientErrorCode code, std::string_view message,
              std::optional<wire::ErrorCode> remoteCode = {}) noexcept {
        try {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (dead) return;
                dead = true;
                graceful = false;
                status = wire::DomainStatus::Faulted;
                failureCode = code;
                failureRemoteCode = remoteCode;
                failure.assign(message.data(), std::min(message.size(), wire::kMaxErrorBytes));
                for (auto& route : routes) {
                    if (!route->userTerminalConsumed) {
                        route->localFailure = true;
                        route->complete = true;
                        route->localSummary.status = wire::RequestStatus::Failed;
                        route->localSummary.endReason = wire::EndReason::Failed;
                        route->localSummary.errorCode = code == ClientErrorCode::ProtocolError
                            ? wire::ErrorCode::ProtocolError : wire::ErrorCode::TransportError;
                        route->localSummary.message = failure;
                        route->localSummary.generatedTokens = route->deliveredTokens;
                        route->abandoned = true;
                        // No wire eventCount or sequence is invented.
                    }
                    route->discardedTokens += route->count;
                    route->count = 0;
                    ReleaseReservationLocked(*route);
                }
                routes.clear();
                for (auto& item : pending) item.second->done = true;
                for (auto& item : queued) item->done = true;
                pending.clear();
                queued.clear();
                releaseMarks.clear();
                pendingCount = normalPendingCount = ingressBytes = 0;
            }
            transport->Stop(true);
            changed.notify_all();
        } catch (...) {
            transport->Stop(true);
            changed.notify_all();
        }
    }

    uint64_t CommandBytes(const wire::Command& command) const {
        // Keep room for the typed command, its encoded/in-flight copies and a
        // separate reply. Frame bounds count escaped UTF-8, not character count.
        uint64_t bytes = sizeof(PendingCommand) + 8ULL + 2ULL * limits.maxHeaderBytes + limits.maxFinalFrameBytes +
            static_cast<uint64_t>(command.promptIds.size()) * 8 +
            static_cast<uint64_t>(command.stopTokenIds.size()) * sizeof(int32_t) + command.modelDirectory.size();
        for (const auto& capability : command.requiredCapabilities) bytes += sizeof(std::string) + capability.size();
        return bytes;
    }

    bool SpaceForCommandLocked(const wire::Command& command) const {
        if (pendingCount >= limits.maxPendingCommands) return false;
        if (!control_method(command.method) &&
            normalPendingCount >= limits.maxPendingCommands - limits.reservedControlCommands) return false;
        uint64_t bytes = CommandBytes(command);
        // Count reservation alone is insufficient when a lowered byte ceiling
        // fills first. Preserve at least one serialized cleanup RPC as well.
        if (!control_method(command.method)) bytes += CommandBytes(wire::Command{});
        return bytes <= limits.maxIngressBytes && ingressBytes <= limits.maxIngressBytes - bytes;
    }

    std::shared_ptr<PendingCommand> ReserveCommandLocked(wire::Command command,
                                                        bool internal = false,
                                                        std::shared_ptr<InferenceRoute> route = {}) {
        if (!SpaceForCommandLocked(command)) {
            throw InferenceClientError(ClientErrorCode::ResourceLimit, "Inference command queue is full");
        }
        auto item = std::make_shared<PendingCommand>();
        item->command = std::move(command);
        item->route = std::move(route);
        item->internal = internal;
        item->bytes = CommandBytes(item->command);
        ++pendingCount;
        if (!control_method(item->command.method)) ++normalPendingCount;
        ingressBytes += item->bytes;
        peakIngress = std::max(peakIngress, ingressBytes);
        return item;
    }

    void ReturnCommandLocked(PendingCommand& item) {
        if (!item.reserved) return;
        item.reserved = false;
        --pendingCount;
        if (!control_method(item.command.method)) --normalPendingCount;
        ingressBytes -= item.bytes;
    }

    void ValidateCommandLocked(wire::Command& command) const {
        // Validate before reserving anything; the real sender later replaces
        // this nonzero placeholder with its monotonically allocated RPC ID.
        command.epoch = hello.epoch;
        command.rpcId = 1;
        ipc::Frame frame;
        wire::ValidationError error;
        if (!wire::make_command(command, frame, error, limits)) {
            throw InferenceClientError(ClientErrorCode::RemoteError, error.message, error.code);
        }
        command.rpcId = 0;
    }

    wire::Response Call(wire::Command command, std::chrono::milliseconds timeout,
                        bool configure = false, std::shared_ptr<InferenceRoute> route = {},
                        const std::atomic<bool>* released = nullptr) {
        timeout = Timeout(timeout);
        std::unique_lock<std::mutex> lock(mutex);
        if (released && released->load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference handle is released");
        if (!configure) CheckUser(!control_method(command.method));
        ValidateCommandLocked(command);
        require(lastEnqueuedTicket != std::numeric_limits<uint64_t>::max(), "Inference command tickets exhausted");
        auto item = ReserveCommandLocked(std::move(command), false, std::move(route));
        item->enqueueTicket = lastEnqueuedTicket + 1;
        try { queued.push_back(item); }
        catch (...) { ReturnCommandLocked(*item); throw; }
        ++lastEnqueuedTicket;
        changed.notify_all();
        if (!changed.wait_for(lock, timeout, [&] { return item->done || dead; })) {
            item->detached = true;
            if (item->route && item->command.method == wire::Method::RequestSubmit) {
                item->route->releaseWanted = true;
                item->route->releaseAfterTicket = lastEnqueuedTicket;
                AbandonLocked(*item->route);
            }
            changed.notify_all();
            throw InferenceClientError(ClientErrorCode::Timeout,
                "Inference RPC wait timed out; connection remains active");
        }
        if (dead && !item->done) throw InferenceClientError(failureCode, failure, failureRemoteCode);
        if (dead && item->response.rpcId == 0) throw InferenceClientError(failureCode, failure, failureRemoteCode);
        auto response = std::move(item->response);
        lock.unlock();
        throw_remote(response);
        return response;
    }

    void MarkReleaseLocked(wire::ObjectType type, uint64_t id) {
        if (dead || shutdownWanted) return;
        if (std::find_if(releaseMarks.begin(), releaseMarks.end(), [&](const auto& marker) {
                return marker.type == type && marker.id == id;
            }) != releaseMarks.end()) return;
        if (releaseMarks.size() == releaseMarks.capacity()) protocol_error("Inference release marker capacity exceeded");
        releaseMarks.push_back({type, id, lastEnqueuedTicket});
        changed.notify_all();
    }

    void ReleaseObject(wire::ObjectType type, uint64_t id) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            MarkReleaseLocked(type, id);
        } catch (...) { Fail(ClientErrorCode::ProtocolError, "Could not retain a bounded remote release marker"); }
    }

    void ReleaseRequest(const std::shared_ptr<InferenceRoute>& route) noexcept {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (route->releaseWanted) return;
            route->releaseWanted = true;
            route->releaseAfterTicket = lastEnqueuedTicket;
            AbandonLocked(*route);
            changed.notify_all();
        } catch (...) { Fail(ClientErrorCode::Closed, "Could not abandon inference request"); }
    }

    std::shared_ptr<PendingCommand> NextMarkerLocked() {
        wire::Command command;
        command.epoch = hello.epoch;
        if (!routes.empty()) {
            for (size_t n = 0; n < routes.size(); ++n) {
                routeCursor %= routes.size();
                auto route = routes[routeCursor++];
                if (!route->accepted) continue;
                // Closing must reach Worker before an indefinitely refilled
                // ACK stream can consume every marker turn for this request.
                if (route->releaseWanted && !route->releaseSent && !shutdownWanted &&
                    route->releaseAfterTicket <= lastSentTicket) {
                    command.method = wire::Method::HandleRelease;
                    command.objectType = wire::ObjectType::Request;
                    command.objectId = route->identity.inferenceId;
                    if (!SpaceForCommandLocked(command)) return {};
                    auto item = ReserveCommandLocked(std::move(command), true, route);
                    route->releaseSent = true;
                    return item;
                }
                if (route->consumed > route->acknowledged && !route->ackInFlight &&
                    (route->wireTerminalConsumed || route->abandoned ||
                     route->consumed - route->acknowledged >= 16 || Clock::now() >= route->ackDue)) {
                    command.method = wire::Method::RequestAck;
                    command.inferenceId = route->identity.inferenceId;
                    command.consumedSequence = route->consumed;
                    if (!SpaceForCommandLocked(command)) return {};
                    auto item = ReserveCommandLocked(std::move(command), true, route);
                    route->ackInFlight = true;
                    return item;
                }
            }
        }
        if (!releaseMarks.empty() && !shutdownWanted && releaseMarks.front().afterTicket <= lastSentTicket) {
            command.method = wire::Method::HandleRelease;
            command.objectType = releaseMarks.front().type;
            command.objectId = releaseMarks.front().id;
            if (!SpaceForCommandLocked(command)) return {};
            auto item = ReserveCommandLocked(std::move(command), true);
            releaseMarks.erase(releaseMarks.begin());
            return item;
        }
        if (shutdownWanted && !shutdownSent && queued.empty()) {
            command.method = wire::Method::WorkerShutdown;
            command.graceMs = shutdownGraceMs;
            if (!SpaceForCommandLocked(command)) return {};
            auto item = ReserveCommandLocked(std::move(command), true);
            shutdownSent = true;
            return item;
        }
        return {};
    }

    std::shared_ptr<InferenceRoute> FindRouteLocked(uint64_t id) const {
        const auto found = std::find_if(routes.begin(), routes.end(), [&](const auto& item) {
            return item->identity.inferenceId == id;
        });
        if (found == routes.end()) protocol_error("Unassociated inference response/event ID");
        return *found;
    }

    void ValidateCompletionLocked(InferenceRoute& route) {
        if (!route.accepted || !route.finalSeen || !route.terminalSeen || route.complete) return;
        wire::ValidationError error;
        require(wire::match_terminal_final(route.terminal, route.final, error, limits), error.message);
        require(route.received == route.final.summary.eventCount, "Incomplete inference terminal event count");
        route.complete = true;
        if (route.abandoned) AbandonLocked(route);
    }

    bool CloseBarrierLocked() {
        if (!shutdownResponseSeen || dead) return false;
        require(lastOrdinal <= shutdownOrdinal, "Events continued past shutdown ordinal");
        if (lastOrdinal != shutdownOrdinal) return false;
        require(pending.empty() && queued.empty(), "Shutdown completed before earlier RPC responses");
        for (auto& route : routes) {
            require(route->rejected || (route->accepted && route->complete && route->wireTerminalConsumed &&
                    route->acknowledged == route->received), "Shutdown completed before request drain/ACK");
            ReleaseReservationLocked(*route);
        }
        routes.clear();
        dead = true;
        graceful = true;
        status = wire::DomainStatus::Closed;
        changed.notify_all();
        return true;
    }

    void UncleanExit(std::string_view message) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            // CloseBarrierLocked already set dead: ordinary Fail would return
            // early. Change the shared shutdown result before formatting text.
            graceful = false;
            status = wire::DomainStatus::Faulted;
            failureCode = ClientErrorCode::Closed;
            failureRemoteCode = wire::ErrorCode::TransportError;
            failure.assign("Worker did not exit cleanly after shutdown barrier: ");
            failure.append(message.data(), std::min(message.size(), wire::kMaxErrorBytes - failure.size()));
        }
        transport->Stop(true);
    }

    void FinishTransport() {
        transport->Stop(false);
        try {
            std::string error;
            const auto remaining = std::max(0ms, std::chrono::duration_cast<std::chrono::milliseconds>(shutdownDeadline - Clock::now()));
            if (!transport->WaitForExit(remaining, error)) UncleanExit(error);
        } catch (const std::exception& error) { UncleanExit(error.what()); }
        catch (...) { UncleanExit("Unknown process exit failure"); }
        changed.notify_all();
    }

    void ProcessResponse(const ipc::Frame& frame) {
        bool close = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (dead) return;
            if (!helloSeen) {
                wire::Hello received;
                wire::ValidationError error;
                if (frame.header.protocol != wire::kProtocolVersion) {
                    throw InferenceClientError(ClientErrorCode::RemoteError, "Worker does not support IPC v2",
                                               wire::ErrorCode::UnsupportedProtocol);
                }
                if (!wire::parse_hello(frame, received, error)) {
                    throw InferenceClientError(error.code == wire::ErrorCode::UnsupportedProtocol
                            ? ClientErrorCode::RemoteError : ClientErrorCode::ProtocolError,
                        error.message, error.code);
                }
                require(received.buildId == expected.buildId && received.executable == expected.executable &&
                        received.runtimeDirectory == expected.runtimeDirectory &&
                        received.currentDirectory == expected.currentDirectory && received.path == expected.path,
                        "Worker handshake identity/path isolation mismatch");
                hello = std::move(received);
                helloSeen = true;
                changed.notify_all();
                return;
            }
            wire::Response response;
            wire::ValidationError error;
            require(wire::parse_response(frame, response, error, limits), error.message);
            require(response.epoch == hello.epoch, "Worker response epoch mismatch");
            auto found = pending.find(response.rpcId);
            if (found == pending.end()) {
                require(response.method == wire::Method::RequestSubmit && response.phase == wire::SubmissionPhase::Completed,
                        "Unassociated or duplicate RPC response");
                auto route = FindRouteLocked(response.rpcId);
                require(route->accepted && !route->finalSeen && !route->rejected &&
                        identity_equal(route->identity, response.identity), "Invalid final inference response identity/order");
                require(response.summary.generatedTokens <= route->maxGenerated &&
                        response.summary.promptTokensProcessed <= route->promptCount,
                        "Final inference counts exceed the submitted request");
                route->final = std::move(response);
                route->finalSeen = true;
                ValidateCompletionLocked(*route);
            } else {
                auto item = found->second;
                require(response.method == item->command.method, "RPC response method mismatch");
                if (response.kind == wire::ResponseKind::Result && response.method == wire::Method::ModelLoad) {
                    // The v2 lifecycle executor publishes successful creations
                    // in allocation order; failed IDs may leave gaps.
                    require(response.model.modelId > lastModelId, "Worker reused or reordered a model ID");
                    require(response.model.maxSequenceLength <= item->command.maxSequenceLength,
                            "Loaded model capacity exceeds the requested capacity");
                    require(std::find(response.model.capabilities.begin(), response.model.capabilities.end(),
                                      "qwen3_nf4_bf16_kv") != response.model.capabilities.end(),
                            "Loaded model does not support the requested backend/KV type");
                    lastModelId = response.model.modelId;
                }
                if (response.kind == wire::ResponseKind::Result && response.method == wire::Method::SessionCreate) {
                    require(response.session.modelId == item->command.modelId && response.session.contextLength == 0 &&
                            !response.session.busy, "Created session identity/initial state mismatch");
                    require(response.session.sessionId > lastSessionId, "Worker reused or reordered a session ID");
                    require(response.session.capacity == item->command.maxSequenceLength &&
                            response.session.cacheBytes == item->command.maxSessionCacheBytes,
                            "Created session capacity/cache reservation mismatch");
                    lastSessionId = response.session.sessionId;
                }
                if (response.kind == wire::ResponseKind::Result && response.method == wire::Method::SessionQuery) {
                    require(response.session.sessionId == item->command.sessionId &&
                            response.session.modelId == item->command.modelId &&
                            response.session.capacity == item->command.maxSequenceLength &&
                            response.session.cacheBytes == item->command.maxSessionCacheBytes,
                            "Session query identity/metadata mismatch");
                }
                if (response.kind == wire::ResponseKind::Result && response.method == wire::Method::RequestQuery && item->route) {
                    require(response.request.summary.generatedTokens <= item->route->maxGenerated &&
                            response.request.summary.promptTokensProcessed <= item->route->promptCount &&
                            response.request.lastAcknowledgedSequence <= item->route->ackSent,
                            "Request query exceeds submitted or acknowledged progress");
                }
                if (response.kind == wire::ResponseKind::Result && response.method == wire::Method::WorkerQuery &&
                    response.worker.domainStatus == wire::DomainStatus::Faulted) {
                    status = wire::DomainStatus::Faulted;
                }
                if (item->command.method == wire::Method::RequestSubmit) {
                    auto route = item->route;
                    require(route && !route->accepted && !route->rejected, "Duplicate submit acceptance");
                    if (response.phase == wire::SubmissionPhase::Rejected) {
                        require(response.kind == wire::ResponseKind::Error && route->received == 0,
                                "Rejected submit already emitted events");
                        route->rejected = true;
                        ReleaseReservationLocked(*route);
                    } else {
                        require(response.kind == wire::ResponseKind::Accepted && response.phase == wire::SubmissionPhase::Accepted &&
                                identity_equal(route->identity, response.identity) && response.tokenWindow == limits.tokenWindow,
                                "Invalid accepted inference identity/window");
                        route->accepted = true;
                    }
                } else if (item->internal) {
                    require(response.kind == wire::ResponseKind::Result, "Internal ACK/release/shutdown was rejected: " + response.message);
                    if (item->command.method == wire::Method::RequestAck) {
                        auto& route = *item->route;
                        require(response.acknowledgedSequence == item->command.consumedSequence &&
                                response.acknowledgedSequence <= route.consumed, "Invalid cumulative ACK response");
                        route.acknowledged = response.acknowledgedSequence;
                        route.ackInFlight = false;
                        route.ackDue = Clock::now() + 10ms;
                        if (route.wireTerminalConsumed && route.acknowledged == route.received) ReleaseReservationLocked(route);
                    } else if (item->command.method == wire::Method::HandleRelease) {
                        require(response.releaseAccepted, "Remote release was not accepted");
                        if (item->route) item->route->releaseAcknowledged = true;
                    } else if (item->command.method == wire::Method::WorkerShutdown) {
                        require(response.worker.domainStatus == wire::DomainStatus::Closed, "Shutdown response did not close the domain");
                        shutdownResponseSeen = true;
                        shutdownOrdinal = response.lastEventOrdinal;
                    }
                }
                if (response.kind == wire::ResponseKind::Error && response.errorCode == wire::ErrorCode::RuntimeFault) {
                    status = wire::DomainStatus::Faulted;
                }
                if (item->detached && response.kind == wire::ResponseKind::Result) {
                    if (response.method == wire::Method::ModelLoad) MarkReleaseLocked(wire::ObjectType::Model, response.model.modelId);
                    if (response.method == wire::Method::SessionCreate) MarkReleaseLocked(wire::ObjectType::Session, response.session.sessionId);
                }
                item->response = std::move(response);
                item->done = true;
                ReturnCommandLocked(*item);
                pending.erase(found);
                CollectRoutesLocked();
            }
            close = CloseBarrierLocked();
            changed.notify_all();
        }
        if (close) FinishTransport();
    }

    void ProcessEvent(const ipc::Frame& frame) {
        bool close = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (dead) return;
            wire::Event event;
            wire::ValidationError error;
            require(wire::parse_event(frame, event, error, limits), error.message);
            require(helloSeen && event.epoch == hello.epoch, "Worker event epoch mismatch");
            require(lastOrdinal != std::numeric_limits<uint64_t>::max() && event.eventOrdinal == lastOrdinal + 1,
                    "Worker eventOrdinal is duplicated or discontinuous");
            lastOrdinal = event.eventOrdinal;
            if (event.kind == wire::EventKind::Log) {
                // No user callback on the reader. This internal API currently
                // exposes only a drop count; there is no unbounded log queue.
                ++droppedLogs;
            } else if (event.kind == wire::EventKind::DomainFault) {
                status = wire::DomainStatus::Faulted;
            } else {
                auto route = FindRouteLocked(event.identity.inferenceId);
                require(!route->rejected && !route->terminalSeen && identity_equal(route->identity, event.identity),
                        "Invalid request event identity or event after terminal");
                require(event.sequence == route->received + 1 && event.summary.promptTokensProcessed <= route->promptCount,
                        "Request event sequence/progress mismatch");
                require(event.summary.generatedTokens <= route->maxGenerated, "Request generated more tokens than reserved");
                route->received = event.sequence;
                if (event.kind == wire::EventKind::Token) {
                    require(static_cast<uint32_t>(event.tokenId) < route->vocabSize &&
                            event.summary.generatedTokens == event.sequence,
                            "Request token ID/count is invalid");
                    // ACK replies and refilled token events use different
                    // pipes. Credit may be used before its reply arrives.
                    require(event.sequence <= route->ackSent + limits.tokenWindow,
                            "Worker exceeded the negotiated unacknowledged token window");
                    if (route->abandoned) {
                        ++route->discardedTokens;
                        route->consumed = event.sequence;
                        route->ackDue = Clock::now();
                    } else {
                        require(route->count < route->ring.size(), "Worker overflowed a request's reserved token ring");
                        auto& token = route->ring[(route->head + route->count) % route->ring.size()];
                        token = {event.sequence, event.tokenId, event.summary.status,
                                 event.summary.promptTokensProcessed, event.summary.generatedTokens};
                        ++route->count;
                    }
                } else {
                    require(event.sequence == event.summary.generatedTokens + 1,
                            "Terminal sequence does not match published token count");
                    route->terminalSeen = true;
                    route->terminal = std::move(event);
                    ValidateCompletionLocked(*route);
                }
            }
            close = CloseBarrierLocked();
            changed.notify_all();
        }
        if (close) FinishTransport();
    }

    void SenderLoop() {
        for (;;) {
            std::shared_ptr<PendingCommand> item;
            ipc::Frame frame;
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (dead) return;
                if (helloSeen && markerTurn) item = NextMarkerLocked();
                if (!item && !queued.empty()) {
                    item = queued.front();
                    queued.pop_front();
                    markerTurn = true;
                } else if (item) markerTurn = false;
                if (!item && helloSeen) item = NextMarkerLocked();
                if (!item) {
                    changed.wait_for(lock, 10ms);
                    continue;
                }
                require(nextRpc != 0 && nextRpc != std::numeric_limits<uint64_t>::max(), "Inference RPC IDs exhausted");
                item->command.rpcId = nextRpc++;
                item->command.epoch = hello.epoch;
                if (item->enqueueTicket) lastSentTicket = item->enqueueTicket;
                if (item->route && item->command.method == wire::Method::RequestSubmit) {
                    item->route->identity.inferenceId = item->command.rpcId;
                }
                if (item->route && item->command.method == wire::Method::RequestAck) {
                    item->route->ackSent = item->command.consumedSequence;
                }
                pending.emplace(item->command.rpcId, item);
                wire::ValidationError error;
                require(wire::make_command(item->command, frame, error, limits), error.message);
                writing = true;
                writeStarted = Clock::now();
                changed.notify_all();
            }
            std::string error;
            const bool ok = transport->WriteCommand(frame, error);
            {
                std::lock_guard<std::mutex> lock(mutex);
                writing = false;
                changed.notify_all();
                if (dead) return;
            }
            if (!ok) throw InferenceClientError(ClientErrorCode::Closed, "Inference command pipe failed: " + error);
        }
    }

    void ReaderLoop(bool event) {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                // A successful shutdown result is the last response. Its event
                // fence may still be in the other reader; EOF here is normal.
                if (!event && shutdownResponseSeen) changed.wait(lock, [&] { return dead; });
                if (dead) return;
            }
            ipc::Frame frame;
            std::string error;
            const bool ok = event ? transport->ReadEvent(frame, error) : transport->ReadResponse(frame, error);
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (dead) return;
                // The Worker can close its event handle after writing the last
                // event while the response reader has yet to see shutdown.
                if (!ok && event && shutdownWanted) {
                    changed.wait(lock, [&] { return dead || shutdownResponseSeen; });
                    if (dead) return;
                    require(lastOrdinal == shutdownOrdinal, "Event pipe closed before the shutdown fence");
                }
            }
            if (!ok) throw InferenceClientError(ClientErrorCode::Closed, "Inference reader failed: " + error);
            if (event) ProcessEvent(frame);
            else ProcessResponse(frame);
        }
    }

    void WatchdogLoop() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            if (dead) return;
            const bool writeExpired = writing && Clock::now() - writeStarted >=
                std::chrono::milliseconds(limits.connectionWriteTimeoutMs);
            const bool shutdownExpired = shutdownWanted && Clock::now() >= shutdownDeadline;
            if (writeExpired || shutdownExpired) {
                lock.unlock();
                Fail(ClientErrorCode::Closed, writeExpired ? "Inference command writer stopped making progress"
                                                         : "Inference shutdown grace period expired");
                return;
            }
            changed.wait_for(lock, 10ms);
        }
    }

    template <typename Function>
    void StartThread(Function function) {
        auto self = shared_from_this();
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++threadsRunning;
        }
        try {
            std::thread([self, function] {
                try { (self.get()->*function)(); }
                catch (const InferenceClientError& error) { self->Fail(error.Code(), error.what(), error.RemoteCode()); }
                catch (const std::exception& error) { self->Fail(ClientErrorCode::ProtocolError, error.what()); }
                catch (...) { self->Fail(ClientErrorCode::Closed, "Unknown inference I/O failure"); }
                std::lock_guard<std::mutex> lock(self->mutex);
                --self->threadsRunning;
                self->changed.notify_all();
            }).detach();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            --threadsRunning;
            throw;
        }
    }

    void ResponseLoop() { ReaderLoop(false); }
    void EventLoop() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            // Hello and logs have independent pipes. A valid early log must
            // not race the response reader's initial identity validation.
            changed.wait(lock, [&] { return helloSeen || dead; });
            if (dead) return;
        }
        ReaderLoop(true);
    }

    void Initialize() {
        try {
            // Configure must echo these exact limits. Install them before a
            // reader can allocate the frame following the bootstrap hello.
            transport->SetLimits(limits);
            StartThread(&InferenceConnection::SenderLoop);
            StartThread(&InferenceConnection::ResponseLoop);
            StartThread(&InferenceConnection::EventLoop);
            StartThread(&InferenceConnection::WatchdogLoop);
            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!changed.wait_for(lock, options.handshakeTimeout, [&] { return helloSeen || dead; })) {
                    throw InferenceClientError(ClientErrorCode::Timeout, "Inference Worker handshake timed out");
                }
                if (dead) throw InferenceClientError(failureCode, failure, failureRemoteCode);
                wire::ValidationError error;
                wire::Limits selected;
                if (!wire::negotiate(limits, options.requiredCapabilities, hello.hardLimits,
                                     hello.capabilities, selected, error)) {
                    throw InferenceClientError(ClientErrorCode::RemoteError, error.message, error.code);
                }
            }
            wire::Command command;
            command.method = wire::Method::WorkerConfigure;
            command.requiredCapabilities = options.requiredCapabilities;
            command.limits = limits;
            auto response = Call(std::move(command), options.handshakeTimeout, true);
            wire::ValidationError error;
            wire::Limits selected;
            if (!wire::negotiate(limits, options.requiredCapabilities, response.limits,
                                 response.capabilities, selected, error)) protocol_error(error.message);
            // Requesting the selected response limits against the original
            // request also rejects a peer that silently increased a bound.
            require(wire::validate_limits(response.limits, error, limits), "Worker changed negotiated limits: " + error.message);
            transport->SetLimits(response.limits);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (dead) throw InferenceClientError(failureCode, failure, failureRemoteCode);
                limits = response.limits;
                configured = true;
                if (status == wire::DomainStatus::Starting) status = wire::DomainStatus::Ready;
            }
        } catch (...) {
            Fail(ClientErrorCode::Closed, "Inference connection initialization failed");
            throw;
        }
    }

    std::shared_ptr<InferenceRoute> Submit(uint64_t modelId, uint64_t sessionId, uint32_t vocabSize,
                                          std::vector<int32_t> tokens, InferenceRequestOptions request,
                                          std::chrono::milliseconds timeout, const std::atomic<bool>* released) {
        wire::Command command;
        command.method = wire::Method::RequestSubmit;
        command.sessionId = sessionId;
        command.promptIds = std::move(tokens);
        command.prefillChunkSize = request.prefillChunkSize;
        command.stopTokenIds = std::move(request.stopTokenIds);
        command.generation = request.generation;
        auto route = std::make_shared<InferenceRoute>();
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (released->load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference session is released");
            CheckUser(true);
            ValidateCommandLocked(command);
            for (const auto token : command.promptIds) {
                if (token < 0 || static_cast<uint32_t>(token) >= vocabSize) {
                    throw InferenceClientError(ClientErrorCode::RemoteError, "Prompt token is outside the model vocabulary",
                                               wire::ErrorCode::InvalidArgument);
                }
            }
            const uint64_t tokenReservation = command.promptIds.size() + static_cast<uint64_t>(command.generation.maxNewTokens);
            const uint64_t byteReservation = (static_cast<uint64_t>(limits.tokenWindow) + 1) * limits.maxTokenFrameBytes +
                limits.maxTerminalFrameBytes + limits.maxFinalFrameBytes;
            if (outstanding >= limits.maxOutstandingRequests || routes.size() >= limits.maxRetainedRequestHandles ||
                tokenReservation > limits.maxOutstandingTokens || reservedTokens > limits.maxOutstandingTokens - tokenReservation ||
                byteReservation > limits.maxTransportBytes || transportBytes > limits.maxTransportBytes - byteReservation ||
                !SpaceForCommandLocked(command)) {
                throw InferenceClientError(ClientErrorCode::ResourceLimit, "Inference request/transport admission limit reached");
            }
            route->ring.resize(limits.tokenWindow);
            route->localSummary.message.reserve(wire::kMaxErrorBytes);
            route->epoch = hello.epoch;
            route->identity.modelId = modelId;
            route->identity.sessionId = sessionId;
            route->vocabSize = vocabSize;
            route->promptCount = command.promptIds.size();
            route->maxGenerated = command.generation.maxNewTokens;
            route->reservedTokens = tokenReservation;
            route->reservedBytes = byteReservation;
            routes.push_back(route);
            ++outstanding;
            reservedTokens += tokenReservation;
            transportBytes += byteReservation;
            peakTransport = std::max(peakTransport, transportBytes);
        }
        try {
            Call(std::move(command), timeout, false, route, released);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (route->identity.inferenceId == 0 && !dead) {
                const auto queuedItem = std::find_if(queued.begin(), queued.end(), [&](const auto& item) { return item->route == route; });
                if (queuedItem == queued.end()) {
                    route->rejected = true;
                    CollectRoutesLocked();
                }
            }
            throw;
        }
        return route;
    }

    InferenceReadResult Read(const std::shared_ptr<InferenceRoute>& route, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        if (route->releaseWanted) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference request is released");
        changed.wait_for(lock, std::max(timeout, 0ms), [&] {
            return route->count || route->complete || route->releaseWanted || dead;
        });
        if (route->releaseWanted) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference request is released");
        if (route->userTerminalConsumed) return {InferenceReadStatus::Closed, {}};
        InferenceReadEvent event;
        if (route->count) {
            const auto token = route->ring[route->head];
            route->head = (route->head + 1) % route->ring.size();
            --route->count;
            route->consumed = token.sequence;
            ++route->deliveredTokens;
            if (route->consumed == route->acknowledged + 1) route->ackDue = Clock::now() + 10ms;
            event.sequence = token.sequence;
            event.tokenId = token.id;
            event.summary.status = token.status;
            event.summary.promptTokensProcessed = token.promptTokens;
            event.summary.generatedTokens = token.generatedTokens;
        } else if (route->complete) {
            route->userTerminalConsumed = true;
            event.terminal = true;
            event.localFailure = route->localFailure;
            event.summary = route->localFailure ? route->localSummary : route->terminal.summary;
            event.sequence = route->localFailure ? 0 : route->terminal.sequence;
            if (!route->localFailure) {
                route->wireTerminalConsumed = true;
                route->consumed = route->terminal.sequence;
                route->ackDue = Clock::now();
            }
        } else return {InferenceReadStatus::Timeout, {}};
        event.deliveryAbandoned = route->abandoned;
        event.deliveredTokens = route->deliveredTokens;
        event.discardedPublishedTokens = route->discardedTokens;
        changed.notify_all();
        return {InferenceReadStatus::Event, std::move(event)};
    }

    void Shutdown(std::chrono::milliseconds timeout) {
        if (timeout == 0ms) timeout = options.shutdownTimeout;
        if (timeout <= 0ms) throw InferenceClientError(ClientErrorCode::Timeout, "Shutdown timeout must be positive");
        std::unique_lock<std::mutex> lock(mutex);
        RequestShutdownLocked(timeout);
        changed.wait(lock, [&] { return dead; });
        changed.wait(lock, [&] { return threadsRunning == 0; });
        if (!graceful) throw InferenceClientError(failureCode, failure, failureRemoteCode);
    }
};

} // namespace detail

namespace {
// Explicit Release can race an already registered create/submit. Keep that
// operation alive until it either publishes its child owner or rolls back.
class OperationOwner {
public:
    explicit OperationOwner(std::shared_ptr<detail::InferenceConnection> value) : connection(std::move(value)) {
        connection->AddOwner();
    }
    ~OperationOwner() { connection->ReleaseOwner(); }
    OperationOwner(const OperationOwner&) = delete;
    OperationOwner& operator=(const OperationOwner&) = delete;
private:
    std::shared_ptr<detail::InferenceConnection> connection;
};
}

// Defined in the Win32-only transport translation unit; the client and fake
// tests have no SYCL dependencies or GPU runtime imports.
std::shared_ptr<InferenceTransport> StartInferencePipeTransport(
    const std::filesystem::path&, const std::filesystem::path&,
    const std::string&, ExpectedInferenceIdentity&);

InferenceClient::InferenceClient(std::shared_ptr<detail::InferenceConnection> connection)
    : connection_(std::move(connection)) { connection_->AddOwner(); }
InferenceClient::~InferenceClient() { connection_->ReleaseOwner(); }

std::shared_ptr<InferenceClient> InferenceClient::Start(
    const std::filesystem::path& runtime, const std::filesystem::path& executable,
    const std::string& buildId, InferenceClientOptions options) {
    ExpectedInferenceIdentity expected;
    auto transport = StartInferencePipeTransport(runtime, executable, buildId, expected);
    return Connect(std::move(transport), std::move(expected), std::move(options));
}

std::shared_ptr<InferenceClient> InferenceClient::Connect(
    std::shared_ptr<InferenceTransport> transport, ExpectedInferenceIdentity identity, InferenceClientOptions options) {
    wire::ValidationError error;
    if (!transport || !wire::validate_limits(options.limits, error) || options.handshakeTimeout <= 0ms ||
        options.commandTimeout <= 0ms || options.shutdownTimeout <= 0ms) {
        throw InferenceClientError(ClientErrorCode::RemoteError, "Invalid inference client configuration: " + error.message,
                                   wire::ErrorCode::InvalidArgument);
    }
    auto connection = std::make_shared<detail::InferenceConnection>(std::move(transport), std::move(identity), std::move(options));
    connection->Initialize();
    try { return std::shared_ptr<InferenceClient>(new InferenceClient(connection)); }
    catch (...) { connection->Fail(ClientErrorCode::Closed, "Could not allocate inference client handle"); throw; }
}

std::shared_ptr<InferenceModel> InferenceClient::LoadQwen3(
    const std::string& directory, InferenceModelOptions options, std::chrono::milliseconds timeout) {
    OperationOwner operation(connection_);
    wire::Command command;
    command.method = wire::Method::ModelLoad;
    command.modelDirectory = directory;
    command.maxSequenceLength = options.maxSequenceLength;
    command.maxSessions = options.maxSessions;
    command.maxSessionCacheBytes = options.maxSessionCacheBytes;
    const auto response = connection_->Call(std::move(command), timeout);
    std::unique_ptr<InferenceModel> handle;
    try { handle.reset(new InferenceModel(connection_, response.model)); }
    catch (...) { connection_->ReleaseObject(wire::ObjectType::Model, response.model.modelId); throw; }
    return std::shared_ptr<InferenceModel>(std::move(handle));
}

wire::WorkerInfo InferenceClient::Query(std::chrono::milliseconds timeout) {
    wire::Command command;
    command.method = wire::Method::WorkerQuery;
    return connection_->Call(std::move(command), timeout).worker;
}

InferenceClientSnapshot InferenceClient::Snapshot() const {
    std::lock_guard<std::mutex> lock(connection_->mutex);
    const auto& core = *connection_;
    return {core.status, core.hello.epoch, core.outstanding, core.routes.size(), core.pendingCount,
            core.queued.size(), core.reservedTokens, core.ingressBytes, core.transportBytes,
            core.peakIngress, core.peakTransport, core.lastOrdinal, core.droppedLogs, core.failure};
}

void InferenceClient::Shutdown(std::chrono::milliseconds timeout) { connection_->Shutdown(timeout); }

InferenceModel::InferenceModel(std::shared_ptr<detail::InferenceConnection> connection, wire::ModelInfo info)
    : connection_(std::move(connection)), info_(std::move(info)), epoch_(connection_->hello.epoch) { connection_->AddOwner(); }
InferenceModel::~InferenceModel() { Release(); }
void InferenceModel::Release() noexcept {
    if (!released_.exchange(true)) {
        connection_->ReleaseObject(wire::ObjectType::Model, info_.modelId);
        connection_->ReleaseOwner();
    }
}
std::shared_ptr<InferenceSession> InferenceModel::CreateSession(std::chrono::milliseconds timeout) {
    OperationOwner operation(connection_);
    if (released_.load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference model is released");
    wire::Command command;
    command.method = wire::Method::SessionCreate;
    command.modelId = info_.modelId;
    command.maxSequenceLength = info_.maxSequenceLength;
    command.maxSessionCacheBytes = info_.sessionReservationBytes;
    const auto response = connection_->Call(std::move(command), timeout, false, {}, &released_);
    std::unique_ptr<InferenceSession> handle;
    try { handle.reset(new InferenceSession(connection_, response.session, info_.vocabSize)); }
    catch (...) { connection_->ReleaseObject(wire::ObjectType::Session, response.session.sessionId); throw; }
    return std::shared_ptr<InferenceSession>(std::move(handle));
}

InferenceSession::InferenceSession(std::shared_ptr<detail::InferenceConnection> connection,
                                   wire::SessionInfo info, uint32_t vocabSize)
    : connection_(std::move(connection)), info_(std::move(info)), vocabSize_(vocabSize), epoch_(connection_->hello.epoch) {
    connection_->AddOwner();
}
InferenceSession::~InferenceSession() { Release(); }
void InferenceSession::Release() noexcept {
    if (!released_.exchange(true)) {
        connection_->ReleaseObject(wire::ObjectType::Session, info_.sessionId);
        connection_->ReleaseOwner();
    }
}
std::shared_ptr<InferenceRequest> InferenceSession::Submit(
    std::vector<int32_t> tokens, InferenceRequestOptions options, std::chrono::milliseconds timeout) {
    OperationOwner operation(connection_);
    if (released_.load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference session is released");
    auto route = connection_->Submit(info_.modelId, info_.sessionId, vocabSize_, std::move(tokens), std::move(options), timeout, &released_);
    std::unique_ptr<InferenceRequest> handle;
    try { handle.reset(new InferenceRequest(connection_, route)); }
    catch (...) { connection_->ReleaseRequest(route); throw; }
    return std::shared_ptr<InferenceRequest>(std::move(handle));
}
wire::SessionInfo InferenceSession::Query(std::chrono::milliseconds timeout) {
    if (released_.load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference session is released");
    wire::Command command;
    command.method = wire::Method::SessionQuery;
    command.sessionId = info_.sessionId;
    command.modelId = info_.modelId;
    command.maxSequenceLength = info_.capacity;
    command.maxSessionCacheBytes = info_.cacheBytes;
    return connection_->Call(std::move(command), timeout, false, {}, &released_).session;
}

InferenceRequest::InferenceRequest(std::shared_ptr<detail::InferenceConnection> connection,
                                   std::shared_ptr<detail::InferenceRoute> route)
    : connection_(std::move(connection)), route_(std::move(route)) { connection_->AddOwner(); }
InferenceRequest::~InferenceRequest() { Release(); }
uint64_t InferenceRequest::Id() const noexcept { return route_->identity.inferenceId; }
const std::string& InferenceRequest::Epoch() const noexcept { return route_->epoch; }
void InferenceRequest::Release() noexcept {
    if (!released_.exchange(true)) {
        connection_->ReleaseRequest(route_);
        connection_->ReleaseOwner();
    }
}
InferenceReadResult InferenceRequest::Read(std::chrono::milliseconds timeout) { return connection_->Read(route_, timeout); }
bool InferenceRequest::Cancel(std::chrono::milliseconds timeout) {
    if (released_.load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference request is released");
    wire::Command command;
    command.method = wire::Method::RequestCancel;
    command.inferenceId = Id();
    return connection_->Call(std::move(command), timeout, false, route_, &released_).cancelRequested;
}
wire::RequestInfo InferenceRequest::Query(std::chrono::milliseconds timeout) {
    if (released_.load()) throw InferenceClientError(ClientErrorCode::InvalidHandle, "Inference request is released");
    wire::Command command;
    command.method = wire::Method::RequestQuery;
    command.inferenceId = Id();
    return connection_->Call(std::move(command), timeout, false, route_, &released_).request;
}
InferenceRequestSnapshot InferenceRequest::Snapshot() const {
    std::lock_guard<std::mutex> lock(connection_->mutex);
    const auto& route = *route_;
    InferenceRequestSnapshot result;
    result.inferenceId = route.identity.inferenceId;
    result.epoch = route.epoch;
    result.accepted = route.accepted;
    result.finalReceived = route.finalSeen;
    result.terminalReceived = route.terminalSeen;
    result.completionReady = route.complete;
    result.terminalConsumed = route.userTerminalConsumed;
    result.released = route.releaseWanted;
    result.deliveryAbandoned = route.abandoned;
    result.localFailure = route.localFailure;
    result.deliveredTokens = route.deliveredTokens;
    result.discardedPublishedTokens = route.discardedTokens;
    result.bufferedTokens = route.count;
    result.consumedSequence = route.consumed;
    result.acknowledgedSequence = route.acknowledged;
    result.summary = route.localFailure ? route.localSummary : route.terminalSeen ? route.terminal.summary : route.final.summary;
    return result;
}

} // namespace aila::runtime
