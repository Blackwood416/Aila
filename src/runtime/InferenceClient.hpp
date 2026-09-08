#pragma once

#include "ipc/InferenceProtocol.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace aila::runtime {

namespace wire = aila::ipc::v2;
namespace detail {
struct InferenceConnection;
struct InferenceRoute;
}

enum class ClientErrorCode { Timeout, InvalidHandle, ResourceLimit, Closed, ProtocolError, RemoteError };

class InferenceClientError : public std::runtime_error {
public:
    InferenceClientError(ClientErrorCode code, const std::string& message,
                         std::optional<wire::ErrorCode> remoteCode = {})
        : std::runtime_error(message), code_(code), remoteCode_(remoteCode) {}
    ClientErrorCode Code() const noexcept { return code_; }
    std::optional<wire::ErrorCode> RemoteCode() const noexcept { return remoteCode_; }

private:
    ClientErrorCode code_;
    std::optional<wire::ErrorCode> remoteCode_;
};

struct ExpectedInferenceIdentity {
    std::string buildId;
    std::string executable;
    std::string runtimeDirectory;
    std::string currentDirectory;
    std::string path;
};

// Blocking transport seam for the isolated Win32 process and controlled CPU
// tests. One permanent reader per channel and one command writer call these.
// Stop must promptly wake all blocked calls, including a blocked WriteCommand.
// No transport implementation may invoke user callbacks from its I/O threads.
class InferenceTransport {
public:
    virtual ~InferenceTransport() = default;
    virtual bool ReadResponse(ipc::Frame& frame, std::string& error) = 0;
    virtual bool ReadEvent(ipc::Frame& frame, std::string& error) = 0;
    virtual bool WriteCommand(const ipc::Frame& frame, std::string& error) = 0;
    virtual void SetLimits(const wire::Limits& limits) = 0;
    virtual void Stop(bool force) noexcept = 0;
    virtual bool WaitForExit(std::chrono::milliseconds, std::string&) { return true; }
};

struct InferenceClientOptions {
    wire::Limits limits;
    std::vector<std::string> requiredCapabilities = wire::default_capabilities();
    std::chrono::milliseconds handshakeTimeout{2000};
    std::chrono::milliseconds commandTimeout{30000};
    std::chrono::milliseconds shutdownTimeout{10000};
};

struct InferenceModelOptions {
    uint32_t maxSequenceLength = 4096;
    uint32_t maxSessions = 4;
    uint64_t maxSessionCacheBytes = 0;
};

struct InferenceRequestOptions {
    wire::GenerationOptions generation;
    uint32_t prefillChunkSize = 512;
    std::vector<int32_t> stopTokenIds;
};

struct InferenceClientSnapshot {
    wire::DomainStatus status = wire::DomainStatus::Starting;
    std::string epoch;
    uint64_t outstandingRequests = 0;
    uint64_t retainedRequestHandles = 0;
    uint64_t pendingCommands = 0;
    uint64_t queuedCommands = 0;
    uint64_t reservedTokens = 0;
    uint64_t ingressBytes = 0;
    uint64_t transportBytes = 0;
    uint64_t peakIngressBytes = 0;
    uint64_t peakTransportBytes = 0;
    uint64_t receivedEventOrdinal = 0;
    uint64_t droppedLogs = 0;
    std::string error;
};

struct InferenceRequestSnapshot {
    uint64_t inferenceId = 0;
    std::string epoch;
    bool accepted = false;
    bool finalReceived = false;
    bool terminalReceived = false;
    bool completionReady = false;
    bool terminalConsumed = false;
    bool released = false;
    bool deliveryAbandoned = false;
    bool localFailure = false;
    uint64_t deliveredTokens = 0;
    uint64_t discardedPublishedTokens = 0;
    uint64_t bufferedTokens = 0;
    uint64_t consumedSequence = 0;
    uint64_t acknowledgedSequence = 0;
    wire::Summary summary;
};

enum class InferenceReadStatus { Event, Timeout, Closed };
struct InferenceReadEvent {
    bool terminal = false;
    bool localFailure = false; // Never represents an invented wire terminal.
    uint64_t sequence = 0;
    int32_t tokenId = -1;
    wire::Summary summary;
    bool deliveryAbandoned = false;
    uint64_t deliveredTokens = 0;
    uint64_t discardedPublishedTokens = 0;
};
struct InferenceReadResult {
    InferenceReadStatus status = InferenceReadStatus::Timeout;
    std::optional<InferenceReadEvent> event;
};

class InferenceModel;
class InferenceSession;
class InferenceRequest;

// Repository-internal token API. It does not load SYCL or expose public C ABI
// changes. Child handles keep the same connection and epoch alive after the
// client/model/session wrapper has been released. No automatic Worker retry.
class InferenceClient final {
public:
    static std::shared_ptr<InferenceClient> Start(
        const std::filesystem::path& runtimeDirectory,
        const std::filesystem::path& workerExecutable, const std::string& buildId,
        InferenceClientOptions options = {});
    static std::shared_ptr<InferenceClient> Connect(
        std::shared_ptr<InferenceTransport> transport, ExpectedInferenceIdentity identity,
        InferenceClientOptions options = {});
    ~InferenceClient();
    InferenceClient(const InferenceClient&) = delete;
    InferenceClient& operator=(const InferenceClient&) = delete;

    std::shared_ptr<InferenceModel> LoadQwen3(
        const std::string& modelDirectory, InferenceModelOptions options = {},
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Hybrid token sessions retain independent recurrent state and host
    // mirrors. The remote API has no rollback operation, so snapshots are off.
    std::shared_ptr<InferenceModel> LoadQwen35(
        const std::string& modelDirectory, InferenceModelOptions options = {},
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    wire::WorkerInfo Query(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    InferenceClientSnapshot Snapshot() const;
    // Idempotent and shared by concurrent callers. Stops new user operations,
    // auto-drains unread tokens, then validates the eventOrdinal close barrier.
    void Shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

private:
    explicit InferenceClient(std::shared_ptr<detail::InferenceConnection> connection);
    std::shared_ptr<InferenceModel> LoadNf4(
        const std::string& backend, const std::string& modelDirectory,
        InferenceModelOptions options, std::chrono::milliseconds timeout);
    std::shared_ptr<detail::InferenceConnection> connection_;
};

class InferenceModel final {
public:
    ~InferenceModel();
    InferenceModel(const InferenceModel&) = delete;
    InferenceModel& operator=(const InferenceModel&) = delete;
    const wire::ModelInfo& Info() const noexcept { return info_; }
    const std::string& Epoch() const noexcept { return epoch_; }
    std::shared_ptr<InferenceSession> CreateSession(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    void Release() noexcept;

private:
    friend class InferenceClient;
    InferenceModel(std::shared_ptr<detail::InferenceConnection> connection, wire::ModelInfo info);
    std::shared_ptr<detail::InferenceConnection> connection_;
    wire::ModelInfo info_;
    std::string epoch_;
    std::atomic<bool> released_{false};
};

class InferenceSession final {
public:
    ~InferenceSession();
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;
    const wire::SessionInfo& Info() const noexcept { return info_; }
    const std::string& Epoch() const noexcept { return epoch_; }
    std::shared_ptr<InferenceRequest> Submit(
        std::vector<int32_t> promptIds, InferenceRequestOptions options = {},
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    wire::SessionInfo Query(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    void Release() noexcept;

private:
    friend class InferenceModel;
    InferenceSession(std::shared_ptr<detail::InferenceConnection> connection,
                     wire::SessionInfo info, uint32_t vocabSize);
    std::shared_ptr<detail::InferenceConnection> connection_;
    wire::SessionInfo info_;
    uint32_t vocabSize_;
    std::string epoch_;
    std::atomic<bool> released_{false};
};

class InferenceRequest final {
public:
    ~InferenceRequest();
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
    uint64_t Id() const noexcept;
    const std::string& Epoch() const noexcept;
    // Single consumer. Timeout affects this wait only; it never kills Worker.
    InferenceReadResult Read(std::chrono::milliseconds timeout);
    bool Cancel(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    wire::RequestInfo Query(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    InferenceRequestSnapshot Snapshot() const;
    // Nonblocking. Cancel + ACK/drop are completed by the private connection.
    void Release() noexcept;

private:
    friend class InferenceSession;
    InferenceRequest(std::shared_ptr<detail::InferenceConnection> connection,
                     std::shared_ptr<detail::InferenceRoute> route);
    std::shared_ptr<detail::InferenceConnection> connection_;
    std::shared_ptr<detail::InferenceRoute> route_;
    std::atomic<bool> released_{false};
};

} // namespace aila::runtime
