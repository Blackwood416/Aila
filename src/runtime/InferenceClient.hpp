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
    uint32_t maxDecodeBatchSize = 1; // Optional continuous_batching_v1, 1..4 and <= maxActiveRequests.
    std::chrono::milliseconds handshakeTimeout{2000};
    std::chrono::milliseconds commandTimeout{30000};
    std::chrono::milliseconds shutdownTimeout{10000};
};

struct InferenceModelOptions {
    uint32_t maxSequenceLength = 4096;
    uint32_t maxSessions = 4;
    uint64_t maxSessionCacheBytes = 0;
    std::optional<wire::AllocationLimits> allocationLimits;
    // "bf16" (default) or "int8" for backends that advertise an int8 KV
    // capability; must match the worker-side KV layout selection.
    std::string kvType = "bf16";
};

struct InferenceRequestOptions {
    wire::GenerationOptions generation;
    uint32_t prefillChunkSize = 512;
    std::vector<int32_t> stopTokenIds;
    std::optional<wire::ThinkingBudgetOptions> thinkingBudget;
    bool prefillOnly = false;
    bool forceHostSampling = false;
    bool legacyPrefillLogits = false;
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
    uint64_t queuedLogs = 0;
    uint64_t logQueueBytes = 0;
    uint64_t peakLogQueueBytes = 0;
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

struct InferenceLogEvent {
    uint64_t eventOrdinal = 0;
    int32_t level = 0;
    std::string epoch;
    std::string message;
};
struct InferenceLogReadResult {
    InferenceReadStatus status = InferenceReadStatus::Timeout;
    std::optional<InferenceLogEvent> event;
    uint64_t droppedLogs = 0; // Host queue drops, independent of Worker drops.
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
    // Explicit BF16 weights; LoadQwen35 retains the NF4 default. Both Hybrid
    // loaders use BF16 KV and the same Session lifecycle and allocation limits.
    std::shared_ptr<InferenceModel> LoadQwen35Bf16(
        const std::string& modelDirectory, InferenceModelOptions options = {},
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Qwen4Exp NF4 text execution. Uses BF16 KV and the same Session lifecycle
    // and allocation limits. Cross-session decode batching is not offered.
    std::shared_ptr<InferenceModel> LoadQwen4Exp(
        const std::string& modelDirectory, InferenceModelOptions options = {},
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    wire::WorkerInfo Query(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Optional runtime_memory_v1. One bounded RPC; a busy execution domain is
    // reported as Busy, never as a zero or previously cached allocation count.
    wire::RuntimeMemoryInfo QueryMemory(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    InferenceClientSnapshot Snapshot() const;
    // Bounded by limits.maxLogQueueBytes. Overflow drops new logs, never token
    // events. Poll with zero; queued logs remain readable after Shutdown/fault.
    InferenceLogReadResult ReadLog(std::chrono::milliseconds timeout);
    // Nonblocking release of this Client's log consumer; children stay usable.
    // Wakes a blocked ReadLog, discards queued logs and prevents re-opening.
    void CloseLogStream() noexcept;
    // Optional worker_log_control_v1; 0=debug, 1=info, 2=warning, 3=error.
    void SetLogLevel(uint32_t level, std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Idempotent and shared by concurrent callers. Stops new user operations,
    // auto-drains unread tokens, then validates the eventOrdinal close barrier.
    void Shutdown(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

private:
    explicit InferenceClient(std::shared_ptr<detail::InferenceConnection> connection);
    std::shared_ptr<InferenceModel> LoadModel(
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
    // Optional prefill_only_v1. Appends a prefix without sampling or producing
    // token events. The returned Request supports the same read/cancel/release
    // contract as generation; its sole event is Completed/Length on success.
    std::shared_ptr<InferenceRequest> Prefill(
        std::vector<int32_t> promptIds, uint32_t chunkSize = 512,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    wire::SessionInfo Query(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Optional session_context_v1. Busy Sessions reject mutation. Timeout only
    // ends this RPC wait; a submitted operation may still complete remotely.
    wire::SessionInfo Reset(std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    // Returns actual restored checkpoint length, possibly below requested.
    wire::SessionInfo Truncate(uint32_t length,
        std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());
    void Release() noexcept;

private:
    friend class InferenceModel;
    InferenceSession(std::shared_ptr<detail::InferenceConnection> connection,
                     wire::SessionInfo info, uint32_t vocabSize);
    wire::SessionInfo ContextOperation(wire::Method method, uint32_t length, std::chrono::milliseconds timeout);
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
