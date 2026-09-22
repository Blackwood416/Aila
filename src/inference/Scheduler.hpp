#pragma once

#include "inference/Request.hpp"

#include <chrono>
#include <cstdint>
#include <string_view>

namespace aila::inference {

namespace detail {
struct SchedulerState;
struct ScheduledRequestState;
}

struct SchedulerOptions {
    size_t maxActiveRequests = 4;
    // Opt-in continuous decode batching, bounded by maxActiveRequests and 4.
    // One retains the original round-robin execution/numerical path.
    size_t maxDecodeBatchSize = 1;
    // Includes queued/running requests and completed streams whose terminal
    // event has not been read. Reading the terminal or dropping the handle
    // returns the reservation after execution has safely ended.
    size_t maxOutstandingRequests = 64;
    size_t maxPromptTokens = 4096;
    size_t maxGeneratedTokens = 1024;
    size_t maxStopTokenIds = 64;
    size_t maxOutstandingTokens = 32768; // Sum of prompt + max new tokens.
    int maxPrefillChunkSize = 512;
    size_t maxBufferedTokensPerRequest = 128;
    // Optional admission limit for the already allocated Session caches.
    // This is not a total VRAM limit; model/workspace memory is excluded.
    size_t maxSessionCacheBytes = 0;
};

enum class RequestEventKind { Token, Terminal };

struct RequestEvent {
    RequestEventKind kind = RequestEventKind::Token;
    uint64_t requestId = 0;
    uint64_t sequence = 0; // One-based, including the terminal event.
    int tokenId = -1; // Set only for Token events.
    RequestStatus status = RequestStatus::Queued;
    RequestEndReason endReason = RequestEndReason::None;
    size_t promptTokensProcessed = 0;
    size_t generatedTokens = 0; // Tokens published to this stream.
    std::optional<ErrorCode> errorCode;
    std::string error; // At most 1024 bytes.
};

enum class ReadStatus { Event, Timeout, Closed };

struct RequestReadResult {
    ReadStatus status = ReadStatus::Timeout;
    std::optional<RequestEvent> event;
};

struct ScheduledRequestSnapshot {
    uint64_t requestId = 0;
    RequestStatus status = RequestStatus::Queued;
    RequestEndReason endReason = RequestEndReason::None;
    size_t promptTokensProcessed = 0;
    size_t generatedTokens = 0;
    size_t bufferedTokens = 0;
    bool terminalConsumed = false;
    std::optional<ErrorCode> errorCode;
    std::string error;
};

struct SchedulerSnapshot {
    bool accepting = false;
    bool stopping = false;
    bool faulted = false;
    size_t outstandingRequests = 0;
    size_t queuedRequests = 0;
    size_t activeRequests = 0;
    size_t reservedTokens = 0;
    size_t reservedSessionCacheBytes = 0;
    std::string faultMessage; // First runtime failure, at most 1024 UTF-8 bytes.
};

// A thread-safe, single-consumer stream. Cancel is nonblocking and takes
// effect at a complete forward boundary. Dropping the last handle cancels
// its request. The Scheduler never calls transport/user output callbacks.
class ScheduledRequest final {
public:
    ~ScheduledRequest();
    ScheduledRequest(const ScheduledRequest&) = delete;
    ScheduledRequest& operator=(const ScheduledRequest&) = delete;

    uint64_t Id() const noexcept;
    void Cancel() noexcept;
    RequestReadResult Read(std::chrono::milliseconds timeout);
    ScheduledRequestSnapshot Snapshot() const;

private:
    friend class Scheduler;
    explicit ScheduledRequest(std::shared_ptr<detail::ScheduledRequestState> state);
    std::shared_ptr<detail::ScheduledRequestState> state;
};

// One submission thread per Scheduler, round-robin at complete forward
// boundaries (one prefill chunk or a compatible decode/finalization batch).
// Each batch is rebuilt from ready requests of the same LoadedModel; prefill
// remains chunked and cannot starve behind repeatedly formed decode batches.
// Use one
// Scheduler for a device execution domain. A RuntimeFault stops admission
// and fails its remaining requests; ordinary request errors are isolated.
//
// A full output ring cancels only that request before its next token-producing
// step. Its buffered tokens and reserved terminal event remain readable.
// Sessions are supplied by the caller; their model lifetime/count limits
// continue to be enforced by LoadedModel.
class Scheduler final {
public:
    explicit Scheduler(SchedulerOptions options = {});
    ~Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    std::shared_ptr<ScheduledRequest> Submit(
        std::shared_ptr<SessionState> session,
        std::vector<int> promptIds,
        RequestOptions options = {});
    SchedulerSnapshot Snapshot() const;

    // Thread-safe and idempotent. A lifecycle/reaper executor uses this after
    // observing a DeviceRuntime fault outside Submit/Step. Stops admission
    // immediately, without joining or waiting for a device step. Unfinished
    // requests fail at a safe boundary; existing terminal events stay intact.
    void ReportRuntimeFault(std::string_view message) noexcept;

    // Idempotent. Cancels queued/running requests and joins the submission
    // thread without waiting for readers. Called reentrantly from that thread
    // (e.g. a log callback), it requests shutdown and returns without self-join.
    // Handles remain readable after shutdown or Scheduler destruction.
    void Shutdown() noexcept;

private:
    std::shared_ptr<detail::SchedulerState> state;
};

} // namespace aila::inference
