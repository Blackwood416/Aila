#pragma once

#include "inference/Session.hpp"
#include "generation/ThinkingBudget.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace aila::inference {

enum class RequestStatus {
    Queued,
    Prefill,
    Decode,
    Finalizing,
    Completed,
    Cancelled,
    Failed,
};

bool IsTerminal(RequestStatus status) noexcept;

enum class RequestEndReason {
    None,
    Length,
    StopToken,
    Cancelled,
    OutputBackpressure,
    SchedulerStopped,
    RuntimeFault,
    Failed,
};

struct RequestOptions {
    GenerationConfig generation;
    int prefillChunkSize = 512;
    std::vector<int> stopTokenIds;
    std::optional<aila::generation::ThinkingBudgetOptions> thinkingBudget;
    // Explicit replay request. Requires max_new_tokens == 0, no stop tokens
    // and no thinking budget. Uses normal scheduler admission and cancellation.
    bool prefillOnly = false;
    bool forceHostSampling = false;
    bool legacyPrefillLogits = false;
};

struct StepResult {
    RequestStatus status = RequestStatus::Queued;
    std::optional<int> tokenId;
    bool becameTerminal = false;
};

struct RequestSnapshot {
    RequestStatus status = RequestStatus::Queued;
    RequestEndReason endReason = RequestEndReason::None;
    size_t promptTokensProcessed = 0;
    std::vector<int> generatedIds;
    std::string error;
    std::optional<ErrorCode> errorCode;
};

// A bounded-size progress read for schedulers; it does not copy token history.
struct RequestProgress {
    RequestStatus status = RequestStatus::Queued;
    RequestEndReason endReason = RequestEndReason::None;
    size_t promptTokensProcessed = 0;
    size_t generatedTokens = 0;
    std::optional<ErrorCode> errorCode;
    std::array<char, 1024> errorText{};
    size_t errorSize = 0;
    std::string_view ErrorMessage() const noexcept { return {errorText.data(), errorSize}; }
};

// Low-level token request: appends the supplied prompt to the session cache.
// One Step publishes at most one token. A decode window may consume two
// positions and retain an uncached, unpublished bonus for the next Step. Chat
// formatting and stream parsing belong to the caller. Thinking budgets require
// explicit tokenizer metadata. Forced close tokens consume generation slots
// and KV positions, but no sampling RNG; insufficient room for the complete
// closing sequence finishes with Length after flushing the last emitted token.
// A prefill-only request emits no tokens, consumes no sampling RNG, and ends
// with Completed/Length after its last prompt chunk has completed.
class RequestState final : public std::enable_shared_from_this<RequestState> {
public:
    static std::shared_ptr<RequestState> Create(
        std::shared_ptr<SessionState> session,
        std::vector<int> promptIds,
        RequestOptions options = {});
    ~RequestState();
    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;

    StepResult Step(size_t outputSlots = 2);
    // Stable model identity; null means this backend has no decode-batch path.
    // Does not wait for an executing step or inspect mutable request state.
    const void* DecodeBatchKey() const noexcept { return session->DecodeBatchKey(); }
    // Each live row must be in Decode or Finalizing and belong to the same
    // batch-capable model. Cancelled/terminal rows do not submit device work.
    // Results retain input order. One row uses Step's original execution path.
    // Batch execution failures fail all submitted rows; RuntimeFault remains
    // visible in each Progress so a Scheduler can stop its whole device domain.
    static std::vector<StepResult> StepDecodeBatch(
        const std::vector<std::shared_ptr<RequestState>>& requests);
    void Cancel() noexcept { cancelRequested.store(true); }
    RequestSnapshot Snapshot() const;
    RequestProgress Progress() const;

private:
    RequestState(std::shared_ptr<SessionState> session,
                 std::vector<int> promptIds, RequestOptions options);
    StepResult Finish(RequestStatus status, std::string_view error = {},
                      std::optional<ErrorCode> errorCode = std::nullopt,
                      RequestEndReason endReason = RequestEndReason::None);
    StepResult StepLocked(size_t outputSlots);
    bool PrepareThinkingClose();
    StepResult PublishSample(ExecutionResult result, bool closeExceedsRemaining);

    std::shared_ptr<SessionState> session;
    std::vector<int> promptIds;
    RequestOptions options;
    std::mt19937 rng;
    std::atomic<bool> stepInProgress{false};
    mutable std::mutex stepMutex;
    std::atomic<bool> cancelRequested{false};
    RequestSnapshot snapshot;
    int pendingToken = -1;
    // Verified but unpublished and uncached. The preceding output was already
    // committed by ExecuteWindow, so it must never be forwarded a second time.
    std::optional<int> verifiedBonus;
    bool usedDecodeWindow = false;
    bool ownsSession = false;
    bool insideThinking = false;
    size_t thinkingTokensUsed = 0;
    bool forcingThinkingClose = false;
    bool thinkingCloseWasForced = false;
    size_t thinkingCloseIndex = 0;
};

} // namespace aila::inference
