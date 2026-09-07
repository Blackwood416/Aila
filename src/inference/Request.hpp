#pragma once

#include "inference/Session.hpp"

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

struct RequestOptions {
    GenerationConfig generation;
    int prefillChunkSize = 512;
    std::vector<int> stopTokenIds;
};

struct StepResult {
    RequestStatus status = RequestStatus::Queued;
    std::optional<int> tokenId;
    bool becameTerminal = false;
};

struct RequestSnapshot {
    RequestStatus status = RequestStatus::Queued;
    size_t promptTokensProcessed = 0;
    std::vector<int> generatedIds;
    std::string error;
    std::optional<ErrorCode> errorCode;
};

// A bounded-size progress read for schedulers; it does not copy token history.
struct RequestProgress {
    RequestStatus status = RequestStatus::Queued;
    size_t promptTokensProcessed = 0;
    size_t generatedTokens = 0;
    std::optional<ErrorCode> errorCode;
    std::array<char, 1024> errorText{};
    size_t errorSize = 0;
    std::string_view ErrorMessage() const noexcept { return {errorText.data(), errorSize}; }
};

// Low-level token request: appends the supplied prompt to the session cache.
// One Step submits at most one prefill chunk or one decode token. Chat
// formatting, reasoning budgets and stream parsing remain in the legacy
// driver until migrated; unsupported thinking budgets are rejected here.
class RequestState final : public std::enable_shared_from_this<RequestState> {
public:
    static std::shared_ptr<RequestState> Create(
        std::shared_ptr<SessionState> session,
        std::vector<int> promptIds,
        RequestOptions options = {});
    ~RequestState();
    RequestState(const RequestState&) = delete;
    RequestState& operator=(const RequestState&) = delete;

    StepResult Step();
    void Cancel() noexcept { cancelRequested.store(true); }
    RequestSnapshot Snapshot() const;
    RequestProgress Progress() const;

private:
    RequestState(std::shared_ptr<SessionState> session,
                 std::vector<int> promptIds, RequestOptions options);
    StepResult Finish(RequestStatus status, std::string_view error = {},
                      std::optional<ErrorCode> errorCode = std::nullopt);

    std::shared_ptr<SessionState> session;
    std::vector<int> promptIds;
    RequestOptions options;
    std::mt19937 rng;
    std::atomic<bool> stepInProgress{false};
    mutable std::mutex stepMutex;
    std::atomic<bool> cancelRequested{false};
    RequestSnapshot snapshot;
    int pendingToken = -1;
    bool ownsSession = false;
};

} // namespace aila::inference
