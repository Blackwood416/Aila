#pragma once

#include "engine/Types.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <optional>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>

namespace aila::inference {

enum class ErrorCode {
    InvalidArgument,
    Busy,
    ContextOverflow,
    UnsupportedModel,
    ResourceLimit,
    RuntimeFault,
    SchedulerStopped,
};

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), errorCode(code) {}
    ErrorCode Code() const noexcept { return errorCode; }

private:
    ErrorCode errorCode;
};

// Borrowed only for the duration of a synchronous execution step. The RNG
// belongs to the request, even when the execution slot is shared.
struct SamplingInput {
    const GenerationConfig& config;
    const std::vector<int>& generatedIds;
    std::mt19937& rng;
    bool forceHost = false;
    bool legacyPrefillLogits = false;
};

struct ExecutionResult {
    int tokenId = -1; // No sample was requested.
    std::vector<float> logits; // Populated only when explicitly requested.
};

// A single-sequence two-position window. The incoming pending token is always
// consumed. An accepted first output may also be cached; the bonus never is.
// The selector runs synchronously before prefix commit, under execution locks.
struct DecodeWindowResult {
    int tokenId = -1;
    std::optional<int> bonusTokenId;
    int consumedInputs = 1;
    int cacheLength = 0;
    bool firstTokenCached = false;
};
using DecodeWindowSelector = std::function<bool(int)>;

inline constexpr size_t MaxDecodeBatchSize = 4;

class SessionBackend;

// One independent sequence per row. The token is consumed at that session's
// own position; row count must never be passed to a single-sequence Forward.
// All pointers are borrowed until the synchronous batch has drained.
struct BackendDecodeInput {
    SessionBackend* backend = nullptr;
    int tokenId = -1;
    const SamplingInput* sampling = nullptr;
    bool captureLogits = false;
};

// Device-independent seam for a real model session and CPU test executors.
// Forward consumes its input and finishes all device work before returning.
// Implementations retain the model/runtime and serialize shared workspaces.
// SessionState serializes mutable per-session state and ContextLength reads;
// its constructor reads metadata before publishing the session to callers.
class SessionBackend {
public:
    virtual ~SessionBackend() = default;
    virtual int ContextLength() const = 0;
    virtual int Capacity() const = 0;
    virtual int VocabSize() const = 0;
    virtual size_t CacheBytes() const = 0;
    virtual void Reset() = 0;
    virtual int Truncate(int length) = 0;
    virtual ExecutionResult Forward(const std::vector<int>& tokenIds,
                                    const SamplingInput* sampling,
                                    bool captureLogits) = 0;
    // Identity of immutable weights, adapter combination and execution slot.
    // A null key explicitly disables grouping; matching model paths/families
    // alone are insufficient. The key remains stable for this backend's life.
    virtual const void* DecodeBatchKey() const noexcept { return nullptr; }
    virtual bool SupportsDecodeWindow() const noexcept { return false; }
    virtual std::optional<DecodeWindowResult> ForwardWindow(int pendingToken,
        const DecodeWindowSelector& acceptFirst) { return std::nullopt; }
    virtual std::vector<ExecutionResult> ForwardDecodeBatch(
        const std::vector<BackendDecodeInput>& rows);
};

class RequestState;
class SessionState;

struct SessionDecodeInput {
    SessionState* session = nullptr;
    int tokenId = -1;
    const SamplingInput* sampling = nullptr;
    bool captureLogits = false;
};

// A session has one active request. Read-only snapshots may wait for a step;
// mutations fail with Busy instead of changing an active request's cache.
class SessionState final {
public:
    explicit SessionState(std::unique_ptr<SessionBackend> backend);
    SessionState(const SessionState&) = delete;
    SessionState& operator=(const SessionState&) = delete;

    int ContextLength() const;
    int Capacity() const noexcept { return capacity; }
    int VocabSize() const noexcept { return vocabSize; }
    size_t CacheBytes() const noexcept { return cacheBytes; }
    const void* DecodeBatchKey() const noexcept { return backend->DecodeBatchKey(); }
    bool Busy() const noexcept { return busy.load(); }
    bool SupportsDecodeWindow() const noexcept { return backend->SupportsDecodeWindow(); }

    void Reset();
    int Truncate(int length);
    std::vector<float> ForwardLogits(const std::vector<int>& tokenIds);
    // Explicit diagnostic/model API: acquires every idle session atomically
    // with respect to execution (Busy rolls all acquired leases back). Input
    // validation finishes for all rows before any backend work is submitted.
    static std::vector<std::vector<float>> ForwardDecodeBatchLogits(
        const std::vector<std::shared_ptr<SessionState>>& sessions,
        const std::vector<int>& tokenIds);

private:
    friend class RequestState;
    void Acquire();
    void Release() noexcept;
    std::optional<DecodeWindowResult> ExecuteWindow(int pendingToken,
        const DecodeWindowSelector& acceptFirst);
    void ResetCancelledWindow();
    ExecutionResult Execute(const std::vector<int>& tokenIds,
                            const SamplingInput* sampling,
                            bool captureLogits = false);
    static std::vector<ExecutionResult> ExecuteDecodeBatch(
        const std::vector<SessionDecodeInput>& rows);

    std::unique_ptr<SessionBackend> backend;
    const int capacity;
    const int vocabSize;
    const size_t cacheBytes;
    mutable std::mutex operationMutex;
    std::atomic<bool> busy{false};
};

} // namespace aila::inference
