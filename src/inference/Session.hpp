#pragma once

#include "engine/Types.hpp"

#include <atomic>
#include <cstddef>
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
};

struct ExecutionResult {
    int tokenId = -1; // No sample was requested.
    std::vector<float> logits; // Populated only when explicitly requested.
};

// Device-independent seam for a real model session and CPU test executors.
// Forward consumes its input and finishes all device work before returning.
// Implementations retain the model/runtime and serialize shared workspaces.
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
};

class RequestState;

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
    bool Busy() const noexcept { return busy.load(); }

    void Reset();
    int Truncate(int length);
    std::vector<float> ForwardLogits(const std::vector<int>& tokenIds);

private:
    friend class RequestState;
    void Acquire();
    void Release() noexcept;
    ExecutionResult Execute(const std::vector<int>& tokenIds,
                            const SamplingInput* sampling,
                            bool captureLogits = false);

    std::unique_ptr<SessionBackend> backend;
    const int capacity;
    const int vocabSize;
    const size_t cacheBytes;
    mutable std::mutex operationMutex;
    std::atomic<bool> busy{false};
};

} // namespace aila::inference
