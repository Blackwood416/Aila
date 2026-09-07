#include "inference/Session.hpp"
#include "profile/Profiling.hpp"

#include <utility>

namespace aila::inference {
namespace {

int CheckedContextLength(const SessionBackend& backend, int capacity) {
    const int length = backend.ContextLength();
    if (length < 0 || length > capacity) {
        throw Error(ErrorCode::RuntimeFault,
                    "Session backend reported an invalid context length");
    }
    return length;
}

void ValidateTokens(const std::vector<int>& tokenIds, int vocabSize) {
    if (tokenIds.empty()) {
        throw Error(ErrorCode::InvalidArgument, "Token input must not be empty");
    }
    for (const int tokenId : tokenIds) {
        if (tokenId < 0 || tokenId >= vocabSize) {
            throw Error(ErrorCode::InvalidArgument,
                        "Input token ID is outside the model vocabulary");
        }
    }
}

} // namespace

SessionState::SessionState(std::unique_ptr<SessionBackend> backend)
    : backend(std::move(backend)),
      capacity(this->backend ? this->backend->Capacity() : 0),
      vocabSize(this->backend ? this->backend->VocabSize() : 0),
      cacheBytes(this->backend ? this->backend->CacheBytes() : 0) {
    if (!this->backend) {
        throw Error(ErrorCode::InvalidArgument, "Session backend must not be null");
    }
    if (capacity <= 0 || vocabSize <= 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "Session capacity and vocabulary size must be positive");
    }
    CheckedContextLength(*this->backend, capacity);
}

int SessionState::ContextLength() const {
    std::lock_guard<std::mutex> lock(operationMutex);
    return CheckedContextLength(*backend, capacity);
}

void SessionState::Acquire() {
    bool expected = false;
    if (!busy.compare_exchange_strong(expected, true,
                                      std::memory_order_acquire,
                                      std::memory_order_relaxed)) {
        throw Error(ErrorCode::Busy, "Session already has an active operation or request");
    }
}

void SessionState::Release() noexcept {
    busy.store(false, std::memory_order_release);
}

void SessionState::Reset() {
    aila::ScopedLogDeferral deferLogs;
    Acquire();
    try {
        std::lock_guard<std::mutex> lock(operationMutex);
        backend->Reset();
        if (CheckedContextLength(*backend, capacity) != 0) {
            throw Error(ErrorCode::RuntimeFault, "Session backend did not reset its context");
        }
    } catch (...) {
        Release();
        throw;
    }
    Release();
}

int SessionState::Truncate(int length) {
    aila::ScopedLogDeferral deferLogs;
    Acquire();
    int restoredLength = 0;
    try {
        std::lock_guard<std::mutex> lock(operationMutex);
        const int currentLength = CheckedContextLength(*backend, capacity);
        if (length < 0 || length > currentLength) {
            throw Error(ErrorCode::InvalidArgument,
                        "Truncate length must be between zero and the current context length");
        }
        restoredLength = backend->Truncate(length);
        if (restoredLength < 0 || restoredLength > length ||
            CheckedContextLength(*backend, capacity) != restoredLength) {
            throw Error(ErrorCode::RuntimeFault,
                        "Session backend reported an invalid restored context length");
        }
    } catch (...) {
        Release();
        throw;
    }
    Release();
    return restoredLength;
}

std::vector<float> SessionState::ForwardLogits(const std::vector<int>& tokenIds) {
    aila::ScopedLogDeferral deferLogs;
    Acquire();
    try {
        auto result = Execute(tokenIds, nullptr, true);
        Release();
        return std::move(result.logits);
    } catch (...) {
        Release();
        throw;
    }
}

ExecutionResult SessionState::Execute(const std::vector<int>& tokenIds,
                                      const SamplingInput* sampling,
                                      bool captureLogits) {
    std::lock_guard<std::mutex> lock(operationMutex);
    ValidateTokens(tokenIds, vocabSize);
    const int currentLength = CheckedContextLength(*backend, capacity);
    const size_t available = static_cast<size_t>(capacity - currentLength);
    if (tokenIds.size() > available) {
        throw Error(ErrorCode::ContextOverflow, "Token input exceeds the session context capacity");
    }

    auto result = backend->Forward(tokenIds, sampling, captureLogits);
    const int expectedLength = currentLength + static_cast<int>(tokenIds.size());
    if (CheckedContextLength(*backend, capacity) != expectedLength) {
        throw Error(ErrorCode::RuntimeFault,
                    "Session backend did not consume the complete input token sequence");
    }
    if (sampling && (result.tokenId < 0 || result.tokenId >= vocabSize)) {
        throw Error(ErrorCode::RuntimeFault,
                    "Session backend sampled a token outside the model vocabulary");
    }
    if (captureLogits && result.logits.size() != static_cast<size_t>(vocabSize)) {
        throw Error(ErrorCode::RuntimeFault,
                    "Session backend returned an invalid logits vector size");
    }
    return result;
}

} // namespace aila::inference
