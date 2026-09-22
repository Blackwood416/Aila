#include "inference/Session.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <functional>
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

std::vector<ExecutionResult> SessionBackend::ForwardDecodeBatch(
    const std::vector<BackendDecodeInput>& rows) {
    if (rows.size() == 1 && rows[0].backend == this) {
        return {Forward({rows[0].tokenId}, rows[0].sampling, rows[0].captureLogits)};
    }
    throw Error(ErrorCode::UnsupportedModel, "Session backend does not support decode batching");
}

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

std::vector<ExecutionResult> SessionState::ExecuteDecodeBatch(
    const std::vector<SessionDecodeInput>& rows) {
    if (rows.empty() || rows.size() > MaxDecodeBatchSize) {
        throw Error(ErrorCode::InvalidArgument, "Decode batch must contain between one and four rows");
    }
    std::vector<SessionState*> ordered;
    ordered.reserve(rows.size());
    for (const auto& row : rows) {
        if (!row.session) throw Error(ErrorCode::InvalidArgument, "Decode batch session must not be null");
        ordered.push_back(row.session);
    }
    std::sort(ordered.begin(), ordered.end(), std::less<SessionState*>{});
    if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end()) {
        throw Error(ErrorCode::InvalidArgument, "Decode batch contains a duplicate session");
    }
    if (rows.size() == 1) {
        const auto& row = rows.front();
        return {row.session->Execute({row.tokenId}, row.sampling, row.captureLogits)};
    }
    const void* key = rows.front().session->DecodeBatchKey();
    if (!key) throw Error(ErrorCode::UnsupportedModel, "Session backend does not support decode batching");
    for (const auto& row : rows) {
        if (row.session->DecodeBatchKey() != key) {
            throw Error(ErrorCode::InvalidArgument, "Decode batch sessions must share the same loaded model and adapter");
        }
    }

    // Stable lock ordering also protects direct diagnostic batch callers.
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(ordered.size());
    for (auto* session : ordered) locks.emplace_back(session->operationMutex);
    std::vector<int> positions;
    std::vector<BackendDecodeInput> inputs;
    positions.reserve(rows.size());
    inputs.reserve(rows.size());
    for (const auto& row : rows) {
        auto& session = *row.session;
        if (row.tokenId < 0 || row.tokenId >= session.vocabSize) {
            throw Error(ErrorCode::InvalidArgument, "Decode batch token ID is outside the model vocabulary");
        }
        const int position = CheckedContextLength(*session.backend, session.capacity);
        if (position == session.capacity) {
            throw Error(ErrorCode::ContextOverflow, "Decode batch token exceeds the session context capacity");
        }
        positions.push_back(position);
        inputs.push_back({session.backend.get(), row.tokenId, row.sampling, row.captureLogits});
    }
    auto results = rows.front().session->backend->ForwardDecodeBatch(inputs);
    if (results.size() != rows.size()) {
        throw Error(ErrorCode::RuntimeFault, "Session backend returned an invalid decode batch row count");
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& session = *rows[i].session;
        if (CheckedContextLength(*session.backend, session.capacity) != positions[i] + 1) {
            throw Error(ErrorCode::RuntimeFault, "Session backend did not consume exactly one token per decode batch row");
        }
        if (rows[i].sampling && (results[i].tokenId < 0 || results[i].tokenId >= session.vocabSize)) {
            throw Error(ErrorCode::RuntimeFault, "Session backend sampled a token outside the model vocabulary");
        }
        if (rows[i].captureLogits && results[i].logits.size() != static_cast<size_t>(session.vocabSize)) {
            throw Error(ErrorCode::RuntimeFault, "Session backend returned an invalid decode batch logits size");
        }
    }
    return results;
}

std::vector<std::vector<float>> SessionState::ForwardDecodeBatchLogits(
    const std::vector<std::shared_ptr<SessionState>>& sessions,
    const std::vector<int>& tokenIds) {
    aila::ScopedLogDeferral deferLogs;
    if (sessions.empty() || sessions.size() > MaxDecodeBatchSize || sessions.size() != tokenIds.size()) {
        throw Error(ErrorCode::InvalidArgument, "Decode batch requires one token per session and one to four rows");
    }
    std::vector<SessionDecodeInput> inputs;
    inputs.reserve(sessions.size());
    for (size_t i = 0; i < sessions.size(); ++i) {
        if (!sessions[i]) throw Error(ErrorCode::InvalidArgument, "Decode batch session must not be null");
        if (std::find_if(inputs.begin(), inputs.end(), [&](const auto& row) {
                return row.session == sessions[i].get();
            }) != inputs.end()) {
            throw Error(ErrorCode::InvalidArgument, "Decode batch contains a duplicate session");
        }
        inputs.push_back({sessions[i].get(), tokenIds[i], nullptr, true});
    }
    size_t acquired = 0;
    try {
        for (const auto& session : sessions) {
            session->Acquire();
            ++acquired;
        }
        auto results = ExecuteDecodeBatch(inputs);
        std::vector<std::vector<float>> logits;
        logits.reserve(results.size());
        for (auto& result : results) logits.push_back(std::move(result.logits));
        for (const auto& session : sessions) session->Release();
        return logits;
    } catch (...) {
        for (size_t i = 0; i < acquired; ++i) sessions[i]->Release();
        throw;
    }
}

} // namespace aila::inference
