#include "inference/Request.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace aila::inference {
namespace {

void ValidateOptions(const std::shared_ptr<SessionState>& session,
                     const std::vector<int>& promptIds,
                     const RequestOptions& options) {
    if (!session) {
        throw Error(ErrorCode::InvalidArgument, "Request session must not be null");
    }
    if (promptIds.empty()) {
        throw Error(ErrorCode::InvalidArgument, "Request prompt must not be empty");
    }
    if (options.prefillChunkSize <= 0 || options.generation.max_new_tokens <= 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "Prefill chunk size and maximum new tokens must be positive");
    }
    for (const int tokenId : promptIds) {
        if (tokenId < 0 || tokenId >= session->VocabSize()) {
            throw Error(ErrorCode::InvalidArgument,
                        "Prompt token ID is outside the model vocabulary");
        }
    }
    for (const int tokenId : options.stopTokenIds) {
        if (tokenId < 0 || tokenId >= session->VocabSize()) {
            throw Error(ErrorCode::InvalidArgument,
                        "Stop token ID is outside the model vocabulary");
        }
    }

    const auto& generation = options.generation;
    if (!std::isfinite(generation.temperature) || generation.temperature < 0.0f ||
        (generation.do_sample && generation.temperature == 0.0f)) {
        throw Error(ErrorCode::InvalidArgument,
                    "Temperature must be finite and nonnegative, and positive when sampling");
    }
    if (!std::isfinite(generation.top_p) || generation.top_p <= 0.0f ||
        generation.top_p > 1.0f || generation.top_k < 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "top_p must be finite in (0, 1] and top_k must be nonnegative");
    }
    if (!std::isfinite(generation.repetition_penalty) ||
        generation.repetition_penalty <= 0.0f ||
        !std::isfinite(generation.presence_penalty) ||
        !std::isfinite(generation.frequency_penalty)) {
        throw Error(ErrorCode::InvalidArgument,
                    "Penalties must be finite and repetition_penalty must be positive");
    }
    if (generation.thinking_budget_tokens != -1) {
        throw Error(ErrorCode::InvalidArgument,
                    "Thinking budgets are not supported by the token request driver; use -1");
    }

    const size_t capacity = static_cast<size_t>(session->Capacity());
    if (promptIds.size() > capacity ||
        static_cast<size_t>(generation.max_new_tokens) > capacity - promptIds.size()) {
        throw Error(ErrorCode::ContextOverflow,
                    "Prompt and maximum new tokens exceed the session context capacity");
    }
}

class StepFlagRelease final {
public:
    explicit StepFlagRelease(std::atomic<bool>& flag) noexcept : flag(flag) {}
    ~StepFlagRelease() { flag.store(false, std::memory_order_release); }

private:
    std::atomic<bool>& flag;
};

} // namespace

bool IsTerminal(RequestStatus status) noexcept {
    return status == RequestStatus::Completed ||
           status == RequestStatus::Cancelled ||
           status == RequestStatus::Failed;
}

std::shared_ptr<RequestState> RequestState::Create(
    std::shared_ptr<SessionState> session,
    std::vector<int> promptIds,
    RequestOptions options) {
    ValidateOptions(session, promptIds, options);
    session->Acquire();
    try {
        const size_t available =
            static_cast<size_t>(session->Capacity() - session->ContextLength());
        if (promptIds.size() > available ||
            static_cast<size_t>(options.generation.max_new_tokens) >
                available - promptIds.size()) {
            throw Error(ErrorCode::ContextOverflow,
                        "Cached context, prompt and maximum new tokens exceed session capacity");
        }
        auto request = std::shared_ptr<RequestState>(
            new RequestState(session, std::move(promptIds), std::move(options)));
        request->ownsSession = true;
        return request;
    } catch (...) {
        session->Release();
        throw;
    }
}

RequestState::RequestState(std::shared_ptr<SessionState> session,
                           std::vector<int> promptIds,
                           RequestOptions options)
    : session(std::move(session)),
      promptIds(std::move(promptIds)),
      options(std::move(options)),
      rng(this->options.generation.use_fixed_seed
              ? static_cast<uint32_t>(this->options.generation.sampling_seed)
              : std::random_device{}()) {
    // Error paths can publish a bounded message without allocating after a
    // backend failure. Request creation rolls the lease back if this fails.
    snapshot.error.reserve(1024);
    snapshot.generatedIds.reserve(static_cast<size_t>(this->options.generation.max_new_tokens));
}

RequestState::~RequestState() {
    if (ownsSession) {
        session->Release();
    }
}

StepResult RequestState::Finish(RequestStatus status, std::string_view error,
                                std::optional<ErrorCode> errorCode) {
    if (IsTerminal(snapshot.status)) {
        return {snapshot.status, std::nullopt, false};
    }
    snapshot.status = status;
    snapshot.errorCode = errorCode;
    size_t errorSize = std::min(error.size(), size_t{1024});
    if (errorSize < error.size()) {
        while (errorSize && (static_cast<unsigned char>(error[errorSize]) & 0xc0) == 0x80) --errorSize;
    }
    snapshot.error.clear();
    if (errorSize) snapshot.error.assign(error.data(), errorSize);
    pendingToken = -1;
    if (ownsSession) {
        ownsSession = false;
        session->Release();
    }
    return {status, std::nullopt, true};
}

StepResult RequestState::Step() {
    // A deferred log callback may release the caller's last request handle.
    // Keep this object alive until the step flag and all other guards unwind.
    auto keepAlive = shared_from_this();
    bool expected = false;
    if (!stepInProgress.compare_exchange_strong(expected, true,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
        throw Error(ErrorCode::Busy, "Request step is already in progress");
    }
    StepFlagRelease releaseFlag(stepInProgress);
    // Flush after releasing the snapshot/session/device locks, but while the
    // step guard still rejects recursive Step calls from a logging callback.
    aila::ScopedLogDeferral deferLogs;
    std::lock_guard<std::mutex> lock(stepMutex);

    if (IsTerminal(snapshot.status)) {
        return {snapshot.status, std::nullopt, false};
    }
    if (cancelRequested.load(std::memory_order_acquire)) {
        return Finish(RequestStatus::Cancelled);
    }

    try {
        ExecutionResult result;
        SamplingInput sampling{options.generation, snapshot.generatedIds, rng};
        if (snapshot.status == RequestStatus::Queued ||
            snapshot.status == RequestStatus::Prefill) {
            snapshot.status = RequestStatus::Prefill;
            const size_t remaining = promptIds.size() - snapshot.promptTokensProcessed;
            const size_t chunkSize =
                std::min(remaining, static_cast<size_t>(options.prefillChunkSize));
            const auto first = promptIds.begin() + snapshot.promptTokensProcessed;
            const bool finalChunk = chunkSize == remaining;
            result = session->Execute(std::vector<int>(first, first + chunkSize),
                                      finalChunk ? &sampling : nullptr);
            snapshot.promptTokensProcessed += chunkSize;
            if (cancelRequested.load(std::memory_order_acquire)) {
                return Finish(RequestStatus::Cancelled);
            }
            if (!finalChunk) {
                return {RequestStatus::Prefill, std::nullopt, false};
            }
        } else if (snapshot.status == RequestStatus::Decode) {
            result = session->Execute({pendingToken}, &sampling);
            pendingToken = -1;
            if (cancelRequested.load(std::memory_order_acquire)) {
                return Finish(RequestStatus::Cancelled);
            }
        } else if (snapshot.status == RequestStatus::Finalizing) {
            // The last emitted token has not yet been consumed. Flush it so a
            // completed session is ready for an appended prompt without replay.
            session->Execute({pendingToken}, nullptr);
            pendingToken = -1;
            return Finish(cancelRequested.load(std::memory_order_acquire)
                              ? RequestStatus::Cancelled
                              : RequestStatus::Completed);
        } else {
            return Finish(RequestStatus::Failed, "Invalid nonterminal request status");
        }

        if (std::find(options.stopTokenIds.begin(), options.stopTokenIds.end(),
                      result.tokenId) != options.stopTokenIds.end()) {
            return Finish(RequestStatus::Completed);
        }

        snapshot.generatedIds.push_back(result.tokenId);
        pendingToken = result.tokenId;
        snapshot.status = snapshot.generatedIds.size() ==
                                  static_cast<size_t>(options.generation.max_new_tokens)
                              ? RequestStatus::Finalizing
                              : RequestStatus::Decode;
        return {snapshot.status, result.tokenId, false};
    } catch (const Error& error) {
        return Finish(RequestStatus::Failed, error.what(), error.Code());
    } catch (const std::exception& error) {
        return Finish(RequestStatus::Failed, error.what());
    } catch (...) {
        return Finish(RequestStatus::Failed, "Unknown session backend failure");
    }
}

RequestSnapshot RequestState::Snapshot() const {
    std::lock_guard<std::mutex> lock(stepMutex);
    return snapshot;
}

RequestProgress RequestState::Progress() const {
    std::lock_guard<std::mutex> lock(stepMutex);
    RequestProgress result;
    result.status = snapshot.status;
    result.promptTokensProcessed = snapshot.promptTokensProcessed;
    result.generatedTokens = snapshot.generatedIds.size();
    result.errorCode = snapshot.errorCode;
    result.errorSize = std::min(snapshot.error.size(), result.errorText.size());
    std::copy_n(snapshot.error.data(), result.errorSize, result.errorText.data());
    return result;
}

} // namespace aila::inference
