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
    if (options.prefillChunkSize <= 0 ||
        (options.prefillOnly ? options.generation.max_new_tokens != 0
                             : options.generation.max_new_tokens <= 0)) {
        throw Error(ErrorCode::InvalidArgument,
                    "Prefill chunk size must be positive; generation requires a positive token limit and replay requires zero");
    }
    if (options.prefillOnly && (!options.stopTokenIds.empty() || options.thinkingBudget ||
                               options.generation.thinking_budget_tokens != -1)) {
        throw Error(ErrorCode::InvalidArgument, "Prefill-only requests cannot have stop tokens or a thinking budget");
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
    if (generation.thinking_budget_tokens < -1 ||
        (generation.thinking_budget_tokens >= 0 && !options.thinkingBudget)) {
        throw Error(ErrorCode::InvalidArgument,
                    "Thinking budgets require explicit marker and closing-token metadata");
    }
    if (options.thinkingBudget) {
        if (const char* error = aila::generation::ValidateThinkingBudget(
                *options.thinkingBudget, static_cast<uint32_t>(session->VocabSize()), options.stopTokenIds)) {
            throw Error(ErrorCode::InvalidArgument, error);
        }
        if (generation.thinking_budget_tokens >= 0 &&
            static_cast<uint32_t>(generation.thinking_budget_tokens) != options.thinkingBudget->maxTokens) {
            throw Error(ErrorCode::InvalidArgument, "Conflicting thinking budgets");
        }
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
      rng(this->options.prefillOnly ? 0u : this->options.generation.use_fixed_seed
              ? static_cast<uint32_t>(this->options.generation.sampling_seed)
              : std::random_device{}()) {
    // Error paths can publish a bounded message without allocating after a
    // backend failure. Request creation rolls the lease back if this fails.
    snapshot.error.reserve(1024);
    snapshot.generatedIds.reserve(static_cast<size_t>(this->options.generation.max_new_tokens));
    if (this->options.thinkingBudget) {
        insideThinking = this->options.thinkingBudget->promptOpensThink;
        this->options.generation.thinking_budget_tokens = static_cast<int>(this->options.thinkingBudget->maxTokens);
    }
}

RequestState::~RequestState() {
    if (ownsSession) {
        session->Release();
    }
}

StepResult RequestState::Finish(RequestStatus status, std::string_view error,
                                std::optional<ErrorCode> errorCode, RequestEndReason endReason) {
    if (IsTerminal(snapshot.status)) {
        return {snapshot.status, std::nullopt, false};
    }
    snapshot.status = status;
    snapshot.endReason = status == RequestStatus::Completed ? endReason
        : status == RequestStatus::Cancelled ? RequestEndReason::Cancelled
        : errorCode == ErrorCode::RuntimeFault ? RequestEndReason::RuntimeFault : RequestEndReason::Failed;
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

    return StepLocked();
}

bool RequestState::PrepareThinkingClose() {
    bool closeExceedsRemaining = false;
    if (options.thinkingBudget && !forcingThinkingClose && insideThinking &&
        (!options.thinkingBudget->closeOnce || !thinkingCloseWasForced) &&
        thinkingTokensUsed >= options.thinkingBudget->maxTokens) {
        const size_t remaining = static_cast<size_t>(options.generation.max_new_tokens) - snapshot.generatedIds.size();
        closeExceedsRemaining = options.thinkingBudget->closeTokenIds.size() > remaining;
        if (!closeExceedsRemaining) {
            forcingThinkingClose = true;
            thinkingCloseWasForced = true;
            thinkingCloseIndex = 0;
        }
    }
    return closeExceedsRemaining;
}

StepResult RequestState::PublishSample(ExecutionResult result, bool closeExceedsRemaining) {
    if (closeExceedsRemaining) {
        return Finish(RequestStatus::Completed, {}, std::nullopt, RequestEndReason::Length);
    }
    const bool forcedToken = forcingThinkingClose;
    if (forcedToken) {
        result.tokenId = options.thinkingBudget->closeTokenIds[thinkingCloseIndex++];
        if (thinkingCloseIndex == options.thinkingBudget->closeTokenIds.size()) {
            forcingThinkingClose = false;
            insideThinking = false;
        }
    }
    if (std::find(options.stopTokenIds.begin(), options.stopTokenIds.end(),
                  result.tokenId) != options.stopTokenIds.end()) {
        return Finish(RequestStatus::Completed, {}, std::nullopt, RequestEndReason::StopToken);
    }

    snapshot.generatedIds.push_back(result.tokenId);
    if (options.thinkingBudget && !forcedToken) {
        const auto& budget = *options.thinkingBudget;
        if (result.tokenId == budget.openTokenId) insideThinking = true;
        else if (result.tokenId == budget.closeTokenId) insideThinking = false;
        else if (insideThinking) ++thinkingTokensUsed;
    }
    pendingToken = result.tokenId;
    snapshot.status = snapshot.generatedIds.size() ==
                              static_cast<size_t>(options.generation.max_new_tokens)
                          ? RequestStatus::Finalizing
                          : RequestStatus::Decode;
    return {snapshot.status, result.tokenId, false};
}

StepResult RequestState::StepLocked() {

    if (IsTerminal(snapshot.status)) {
        return {snapshot.status, std::nullopt, false};
    }
    if (cancelRequested.load(std::memory_order_acquire)) {
        return Finish(RequestStatus::Cancelled);
    }

    try {
        ExecutionResult result;
        SamplingInput sampling{options.generation, snapshot.generatedIds, rng,
                               options.forceHostSampling, options.legacyPrefillLogits};
        const bool closeExceedsRemaining = PrepareThinkingClose();
        const SamplingInput* nextSample = options.prefillOnly || forcingThinkingClose || closeExceedsRemaining ? nullptr : &sampling;
        if (snapshot.status == RequestStatus::Queued ||
            snapshot.status == RequestStatus::Prefill) {
            snapshot.status = RequestStatus::Prefill;
            const size_t remaining = promptIds.size() - snapshot.promptTokensProcessed;
            const size_t chunkSize =
                std::min(remaining, static_cast<size_t>(options.prefillChunkSize));
            const auto first = promptIds.begin() + snapshot.promptTokensProcessed;
            const bool finalChunk = chunkSize == remaining;
            result = session->Execute(std::vector<int>(first, first + chunkSize),
                                      finalChunk ? nextSample : nullptr);
            snapshot.promptTokensProcessed += chunkSize;
            if (cancelRequested.load(std::memory_order_acquire)) {
                return Finish(RequestStatus::Cancelled);
            }
            if (!finalChunk) {
                return {RequestStatus::Prefill, std::nullopt, false};
            }
            if (options.prefillOnly) {
                return Finish(RequestStatus::Completed, {}, std::nullopt, RequestEndReason::Length);
            }
        } else if (snapshot.status == RequestStatus::Decode) {
            result = session->Execute({pendingToken}, nextSample);
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
                              : RequestStatus::Completed, {}, std::nullopt, RequestEndReason::Length);
        } else {
            return Finish(RequestStatus::Failed, "Invalid nonterminal request status");
        }

        return PublishSample(std::move(result), closeExceedsRemaining);
    } catch (const Error& error) {
        return Finish(RequestStatus::Failed, error.what(), error.Code());
    } catch (const std::exception& error) {
        return Finish(RequestStatus::Failed, error.what());
    } catch (...) {
        return Finish(RequestStatus::Failed, "Unknown session backend failure");
    }
}

std::vector<StepResult> RequestState::StepDecodeBatch(
    const std::vector<std::shared_ptr<RequestState>>& requests) {
    constexpr size_t MaxBatchSize = MaxDecodeBatchSize;
    if (requests.empty() || requests.size() > MaxBatchSize) {
        throw Error(ErrorCode::InvalidArgument, "Decode batch must contain between one and four requests");
    }
    for (size_t i = 0; i < requests.size(); ++i) {
        if (!requests[i]) throw Error(ErrorCode::InvalidArgument, "Decode batch request must not be null");
        for (size_t j = 0; j < i; ++j) {
            if (requests[i] == requests[j]) {
                throw Error(ErrorCode::InvalidArgument, "Decode batch contains a duplicate request");
            }
        }
    }
    if (requests.size() == 1) return {requests.front()->Step()};
    const void* batchKey = requests.front()->DecodeBatchKey();
    if (!batchKey || std::any_of(requests.begin(), requests.end(), [&](const auto& request) {
            return request->DecodeBatchKey() != batchKey;
        })) {
        throw Error(ErrorCode::InvalidArgument, "Decode batch requests must share a batch-capable model");
    }

    // Retain every row through deferred callbacks, even if the caller releases
    // its vector/handles from a callback. Claim all step guards before touching
    // request state; overlapping/reentrant batches fail Busy without a partial
    // advance. Locks unwind before logs, and flags unwind after those callbacks.
    const auto retained = requests;
    std::vector<StepResult> steps(retained.size());
    struct ReleaseFlags {
        std::array<std::atomic<bool>*, MaxBatchSize> flags{};
        size_t count = 0;
        ~ReleaseFlags() {
            for (size_t i = 0; i < count; ++i) flags[i]->store(false, std::memory_order_release);
        }
    } releaseFlags;
    for (const auto& request : retained) {
        bool expected = false;
        if (!request->stepInProgress.compare_exchange_strong(expected, true,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            throw Error(ErrorCode::Busy, "Decode batch request step is already in progress");
        }
        releaseFlags.flags[releaseFlags.count++] = &request->stepInProgress;
    }
    aila::ScopedLogDeferral deferLogs;
    std::array<std::unique_lock<std::mutex>, MaxBatchSize> locks;
    for (size_t i = 0; i < retained.size(); ++i) {
        locks[i] = std::unique_lock<std::mutex>(retained[i]->stepMutex);
    }
    // Validate the whole call before terminalizing cancellations or changing
    // thinking state, so an invalid direct caller has no partial side effects.
    for (const auto& request : retained) {
        if (!IsTerminal(request->snapshot.status) &&
            !request->cancelRequested.load(std::memory_order_acquire) &&
            request->snapshot.status != RequestStatus::Decode &&
            request->snapshot.status != RequestStatus::Finalizing) {
            throw Error(ErrorCode::InvalidArgument, "Decode batch contains a request that still requires prefill");
        }
    }
    for (size_t i = 0; i < retained.size(); ++i) {
        steps[i] = {retained[i]->snapshot.status, std::nullopt, false};
    }

    std::array<std::optional<SamplingInput>, MaxBatchSize> sampling;
    std::array<size_t, MaxBatchSize> rowIndices{};
    std::array<bool, MaxBatchSize> closeExceedsRemaining{};
    try {
        std::vector<SessionDecodeInput> rows;
        rows.reserve(retained.size());
        for (size_t i = 0; i < retained.size(); ++i) {
            auto& request = *retained[i];
            if (IsTerminal(request.snapshot.status)) {
                steps[i] = {request.snapshot.status, std::nullopt, false};
                continue;
            }
            if (request.cancelRequested.load(std::memory_order_acquire)) {
                steps[i] = request.Finish(RequestStatus::Cancelled);
                continue;
            }
            closeExceedsRemaining[i] = request.PrepareThinkingClose();
            sampling[i].emplace(SamplingInput{request.options.generation, request.snapshot.generatedIds,
                request.rng, request.options.forceHostSampling, request.options.legacyPrefillLogits});
            const SamplingInput* nextSample =
                request.snapshot.status == RequestStatus::Finalizing || request.forcingThinkingClose ||
                closeExceedsRemaining[i] ? nullptr : &*sampling[i];
            rowIndices[rows.size()] = i;
            rows.push_back({request.session.get(), request.pendingToken, nextSample, false});
        }
        if (rows.empty()) return steps;
        auto results = SessionState::ExecuteDecodeBatch(rows);
        if (results.size() != rows.size()) {
            throw Error(ErrorCode::RuntimeFault, "Decode batch backend returned an invalid result count");
        }
        for (size_t row = 0; row < rows.size(); ++row) {
            const size_t i = rowIndices[row];
            auto& request = *retained[i];
            request.pendingToken = -1;
            if (request.cancelRequested.load(std::memory_order_acquire)) {
                steps[i] = request.Finish(RequestStatus::Cancelled);
            } else if (request.snapshot.status == RequestStatus::Finalizing) {
                steps[i] = request.Finish(RequestStatus::Completed, {}, std::nullopt, RequestEndReason::Length);
            } else {
                steps[i] = request.PublishSample(std::move(results[row]), closeExceedsRemaining[i]);
            }
        }
    } catch (const Error& error) {
        for (size_t i = 0; i < retained.size(); ++i) {
            if (!IsTerminal(retained[i]->snapshot.status)) {
                steps[i] = retained[i]->Finish(RequestStatus::Failed, error.what(), error.Code());
            }
        }
    } catch (const std::exception& error) {
        for (size_t i = 0; i < retained.size(); ++i) {
            if (!IsTerminal(retained[i]->snapshot.status)) {
                steps[i] = retained[i]->Finish(RequestStatus::Failed, error.what());
            }
        }
    } catch (...) {
        for (size_t i = 0; i < retained.size(); ++i) {
            if (!IsTerminal(retained[i]->snapshot.status)) {
                steps[i] = retained[i]->Finish(RequestStatus::Failed, "Unknown decode batch failure");
            }
        }
    }
    return steps;
}

RequestSnapshot RequestState::Snapshot() const {
    std::lock_guard<std::mutex> lock(stepMutex);
    return snapshot;
}

RequestProgress RequestState::Progress() const {
    std::lock_guard<std::mutex> lock(stepMutex);
    RequestProgress result;
    result.status = snapshot.status;
    result.endReason = snapshot.endReason;
    result.promptTokensProcessed = snapshot.promptTokensProcessed;
    result.generatedTokens = snapshot.generatedIds.size();
    result.errorCode = snapshot.errorCode;
    result.errorSize = std::min(snapshot.error.size(), result.errorText.size());
    std::copy_n(snapshot.error.data(), result.errorSize, result.errorText.data());
    return result;
}

} // namespace aila::inference
