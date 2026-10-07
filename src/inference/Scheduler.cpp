#include "inference/Scheduler.hpp"
#include "inference/RuntimeFault.hpp"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <limits>
#include <list>
#include <string_view>
#include <thread>
#include <utility>

namespace aila::inference {
namespace {

constexpr size_t MaxErrorBytes = detail::RuntimeFaultRecord::MaxMessageBytes;

std::string_view BoundedError(std::string_view error) noexcept {
    size_t count = std::min(error.size(), MaxErrorBytes);
    // Do not truncate in the middle of a UTF-8 sequence.
    if (count < error.size()) {
        while (count && (static_cast<unsigned char>(error[count]) & 0xc0) == 0x80) --count;
    }
    return error.substr(0, count);
}

void ValidateOptions(const SchedulerOptions& options) {
    if (options.maxActiveRequests == 0 || options.maxOutstandingRequests == 0 ||
        options.maxActiveRequests > options.maxOutstandingRequests ||
        options.maxDecodeBatchSize == 0 || options.maxDecodeBatchSize > MaxDecodeBatchSize ||
        options.maxDecodeBatchSize > options.maxActiveRequests ||
        options.maxPromptTokens == 0 || options.maxGeneratedTokens == 0 ||
        options.maxStopTokenIds == 0 ||
        options.maxOutstandingTokens == 0 || options.maxPrefillChunkSize <= 0 ||
        options.maxBufferedTokensPerRequest == 0) {
        throw Error(ErrorCode::InvalidArgument,
                    "Scheduler limits must be positive, active <= outstanding, and decode batch <= active and 4");
    }
}

} // namespace

namespace detail {

struct AdmissionCredit {
    std::weak_ptr<SchedulerState> scheduler;
    size_t tokens = 0;
    size_t cacheBytes = 0;
    bool held = false;
    ~AdmissionCredit();
};

struct BufferedToken {
    int tokenId = -1;
    RequestStatus status = RequestStatus::Decode;
    size_t promptTokensProcessed = 0;
};

struct ScheduledRequestState {
    explicit ScheduledRequestState(size_t bufferCapacity) : buffer(bufferCapacity) {
        snapshot.error.reserve(MaxErrorBytes);
    }

    void Cancel(RequestEndReason reason) noexcept;
    void Detach() noexcept;

    mutable std::mutex mutex;
    std::condition_variable changed;
    std::weak_ptr<SchedulerState> scheduler;
    std::unique_ptr<AdmissionCredit> credit;
    std::shared_ptr<RequestState> request;
    const void* decodeBatchKey = nullptr; // Immutable, captured outside scheduler/stream locks.
    bool activeSlot = false; // Protected by the scheduler queue mutex.
    std::vector<BufferedToken> buffer;
    size_t bufferHead = 0;
    size_t buffered = 0;
    uint64_t deliveredTokens = 0;
    size_t maxNewTokens = 0;
    ScheduledRequestSnapshot snapshot;
    std::atomic<RequestEndReason> cancellation{RequestEndReason::None};
    bool terminalReady = false;
    bool detached = false;
};

struct SchedulerState : std::enable_shared_from_this<SchedulerState> {
    explicit SchedulerState(SchedulerOptions value) : options(std::move(value)) {}

    ~SchedulerState() {
        // A logging callback may destroy the Scheduler on its own worker.
        // The worker's strong reference then keeps all state alive until Run
        // has finished. There is nothing left to execute when this detaches.
        if (worker.joinable()) worker.detach();
    }

    void Start() {
        auto self = shared_from_this();
        worker = std::thread([self] { self->Run(); });
        workerId = worker.get_id();
    }

    void CheckAccepting() const {
        fault.CheckUsable();
        if (stopping) throw Error(ErrorCode::SchedulerStopped, "Scheduler has been shut down");
    }

    void Shutdown() noexcept {
        std::array<std::shared_ptr<ScheduledRequestState>, MaxDecodeBatchSize> runningRequests;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            runningRequests = running;
        }
        for (const auto& entry : runningRequests) {
            if (entry) entry->Cancel(RequestEndReason::SchedulerStopped);
        }
        changed.notify_all();
        // Check before acquiring joinMutex: an external join may already be
        // waiting for this callback to return.
        if (std::this_thread::get_id() == workerId) return;
        std::lock_guard<std::mutex> lock(joinMutex);
        if (worker.joinable()) worker.join();
    }

    void Fault(std::string_view message) noexcept {
        std::array<std::shared_ptr<ScheduledRequestState>, MaxDecodeBatchSize> runningRequests;
        {
            std::lock_guard<std::mutex> lock(mutex);
            fault.Record(message);
            runningRequests = running;
        }
        for (const auto& entry : runningRequests) {
            if (entry) entry->Cancel(RequestEndReason::RuntimeFault);
        }
        changed.notify_all();
    }

    // No external code runs while either the queue or stream mutex is held.
    // Terminal storage is independent of the token ring and pre-reserved.
    void Complete(const std::shared_ptr<ScheduledRequestState>& entry,
                  RequestStatus status, RequestEndReason reason,
                  std::optional<ErrorCode> errorCode, std::string_view error,
                  size_t promptTokensProcessed) noexcept {
        std::shared_ptr<RequestState> finishedRequest;
        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            finishedRequest = std::move(entry->request);
        }
        // Request::Step has already drained any submitted device work. Drop
        // model/session ownership before exposing the terminal to readers.
        finishedRequest.reset();

        std::unique_ptr<AdmissionCredit> releasedCredit;
        std::vector<BufferedToken> releasedBuffer;
        {
            // Linearize completion with lifecycle faults. A terminal published
            // before Fault remains final; a still-unpublished terminal fails
            // with the domain's first diagnosis, including a concurrent fault.
            std::lock_guard<std::mutex> queueLock(mutex);
            std::lock_guard<std::mutex> lock(entry->mutex);
            if (entry->terminalReady) return;
            // A reader can return the admission credit as soon as this terminal
            // is visible. Retire the active slot in the same queue transaction
            // so Snapshot never counts an active request without its credit,
            // even while another completed batch row is still being destroyed.
            if (entry->activeSlot) {
                entry->activeSlot = false;
                --activeRequests;
            }
            if (fault.Faulted()) {
                status = RequestStatus::Failed;
                reason = RequestEndReason::RuntimeFault;
                errorCode = ErrorCode::RuntimeFault;
                error = fault.Message();
            }
            auto& snapshot = entry->snapshot;
            snapshot.status = status;
            snapshot.endReason = reason;
            snapshot.promptTokensProcessed = promptTokensProcessed;
            snapshot.errorCode = errorCode;
            const auto bounded = BoundedError(error);
            snapshot.error.clear();
            if (!bounded.empty()) snapshot.error.assign(bounded.data(), bounded.size());
            entry->terminalReady = true;
            if (entry->detached) {
                releasedCredit = std::move(entry->credit);
                releasedBuffer.swap(entry->buffer);
                entry->buffered = 0;
                snapshot.bufferedTokens = 0;
            }
        }
        std::vector<BufferedToken>().swap(releasedBuffer);
        releasedCredit.reset();
        entry->changed.notify_all();
    }

    std::shared_ptr<RequestState> PrepareOne(
        const std::shared_ptr<ScheduledRequestState>& entry, size_t& processed) {
        bool domainFaulted = false;
        bool shutdown = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            domainFaulted = fault.Faulted();
            shutdown = stopping;
        }
        if (domainFaulted) entry->Cancel(RequestEndReason::RuntimeFault);
        else if (shutdown) entry->Cancel(RequestEndReason::SchedulerStopped);

        std::shared_ptr<RequestState> request;
        bool full = false;
        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            request = entry->request;
            processed = entry->snapshot.promptTokensProcessed;
            // Finalizing cannot emit another token; a full ring must not
            // turn successful finalization into slow-consumer cancellation.
            full = !entry->detached && entry->buffered == entry->buffer.size() &&
                   entry->snapshot.status != RequestStatus::Finalizing;
        }
        if (full) entry->Cancel(RequestEndReason::OutputBackpressure);
        // Pending-cancel maintenance must never execute a token-producing step.
        if (entry->cancellation.load(std::memory_order_acquire) != RequestEndReason::None) {
            request->Cancel();
        }
        return request;
    }

    bool PublishOne(const std::shared_ptr<ScheduledRequestState>& entry,
                    std::shared_ptr<RequestState>& request,
                    const StepResult& step, const RequestProgress& progress) {
        const size_t processed = progress.promptTokensProcessed;
        if (IsTerminal(step.status)) {
            RequestStatus status = step.status;
            RequestEndReason reason = RequestEndReason::Failed;
            auto errorCode = progress.errorCode;
            std::string_view error = progress.ErrorMessage();
            if (errorCode == ErrorCode::RuntimeFault) {
                Fault(error);
                status = RequestStatus::Failed;
                reason = RequestEndReason::RuntimeFault;
                errorCode = ErrorCode::RuntimeFault;
            } else if (status == RequestStatus::Completed) {
                reason = progress.endReason;
            } else if (status == RequestStatus::Cancelled) {
                reason = entry->cancellation.load(std::memory_order_acquire);
                if (reason == RequestEndReason::None) reason = RequestEndReason::Cancelled;
                if (reason == RequestEndReason::OutputBackpressure) {
                    errorCode = ErrorCode::ResourceLimit;
                    error = "Request output buffer is full";
                } else if (reason == RequestEndReason::SchedulerStopped) {
                    errorCode = ErrorCode::SchedulerStopped;
                    error = "Scheduler has been shut down";
                }
            }
            request.reset();
            Complete(entry, status, reason, errorCode, error, processed);
            return true;
        }

        bool domainFaulted = false;
        {
            // No sample may be published after ReportRuntimeFault returns.
            // Both locks protect host-only, preallocated state; device
            // work, destruction and callbacks stay outside these locks.
            std::lock_guard<std::mutex> queueLock(mutex);
            domainFaulted = fault.Faulted();
            if (!domainFaulted) {
                std::lock_guard<std::mutex> lock(entry->mutex);
                auto& snapshot = entry->snapshot;
                snapshot.status = step.status;
                snapshot.promptTokensProcessed = processed;
                if (step.tokenId && !entry->detached) {
                    const size_t tail = (entry->bufferHead + entry->buffered) % entry->buffer.size();
                    entry->buffer[tail] = {*step.tokenId, step.status, processed};
                    ++entry->buffered;
                    ++snapshot.generatedTokens;
                    snapshot.bufferedTokens = entry->buffered;
                }
            }
        }
        if (domainFaulted) {
            // A report can arrive after Step prepared a sample, including
            // from a deferred logging callback. Retire it without publication.
            request->Cancel();
            request->Step();
            request.reset();
            Complete(entry, RequestStatus::Failed, RequestEndReason::RuntimeFault,
                     ErrorCode::RuntimeFault, {}, processed);
            return true;
        }
        entry->changed.notify_all();
        return false;
    }

    void FailOne(const std::shared_ptr<ScheduledRequestState>& entry,
                 std::optional<ErrorCode> code, std::string_view error) noexcept {
        std::shared_ptr<RequestState> request;
        size_t processed = 0;
        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            request = entry->request;
            processed = entry->snapshot.promptTokensProcessed;
        }
        // Failures outside Step must also retire the session at a safe boundary.
        // Request::Cancel is observed before any further Forward is allowed.
        if (request) {
            request->Cancel();
            try {
                request->Step();
                processed = request->Progress().promptTokensProcessed;
            } catch (...) {}
            request.reset();
        }
        Complete(entry, RequestStatus::Failed, RequestEndReason::Failed, code, error, processed);
    }

    bool RunOne(const std::shared_ptr<ScheduledRequestState>& entry) noexcept {
        try {
            size_t processed = 0;
            auto request = PrepareOne(entry, processed);
            size_t outputSlots = 0;
            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                outputSlots = entry->buffer.size() - entry->buffered;
            }
            const auto step = request->Step(outputSlots);
            return PublishOne(entry, request, step, request->Progress());
        } catch (const Error& error) {
            if (error.Code() == ErrorCode::RuntimeFault) Fault(error.what());
            FailOne(entry, error.Code(), error.what());
        } catch (const std::exception& error) {
            FailOne(entry, std::nullopt, error.what());
        } catch (...) {
            FailOne(entry, std::nullopt, "Unknown scheduler request failure");
        }
        return true;
    }

    // One bit per input node marks terminal completion. A batch-wide execution
    // error fails this group; a RuntimeFault is recorded before publishing any
    // row, including a successful row returned alongside another row's failure.
    unsigned RunBatch(const std::list<std::shared_ptr<ScheduledRequestState>>& entries) noexcept {
        const unsigned allCompleted = (1u << entries.size()) - 1u;
        std::optional<ErrorCode> failureCode;
        std::array<char, MaxErrorBytes> failure{};
        size_t failureSize = 0;
        auto retainFailure = [&](std::optional<ErrorCode> code, std::string_view message) {
            failureCode = code;
            const auto bounded = BoundedError(message);
            failureSize = bounded.size();
            std::copy_n(bounded.data(), failureSize, failure.data());
        };
        try {
            std::vector<std::shared_ptr<RequestState>> requests;
            requests.reserve(entries.size());
            for (const auto& entry : entries) {
                size_t processed = 0;
                requests.push_back(PrepareOne(entry, processed));
            }
            const auto steps = RequestState::StepDecodeBatch(requests);
            std::array<RequestProgress, MaxDecodeBatchSize> progress;
            for (size_t i = 0; i < requests.size(); ++i) {
                progress[i] = requests[i]->Progress();
                if (progress[i].errorCode == ErrorCode::RuntimeFault) Fault(progress[i].ErrorMessage());
            }
            unsigned completed = 0;
            size_t index = 0;
            for (const auto& entry : entries) {
                if (PublishOne(entry, requests[index], steps[index], progress[index])) completed |= 1u << index;
                ++index;
            }
            return completed;
        } catch (const Error& error) {
            if (error.Code() == ErrorCode::RuntimeFault) Fault(error.what());
            retainFailure(error.Code(), error.what());
        } catch (const std::exception& error) {
            retainFailure(std::nullopt, error.what());
        } catch (...) {
            retainFailure(std::nullopt, "Unknown scheduler decode batch failure");
        }
        for (const auto& entry : entries) {
            FailOne(entry, failureCode, {failure.data(), failureSize});
        }
        return allCompleted;
    }

    const void* EligibleBatchKey(const std::shared_ptr<ScheduledRequestState>& entry) const noexcept {
        // Called with the scheduler mutex held. Read only the preallocated
        // stream snapshot and immutable model identity, never Progress/Step.
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->detached || entry->cancellation.load(std::memory_order_acquire) != RequestEndReason::None ||
            (entry->snapshot.status != RequestStatus::Decode && entry->snapshot.status != RequestStatus::Finalizing)) {
            return nullptr;
        }
        return entry->decodeBatchKey;
    }

    void Run() noexcept {
        // Splicing list nodes rotates requests without allocating in the hot
        // queue path. Running nodes remain retained until locks are released.
        std::list<std::shared_ptr<ScheduledRequestState>> current;
        bool lastWasQueuedCancellation = false;
        for (;;) {
            bool wasActive = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || fault.Faulted() || !waiting.empty() || !runnable.empty(); });
                if (waiting.empty() && runnable.empty() && (stopping || fault.Faulted())) break;

                while (activeRequests < options.maxActiveRequests && !waiting.empty()) {
                    waiting.front()->activeSlot = true;
                    runnable.splice(runnable.end(), waiting, waiting.begin());
                    ++activeRequests;
                }

                // Pending cancellation does not need an active slot or GPU
                // work. Alternate this maintenance with runnable work so a
                // stream of new cancelled submissions cannot starve decode.
                auto cancelled = waiting.end();
                if (!lastWasQueuedCancellation || runnable.empty()) {
                    cancelled = std::find_if(waiting.begin(), waiting.end(), [&](const auto& entry) {
                        return stopping || fault.Faulted() ||
                               entry->cancellation.load(std::memory_order_acquire) != RequestEndReason::None;
                    });
                }
                if (cancelled != waiting.end()) {
                    current.splice(current.end(), waiting, cancelled);
                    lastWasQueuedCancellation = true;
                } else if (!runnable.empty()) {
                    current.splice(current.end(), runnable, runnable.begin());
                    wasActive = true;
                    lastWasQueuedCancellation = false;
                    // The oldest runnable request anchors this quantum. Take
                    // only ready peers of its model, then rotate every selected
                    // row to the tail. An unselected prefill/other model becomes
                    // the next anchor; decode batches cannot keep overtaking it.
                    const void* batchKey = options.maxDecodeBatchSize > 1
                        ? EligibleBatchKey(current.front()) : nullptr;
                    if (batchKey && !stopping && !fault.Faulted()) {
                        for (auto it = runnable.begin(); it != runnable.end() &&
                             current.size() < options.maxDecodeBatchSize;) {
                            auto candidate = it++;
                            if (EligibleBatchKey(*candidate) == batchKey) {
                                current.splice(current.end(), runnable, candidate);
                            }
                        }
                    }
                } else {
                    continue;
                }
                size_t index = 0;
                for (const auto& entry : current) running[index++] = entry;
            }

            const unsigned completed = current.size() == 1
                ? (RunOne(current.front()) ? 1u : 0u) : RunBatch(current);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (wasActive) {
                    // Admit newly eligible requests before appending every
                    // surviving batch row. Completed unread streams retain
                    // admission credits, but no longer occupy an active slot.
                    while (activeRequests < options.maxActiveRequests && !waiting.empty()) {
                        waiting.front()->activeSlot = true;
                        runnable.splice(runnable.end(), waiting, waiting.begin());
                        ++activeRequests;
                    }
                    size_t index = 0;
                    for (auto it = current.begin(); it != current.end(); ++index) {
                        auto selected = it++;
                        if (!(completed & (1u << index))) {
                            runnable.splice(runnable.end(), current, selected);
                        }
                    }
                } else if (!completed) {
                    // A maintenance node never owns an active slot. Preserve
                    // accounting even if a future maintenance action yields.
                    waiting.splice(waiting.end(), current, current.begin());
                }
                // current/runnable still own all nodes until after unlock.
                for (auto& entry : running) entry.reset();
            }
            // Destructors can drain device work and invoke deferred log
            // callbacks. Neither scheduler nor stream locks may be held.
            current.clear();
        }
    }

    const SchedulerOptions options;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::list<std::shared_ptr<ScheduledRequestState>> waiting;
    std::list<std::shared_ptr<ScheduledRequestState>> runnable;
    std::array<std::shared_ptr<ScheduledRequestState>, MaxDecodeBatchSize> running;
    size_t activeRequests = 0; // Includes all currently executing batch rows.
    size_t outstandingRequests = 0;
    size_t reservedTokens = 0;
    size_t reservedCacheBytes = 0;
    uint64_t nextRequestId = 1;
    bool stopping = false;
    RuntimeFaultRecord fault;
    std::mutex joinMutex;
    std::thread worker;
    std::thread::id workerId;
};

AdmissionCredit::~AdmissionCredit() {
    if (!held) return;
    if (auto owner = scheduler.lock()) {
        std::lock_guard<std::mutex> lock(owner->mutex);
        --owner->outstandingRequests;
        owner->reservedTokens -= tokens;
        owner->reservedCacheBytes -= cacheBytes;
    }
}

void ScheduledRequestState::Cancel(RequestEndReason reason) noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!terminalReady) {
            // Request::Cancel is only an atomic store (no callbacks/locks).
            // Apply it before publishing the marker used by the work queue.
            if (request) request->Cancel();
            auto expected = RequestEndReason::None;
            cancellation.compare_exchange_strong(expected, reason, std::memory_order_release);
        }
    }
    if (auto owner = scheduler.lock()) owner->changed.notify_all();
}

void ScheduledRequestState::Detach() noexcept {
    std::unique_ptr<AdmissionCredit> releasedCredit;
    std::vector<BufferedToken> releasedBuffer;
    {
        std::lock_guard<std::mutex> lock(mutex);
        detached = true;
        buffered = 0;
        snapshot.bufferedTokens = 0;
        releasedBuffer.swap(buffer);
        if (terminalReady) releasedCredit = std::move(credit);
        // Complete may observe detached as soon as we unlock and return the
        // credit. Free the old ring before making that observation possible.
        std::vector<BufferedToken>().swap(releasedBuffer);
    }
    Cancel(RequestEndReason::Cancelled);
    releasedCredit.reset();
    changed.notify_all();
}

} // namespace detail

ScheduledRequest::ScheduledRequest(std::shared_ptr<detail::ScheduledRequestState> value)
    : state(std::move(value)) {}

ScheduledRequest::~ScheduledRequest() { state->Detach(); }

uint64_t ScheduledRequest::Id() const noexcept { return state->snapshot.requestId; }

void ScheduledRequest::Cancel() noexcept { state->Cancel(RequestEndReason::Cancelled); }

ScheduledRequestSnapshot ScheduledRequest::Snapshot() const {
    const auto retained = state;
    std::lock_guard<std::mutex> lock(retained->mutex);
    return retained->snapshot;
}

RequestReadResult ScheduledRequest::Read(std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) throw Error(ErrorCode::InvalidArgument, "Read timeout must not be negative");
    const auto retained = state;
    std::unique_ptr<detail::AdmissionCredit> releasedCredit;
    std::vector<detail::BufferedToken> releasedBuffer;
    RequestEvent event;
    {
        std::unique_lock<std::mutex> lock(retained->mutex);
        if (!retained->changed.wait_for(lock, timeout, [&] {
                return retained->buffered != 0 || retained->terminalReady || retained->detached;
            })) return {ReadStatus::Timeout, std::nullopt};
        if (retained->detached || retained->snapshot.terminalConsumed) {
            return {ReadStatus::Closed, std::nullopt};
        }
        event.requestId = retained->snapshot.requestId;
        event.sequence = retained->deliveredTokens + 1;
        if (retained->buffered) {
            const auto& token = retained->buffer[retained->bufferHead];
            event.kind = RequestEventKind::Token;
            event.tokenId = token.tokenId;
            event.status = token.status;
            event.promptTokensProcessed = token.promptTokensProcessed;
            event.generatedTokens = static_cast<size_t>(++retained->deliveredTokens);
            retained->bufferHead = (retained->bufferHead + 1) % retained->buffer.size();
            --retained->buffered;
            retained->snapshot.bufferedTokens = retained->buffered;
        } else {
            event.kind = RequestEventKind::Terminal;
            event.status = retained->snapshot.status;
            event.endReason = retained->snapshot.endReason;
            event.promptTokensProcessed = retained->snapshot.promptTokensProcessed;
            event.generatedTokens = retained->snapshot.generatedTokens;
            event.errorCode = retained->snapshot.errorCode;
            event.error = retained->snapshot.error;
            retained->snapshot.terminalConsumed = true;
            releasedCredit = std::move(retained->credit);
            releasedBuffer.swap(retained->buffer);
        }
    }
    std::vector<detail::BufferedToken>().swap(releasedBuffer);
    releasedCredit.reset();
    return {ReadStatus::Event, std::move(event)};
}

Scheduler::Scheduler(SchedulerOptions options) {
    ValidateOptions(options);
    state = std::make_shared<detail::SchedulerState>(std::move(options));
    state->Start();
}

Scheduler::~Scheduler() { Shutdown(); }

void Scheduler::Shutdown() noexcept {
    const auto retained = state;
    if (retained) retained->Shutdown();
}

void Scheduler::ReportRuntimeFault(std::string_view message) noexcept {
    const auto retained = state;
    if (retained) retained->Fault(message);
}

SchedulerSnapshot Scheduler::Snapshot() const {
    const auto retained = state;
    std::lock_guard<std::mutex> lock(retained->mutex);
    return {!retained->stopping && !retained->fault.Faulted(), retained->stopping, retained->fault.Faulted(),
            retained->outstandingRequests, retained->waiting.size(), retained->activeRequests,
            retained->reservedTokens, retained->reservedCacheBytes,
            retained->fault.Faulted() ? retained->fault.Message() : ""};
}

std::shared_ptr<ScheduledRequest> Scheduler::Submit(
    std::shared_ptr<SessionState> session, std::vector<int> promptIds, RequestOptions options) {
    const auto retained = state;
    const auto& limits = retained->options;
    if (!session || promptIds.empty() ||
        (options.prefillOnly ? options.generation.max_new_tokens != 0 : options.generation.max_new_tokens <= 0) ||
        options.prefillChunkSize <= 0) {
        throw Error(ErrorCode::InvalidArgument, "A session, prompt, and positive request limits are required");
    }
    const size_t maxNewTokens = static_cast<size_t>(options.generation.max_new_tokens);
    if (promptIds.size() > limits.maxPromptTokens || maxNewTokens > limits.maxGeneratedTokens ||
        options.stopTokenIds.size() > limits.maxStopTokenIds ||
        maxNewTokens > std::numeric_limits<size_t>::max() - promptIds.size()) {
        throw Error(ErrorCode::ResourceLimit, "Request exceeds scheduler token limits");
    }
    const size_t tokens = promptIds.size() + maxNewTokens;
    const size_t cacheBytes = session->CacheBytes();
    options.prefillChunkSize = std::min(options.prefillChunkSize, limits.maxPrefillChunkSize);

    auto credit = std::make_unique<detail::AdmissionCredit>();
    credit->scheduler = retained;
    credit->tokens = tokens;
    credit->cacheBytes = cacheBytes;
    uint64_t requestId = 0;
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->CheckAccepting();
        if (retained->outstandingRequests >= limits.maxOutstandingRequests ||
            tokens > limits.maxOutstandingTokens - retained->reservedTokens ||
            (limits.maxSessionCacheBytes != 0 &&
             cacheBytes > limits.maxSessionCacheBytes - retained->reservedCacheBytes) ||
            cacheBytes > std::numeric_limits<size_t>::max() - retained->reservedCacheBytes ||
            retained->nextRequestId == std::numeric_limits<uint64_t>::max()) {
            throw Error(ErrorCode::ResourceLimit, "Scheduler admission limit reached");
        }
        ++retained->outstandingRequests;
        retained->reservedTokens += tokens;
        retained->reservedCacheBytes += cacheBytes;
        credit->held = true;
        requestId = retained->nextRequestId++;
    }
    // Reserve before allocating the ring or acquiring the session. Concurrent
    // rejected Submit calls cannot allocate full per-request buffers first.
    auto entry = std::make_shared<detail::ScheduledRequestState>(limits.maxBufferedTokensPerRequest);
    entry->scheduler = retained;
    entry->maxNewTokens = maxNewTokens;
    entry->snapshot.requestId = requestId;
    entry->credit = std::move(credit);
    // Retain only the validated input size, not a caller's arbitrarily large
    // vector capacity. The caller owns any input memory before Submit.
    std::vector<int>(promptIds.begin(), promptIds.end()).swap(promptIds);
    std::vector<int>(options.stopTokenIds.begin(), options.stopTokenIds.end()).swap(options.stopTokenIds);
    // Session acquisition and validation can wait on the executing step. Do
    // them outside the scheduler mutex, then recheck shutdown before enqueue.
    try {
        entry->request = RequestState::Create(std::move(session), std::move(promptIds), std::move(options));
        entry->decodeBatchKey = entry->request->DecodeBatchKey();
    } catch (const Error& error) {
        if (error.Code() == ErrorCode::RuntimeFault) retained->Fault(error.what());
        throw;
    }
    auto handle = std::shared_ptr<ScheduledRequest>(new ScheduledRequest(entry));
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->CheckAccepting();
        retained->waiting.push_back(entry);
    }
    retained->changed.notify_one();
    return handle;
}

} // namespace aila::inference
