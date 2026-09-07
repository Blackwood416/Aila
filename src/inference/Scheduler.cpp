#include "inference/Scheduler.hpp"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <list>
#include <string_view>
#include <thread>
#include <utility>

namespace aila::inference {
namespace {

constexpr size_t MaxErrorBytes = 1024;

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
        options.maxPromptTokens == 0 || options.maxGeneratedTokens == 0 ||
        options.maxStopTokenIds == 0 ||
        options.maxOutstandingTokens == 0 || options.maxPrefillChunkSize <= 0 ||
        options.maxBufferedTokensPerRequest == 0) {
        throw Error(ErrorCode::InvalidArgument, "Scheduler limits must be positive and active <= outstanding");
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
        if (faulted) throw Error(ErrorCode::RuntimeFault, "Scheduler execution domain has failed");
        if (stopping) throw Error(ErrorCode::SchedulerStopped, "Scheduler has been shut down");
    }

    void Shutdown() noexcept {
        std::shared_ptr<ScheduledRequestState> runningRequest;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            runningRequest = running;
        }
        if (runningRequest) runningRequest->Cancel(RequestEndReason::SchedulerStopped);
        changed.notify_all();
        // Check before acquiring joinMutex: an external join may already be
        // waiting for this callback to return.
        if (std::this_thread::get_id() == workerId) return;
        std::lock_guard<std::mutex> lock(joinMutex);
        if (worker.joinable()) worker.join();
    }

    void Fault() noexcept {
        std::shared_ptr<ScheduledRequestState> runningRequest;
        {
            std::lock_guard<std::mutex> lock(mutex);
            faulted = true;
            runningRequest = running;
        }
        if (runningRequest) runningRequest->Cancel(RequestEndReason::RuntimeFault);
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
            std::lock_guard<std::mutex> lock(entry->mutex);
            if (entry->terminalReady) return;
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

    bool RunOne(const std::shared_ptr<ScheduledRequestState>& entry) noexcept {
        std::shared_ptr<RequestState> request;
        size_t processed = 0;
        try {
            bool domainFaulted = false;
            bool shutdown = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                domainFaulted = faulted;
                shutdown = stopping;
            }
            if (domainFaulted) entry->Cancel(RequestEndReason::RuntimeFault);
            else if (shutdown) entry->Cancel(RequestEndReason::SchedulerStopped);

            bool full = false;
            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                request = entry->request;
                processed = entry->snapshot.promptTokensProcessed;
                // Finalizing consumes the last emitted token but cannot emit
                // another. A full ring must not turn successful finalization
                // into a slow-consumer cancellation.
                full = !entry->detached && entry->buffered == entry->buffer.size() &&
                       entry->snapshot.status != RequestStatus::Finalizing;
            }
            if (full) entry->Cancel(RequestEndReason::OutputBackpressure);
            // Also observe the scheduler marker directly. Pending-cancel
            // maintenance must never execute a token-producing step.
            if (entry->cancellation.load(std::memory_order_acquire) != RequestEndReason::None) {
                request->Cancel();
            }

            const auto step = request->Step();
            const auto progress = request->Progress();
            processed = progress.promptTokensProcessed;
            {
                std::lock_guard<std::mutex> lock(mutex);
                domainFaulted = faulted;
            }
            if (domainFaulted && !IsTerminal(step.status)) {
                // A deferred log callback or concurrent Submit can discover
                // a domain failure after Step prepared its return value.
                // Retire it without publishing that now-invalid sample.
                request->Cancel();
                request->Step();
                request.reset();
                Complete(entry, RequestStatus::Failed, RequestEndReason::RuntimeFault,
                         ErrorCode::RuntimeFault, "Scheduler execution domain has failed", processed);
                return true;
            }
            if (IsTerminal(step.status)) {
                RequestStatus status = step.status;
                RequestEndReason reason = RequestEndReason::Failed;
                auto errorCode = progress.errorCode;
                std::string_view error = progress.ErrorMessage();
                if (domainFaulted || errorCode == ErrorCode::RuntimeFault) {
                    Fault();
                    status = RequestStatus::Failed;
                    reason = RequestEndReason::RuntimeFault;
                    errorCode = ErrorCode::RuntimeFault;
                    if (error.empty()) error = "Scheduler execution domain has failed";
                } else if (status == RequestStatus::Completed) {
                    reason = progress.generatedTokens == entry->maxNewTokens
                                 ? RequestEndReason::Length : RequestEndReason::StopToken;
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

            {
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
            entry->changed.notify_all();
            return false;
        } catch (const std::exception& error) {
            // A failure outside Request::Step must still release the session
            // lease safely. Progress itself has fixed, nonallocating storage.
            if (request) {
                request->Cancel();
                try { request->Step(); } catch (...) {}
                request.reset();
            }
            Complete(entry, RequestStatus::Failed, RequestEndReason::Failed,
                     std::nullopt, error.what(), processed);
            return true;
        } catch (...) {
            if (request) {
                request->Cancel();
                try { request->Step(); } catch (...) {}
                request.reset();
            }
            Complete(entry, RequestStatus::Failed, RequestEndReason::Failed,
                     std::nullopt, "Unknown scheduler request failure", processed);
            return true;
        }
    }

    void Run() noexcept {
        // Splicing list nodes rotates requests without allocating in the hot
        // path. The running node remains local until all locks are released.
        std::list<std::shared_ptr<ScheduledRequestState>> current;
        bool lastWasQueuedCancellation = false;
        for (;;) {
            bool wasActive = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return stopping || faulted || !waiting.empty() || !runnable.empty(); });
                if (waiting.empty() && runnable.empty() && (stopping || faulted)) break;

                while (activeRequests < options.maxActiveRequests && !waiting.empty()) {
                    runnable.splice(runnable.end(), waiting, waiting.begin());
                    ++activeRequests;
                }

                // Pending cancellation does not need an active slot or GPU
                // work. Alternate this maintenance with runnable work so a
                // stream of new cancelled submissions cannot starve decode.
                auto cancelled = waiting.end();
                if (!lastWasQueuedCancellation || runnable.empty()) {
                    cancelled = std::find_if(waiting.begin(), waiting.end(), [&](const auto& entry) {
                        return stopping || faulted ||
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
                } else {
                    continue;
                }
                running = current.front();
            }

            const bool completed = RunOne(current.front());
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (completed) {
                    if (wasActive) --activeRequests;
                } else if (wasActive) {
                    // Give newly eligible requests a turn before the one
                    // that just finished its quantum is appended again.
                    while (activeRequests < options.maxActiveRequests && !waiting.empty()) {
                        runnable.splice(runnable.end(), waiting, waiting.begin());
                        ++activeRequests;
                    }
                    runnable.splice(runnable.end(), current, current.begin());
                } else {
                    // A maintenance node never owns an active slot. Preserve
                    // accounting even if a future maintenance action yields.
                    waiting.splice(waiting.end(), current, current.begin());
                }
                running.reset(); // current still owns it until after unlock.
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
    std::shared_ptr<ScheduledRequestState> running;
    size_t activeRequests = 0; // Includes the currently executing node.
    size_t outstandingRequests = 0;
    size_t reservedTokens = 0;
    size_t reservedCacheBytes = 0;
    uint64_t nextRequestId = 1;
    bool stopping = false;
    bool faulted = false;
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

SchedulerSnapshot Scheduler::Snapshot() const {
    const auto retained = state;
    std::lock_guard<std::mutex> lock(retained->mutex);
    return {!retained->stopping && !retained->faulted, retained->stopping, retained->faulted,
            retained->outstandingRequests, retained->waiting.size(), retained->activeRequests,
            retained->reservedTokens, retained->reservedCacheBytes};
}

std::shared_ptr<ScheduledRequest> Scheduler::Submit(
    std::shared_ptr<SessionState> session, std::vector<int> promptIds, RequestOptions options) {
    const auto retained = state;
    const auto& limits = retained->options;
    if (!session || promptIds.empty() || options.generation.max_new_tokens <= 0 ||
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
    } catch (const Error& error) {
        if (error.Code() == ErrorCode::RuntimeFault) retained->Fault();
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
