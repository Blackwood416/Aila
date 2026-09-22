#include "inference/Scheduler.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using namespace std::chrono_literals;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Callable>
void ExpectError(ErrorCode code, Callable&& callable) {
    try {
        callable();
    } catch (const Error& error) {
        Expect(error.Code() == code, "Wrong inference error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected an inference error");
}

// The action retains the shared gate state after this owner's destructor has
// released it. Declaring a gate after Scheduler makes assertion unwinding safe.
class ForwardGate final {
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered = false;
        bool released = false;
    };

public:
    ForwardGate() : state(std::make_shared<State>()) {}
    ~ForwardGate() { Release(); }
    std::function<void()> Action() const {
        auto retained = state;
        return [retained] {
            std::unique_lock<std::mutex> lock(retained->mutex);
            retained->entered = true;
            retained->changed.notify_all();
            if (!retained->changed.wait_for(lock, 5s, [&] { return retained->released; })) {
                throw std::runtime_error("CPU forward gate was not released");
            }
        };
    }
    void AwaitEntry() const {
        std::unique_lock<std::mutex> lock(state->mutex);
        Expect(state->changed.wait_for(lock, 5s, [&] { return state->entered; }),
               "CPU forward did not reach its expected gate");
    }
    void Release() const noexcept {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->released = true;
        state->changed.notify_all();
    }

private:
    std::shared_ptr<State> state;
};

struct TraceEntry {
    int sessionId;
    std::vector<int> input;
    bool sampling;
    std::thread::id thread;
};

class ExecutionTrace final {
public:
    void Enter(int sessionId, const std::vector<int>& input, bool sampling) {
        std::lock_guard<std::mutex> lock(mutex);
        ++inFlight;
        maximumInFlight = std::max(maximumInFlight, inFlight);
        entries.push_back({sessionId, input, sampling, std::this_thread::get_id()});
    }
    void Leave() {
        std::lock_guard<std::mutex> lock(mutex);
        --inFlight;
    }
    std::vector<TraceEntry> Entries() const {
        std::lock_guard<std::mutex> lock(mutex);
        return entries;
    }
    size_t MaximumInFlight() const {
        std::lock_guard<std::mutex> lock(mutex);
        return maximumInFlight;
    }

private:
    mutable std::mutex mutex;
    std::vector<TraceEntry> entries;
    size_t inFlight = 0;
    size_t maximumInFlight = 0;
};

enum class Failure { None, Ordinary, InvalidArgument, RuntimeFault };

struct FakeBatch {
    std::function<void(size_t)> duringBatch;
    std::function<void(size_t)> afterBatch;
    std::vector<std::vector<int>> Groups() const {
        std::lock_guard<std::mutex> lock(mutex);
        return groups;
    }
    size_t Record(std::vector<int> ids) {
        std::lock_guard<std::mutex> lock(mutex);
        groups.push_back(std::move(ids));
        return groups.size();
    }
    mutable std::mutex mutex;
    std::vector<std::vector<int>> groups;
};

struct FakeState {
    explicit FakeState(int id, int capacity, size_t cacheBytes,
                       std::shared_ptr<ExecutionTrace> trace, std::shared_ptr<FakeBatch> batch)
        : id(id), capacity(capacity), cacheBytes(cacheBytes), trace(std::move(trace)), batch(std::move(batch)) {}

    std::vector<int> Context() const {
        std::lock_guard<std::mutex> lock(mutex);
        return context;
    }
    std::vector<std::vector<int>> Inputs() const {
        std::lock_guard<std::mutex> lock(mutex);
        return inputs;
    }
    std::vector<std::vector<int>> Histories() const {
        std::lock_guard<std::mutex> lock(mutex);
        return histories;
    }

    const int id;
    const int capacity;
    const size_t cacheBytes;
    const int vocabSize = 128;
    const std::shared_ptr<ExecutionTrace> trace;
    const std::shared_ptr<FakeBatch> batch;
    // Configured before Submit, then read only by request operations.
    std::function<void(size_t, bool)> duringForward;
    std::function<void()> duringContextLength;
    std::function<void()> duringDestruction;
    std::atomic<bool> destructionCallbackFailed{false};
    std::vector<int> samples;
    Failure failure = Failure::None;
    Failure contextFailure = Failure::None;
    bool invalidContextLength = false;
    size_t failOnCall = 1;
    std::string failureMessage = "fake backend failure";

    mutable std::mutex mutex;
    std::vector<int> context;
    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> histories;
    size_t samplesTaken = 0;
};

class FakeBackend final : public SessionBackend {
public:
    explicit FakeBackend(std::shared_ptr<FakeState> state) : state(std::move(state)) {}
    ~FakeBackend() override {
        try {
            if (state->duringDestruction) state->duringDestruction();
        } catch (...) {
            state->destructionCallbackFailed.store(true);
        }
    }
    int ContextLength() const override {
        if (state->duringContextLength) state->duringContextLength();
        if (state->contextFailure == Failure::RuntimeFault) {
            throw Error(ErrorCode::RuntimeFault, "fake context runtime fault");
        }
        if (state->contextFailure == Failure::InvalidArgument) {
            throw Error(ErrorCode::InvalidArgument, "fake context validation failure");
        }
        if (state->invalidContextLength) return state->capacity + 1;
        return static_cast<int>(state->Context().size());
    }
    int Capacity() const override { return state->capacity; }
    int VocabSize() const override { return state->vocabSize; }
    size_t CacheBytes() const override { return state->cacheBytes; }
    const void* DecodeBatchKey() const noexcept override { return state->batch.get(); }
    std::vector<ExecutionResult> ForwardDecodeBatch(const std::vector<BackendDecodeInput>& rows) override {
        Expect(state->batch != nullptr, "Unsupported fake backend received a batch");
        std::vector<int> ids;
        for (const auto& row : rows) {
            auto* peer = dynamic_cast<FakeBackend*>(row.backend);
            Expect(peer && peer->state->batch == state->batch, "Decode batch mixed loaded model identities");
            ids.push_back(peer->state->id);
        }
        const size_t call = state->batch->Record(std::move(ids));
        if (state->batch->duringBatch) state->batch->duringBatch(call);
        std::vector<ExecutionResult> result;
        for (const auto& row : rows) {
            result.push_back(row.backend->Forward({row.tokenId}, row.sampling, row.captureLogits));
        }
        if (state->batch->afterBatch) state->batch->afterBatch(call);
        return result;
    }
    void Reset() override {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->context.clear();
    }
    int Truncate(int length) override {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->context.resize(static_cast<size_t>(length));
        return length;
    }
    ExecutionResult Forward(const std::vector<int>& input,
                            const SamplingInput* sampling, bool captureLogits) override {
        state->trace->Enter(state->id, input, sampling != nullptr);
        struct EndForward {
            ExecutionTrace& trace;
            ~EndForward() { trace.Leave(); }
        } end{*state->trace};
        size_t call;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->inputs.push_back(input);
            call = state->inputs.size();
            state->context.insert(state->context.end(), input.begin(), input.end());
        }
        if (state->duringForward) state->duringForward(call, sampling != nullptr);
        if (call == state->failOnCall) {
            if (state->failure == Failure::Ordinary) throw std::runtime_error(state->failureMessage);
            if (state->failure == Failure::InvalidArgument) {
                throw Error(ErrorCode::InvalidArgument, state->failureMessage);
            }
            if (state->failure == Failure::RuntimeFault) {
                throw Error(ErrorCode::RuntimeFault, state->failureMessage);
            }
        }

        std::lock_guard<std::mutex> lock(state->mutex);
        uint32_t hash = 19;
        for (int token : state->context) hash = hash * 33u + static_cast<uint32_t>(token);
        ExecutionResult result;
        if (sampling) {
            state->histories.push_back(sampling->generatedIds);
            for (int token : sampling->generatedIds) hash = hash * 17u + static_cast<uint32_t>(token);
            hash ^= static_cast<uint32_t>(static_cast<int>(sampling->config.repetition_penalty * 1000.0f));
            hash += static_cast<uint32_t>(static_cast<int>(sampling->config.frequency_penalty * 1000.0f));
            if (sampling->config.do_sample) hash ^= sampling->rng();
            result.tokenId = state->samplesTaken < state->samples.size()
                                 ? state->samples[state->samplesTaken]
                                 : static_cast<int>(hash % static_cast<uint32_t>(state->vocabSize));
            ++state->samplesTaken;
        }
        if (captureLogits) result.logits.assign(static_cast<size_t>(state->vocabSize), 0.5f);
        return result;
    }

private:
    std::shared_ptr<FakeState> state;
};

struct Fixture {
    std::shared_ptr<FakeState> state;
    std::shared_ptr<SessionState> session;
};

Fixture MakeFixture(int id = 1,
                    std::shared_ptr<ExecutionTrace> trace = std::make_shared<ExecutionTrace>(),
                    int capacity = 128, size_t cacheBytes = 64, std::shared_ptr<FakeBatch> batch = {}) {
    auto state = std::make_shared<FakeState>(id, capacity, cacheBytes, std::move(trace), std::move(batch));
    auto session = std::make_shared<SessionState>(std::make_unique<FakeBackend>(state));
    return {std::move(state), std::move(session)};
}

RequestOptions Options(int newTokens = 4, int chunk = 2, uint64_t seed = 42) {
    RequestOptions options;
    options.generation.max_new_tokens = newTokens;
    options.generation.do_sample = true;
    options.generation.use_fixed_seed = true;
    options.generation.sampling_seed = seed;
    options.prefillChunkSize = chunk;
    return options;
}

SchedulerOptions Limits() {
    SchedulerOptions options;
    options.maxActiveRequests = 2;
    options.maxOutstandingRequests = 8;
    options.maxPromptTokens = 128;
    options.maxGeneratedTokens = 32;
    options.maxOutstandingTokens = 1024;
    options.maxPrefillChunkSize = 2;
    options.maxBufferedTokensPerRequest = 32;
    return options;
}

struct StreamResult {
    std::vector<int> tokens;
    RequestEvent terminal;
};

StreamResult Drain(const std::shared_ptr<ScheduledRequest>& stream,
                   std::vector<RequestEvent> prefix = {}) {
    for (size_t i = prefix.size(); i < 128; ++i) {
        if (!prefix.empty() && prefix.back().kind == RequestEventKind::Terminal) break;
        auto read = stream->Read(5s);
        Expect(read.status == ReadStatus::Event && read.event.has_value(),
               "Stream did not provide its next event within the CPU test deadline");
        prefix.push_back(std::move(*read.event));
    }
    Expect(!prefix.empty() && prefix.back().kind == RequestEventKind::Terminal,
           "Stream did not terminate within the event bound");
    StreamResult result;
    for (size_t i = 0; i < prefix.size(); ++i) {
        const auto& event = prefix[i];
        Expect(event.requestId == stream->Id() && event.sequence == i + 1,
               "Event ID or one-based sequence is inconsistent");
        Expect(event.error.size() <= 1024, "Stream error exceeds its bounded payload");
        if (event.kind == RequestEventKind::Token) {
            result.tokens.push_back(event.tokenId);
            Expect(event.tokenId >= 0 && !IsTerminal(event.status) &&
                       event.endReason == RequestEndReason::None &&
                       event.generatedTokens == result.tokens.size(),
                   "Token event contains inconsistent progress or terminal metadata");
        } else {
            Expect(i + 1 == prefix.size() && IsTerminal(event.status) && event.tokenId == -1 &&
                       event.endReason != RequestEndReason::None &&
                       event.generatedTokens == result.tokens.size(),
                   "Terminal is duplicated, out of order, or has inconsistent progress");
            result.terminal = event;
        }
    }
    const auto closed = stream->Read(0ms);
    Expect(closed.status == ReadStatus::Closed && !closed.event,
           "Reading after terminal did not return Closed without an event");
    stream->Cancel();
    const auto closedAgain = stream->Read(0ms);
    Expect(closedAgain.status == ReadStatus::Closed && !closedAgain.event,
           "Cancel after terminal reopened or duplicated the stream");
    const auto snapshot = stream->Snapshot();
    Expect(snapshot.terminalConsumed && snapshot.bufferedTokens == 0 &&
               snapshot.status == result.terminal.status &&
               snapshot.endReason == result.terminal.endReason &&
               snapshot.generatedTokens == result.tokens.size() &&
               snapshot.errorCode == result.terminal.errorCode && snapshot.error == result.terminal.error &&
               snapshot.error.size() <= 1024,
           "Consumed stream snapshot disagrees with its terminal event");
    return result;
}

void ExpectEnd(const StreamResult& result, RequestStatus status, RequestEndReason reason) {
    Expect(result.terminal.status == status && result.terminal.endReason == reason,
           "Unexpected terminal status or end reason");
}

void ExpectNoReservations(const Scheduler& scheduler) {
    const auto snapshot = scheduler.Snapshot();
    Expect(snapshot.outstandingRequests == 0 && snapshot.queuedRequests == 0 &&
               snapshot.activeRequests == 0 && snapshot.reservedTokens == 0 &&
               snapshot.reservedSessionCacheBytes == 0,
           "Scheduler leaked requests, active slots, or admission reservations");
}

void TestPrefillAdmissionCancellationAndPeer() {
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    Scheduler scheduler(Limits());
    ForwardGate gate;
    a.state->duringForward = [action = gate.Action()](size_t, bool sampling) {
        Expect(!sampling, "Replay reached the sampler");
        action();
    };
    auto options = Options(0, 99);
    options.prefillOnly = true;
    auto replay = scheduler.Submit(a.session, {1, 2, 3, 4, 5}, options);
    gate.AwaitEntry();
    const auto reserved = scheduler.Snapshot();
    Expect(reserved.reservedTokens == 5 && reserved.outstandingRequests == 1,
           "Replay did not reserve its prompt-only token count");
    auto peer = scheduler.Submit(b.session, {6, 7}, Options());
    replay->Cancel();
    gate.Release();
    const auto cancelled = Drain(replay);
    ExpectEnd(cancelled, RequestStatus::Cancelled, RequestEndReason::Cancelled);
    Expect(cancelled.tokens.empty() && cancelled.terminal.sequence == 1 &&
               cancelled.terminal.promptTokensProcessed == 2 && a.state->Context() == std::vector<int>({1, 2}),
           "Cancelled scheduled replay did not retain exactly one clamped prompt chunk");
    ExpectEnd(Drain(peer), RequestStatus::Completed, RequestEndReason::Length);
    a.state->duringForward = {};
    const auto repaired = Drain(scheduler.Submit(a.session, {3, 4, 5}, options));
    ExpectEnd(repaired, RequestStatus::Completed, RequestEndReason::Length);
    Expect(repaired.tokens.empty() && a.state->Context() == std::vector<int>({1, 2, 3, 4, 5}) &&
               a.state->Histories().empty(), "Scheduled replay recovery sampled or corrupted the accepted prefix");
    ExpectNoReservations(scheduler);
}

std::vector<int> RunDirect(const Fixture& fixture, const std::vector<int>& prompt,
                           RequestOptions options) {
    auto request = RequestState::Create(fixture.session, prompt, std::move(options));
    std::vector<int> tokens;
    for (int i = 0; i < 128; ++i) {
        const auto step = request->Step();
        if (step.tokenId) tokens.push_back(*step.tokenId);
        if (IsTerminal(step.status)) {
            Expect(step.status == RequestStatus::Completed, "Serial reference request failed");
            return tokens;
        }
    }
    throw std::runtime_error("Serial reference did not finish");
}

void TestThinkingBudgetEarlyLengthAndPeer() {
    Scheduler scheduler(Limits());
    auto a = MakeFixture(1), b = MakeFixture(2);
    a.state->samples = {20, 21, 22};
    b.state->samples = {30, 31, 32};
    auto options = Options(3);
    options.thinkingBudget = aila::generation::ThinkingBudgetOptions{2, 10, 11, true, {11, 3}};
    auto limited = scheduler.Submit(a.session, {1}, options);
    auto peer = scheduler.Submit(b.session, {2}, Options(3));
    const auto result = Drain(limited);
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::Length);
    Expect(result.tokens == std::vector<int>({20, 21}) && a.state->Context() == std::vector<int>({1, 20, 21}),
           "Thinking close overflow did not retain exact emitted/consumed prefix");
    const auto other = Drain(peer);
    ExpectEnd(other, RequestStatus::Completed, RequestEndReason::Length);
    Expect(other.tokens == std::vector<int>({30, 31, 32}), "Budgeted request damaged a peer");
    ExpectNoReservations(scheduler);
    scheduler.Shutdown();
}

void TestValidationAndAdmissionFailures() {
    const std::vector<std::function<void(SchedulerOptions&)>> invalidLimits{
        [](auto& o) { o.maxActiveRequests = 0; },
        [](auto& o) { o.maxOutstandingRequests = 0; },
        [](auto& o) { o.maxPromptTokens = 0; },
        [](auto& o) { o.maxGeneratedTokens = 0; },
        [](auto& o) { o.maxStopTokenIds = 0; },
        [](auto& o) { o.maxOutstandingTokens = 0; },
        [](auto& o) { o.maxPrefillChunkSize = 0; },
        [](auto& o) { o.maxPrefillChunkSize = -1; },
        [](auto& o) { o.maxBufferedTokensPerRequest = 0; },
    };
    for (const auto& edit : invalidLimits) {
        auto limits = Limits();
        edit(limits);
        ExpectError(ErrorCode::InvalidArgument, [&] { Scheduler scheduler(limits); });
    }
    auto limits = Limits();
    limits.maxPromptTokens = 4;
    limits.maxGeneratedTokens = 3;
    Scheduler scheduler(limits);
    auto fixture = MakeFixture();
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(nullptr, {1}, Options(1)); });
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(fixture.session, {}, Options(1)); });
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(fixture.session, {-1}, Options(1)); });
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(fixture.session, {1}, Options(1, 0)); });
    auto unsupported = Options(1);
    unsupported.generation.thinking_budget_tokens = 1;
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(fixture.session, {1}, unsupported); });
    ExpectError(ErrorCode::ResourceLimit, [&] {
        scheduler.Submit(fixture.session, {1, 2, 3, 4, 5}, Options(1));
    });
    ExpectError(ErrorCode::ResourceLimit, [&] { scheduler.Submit(fixture.session, {1}, Options(4)); });
    auto small = MakeFixture(2, std::make_shared<ExecutionTrace>(), 4);
    ExpectError(ErrorCode::ContextOverflow, [&] { scheduler.Submit(small.session, {1, 2, 3}, Options(2)); });
    Expect(!fixture.session->Busy() && !small.session->Busy() && fixture.state->Inputs().empty(),
           "Rejected Submit acquired a lasting session lease or submitted work");
    ExpectNoReservations(scheduler);
    const auto good = Drain(scheduler.Submit(fixture.session, {1, 2, 3, 4}, Options(3)));
    ExpectEnd(good, RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestReadAndFinalizationWithFullRing() {
    auto limits = Limits();
    limits.maxBufferedTokensPerRequest = 1;
    Scheduler scheduler(limits);
    ForwardGate first;
    ForwardGate finalizing;
    auto fixture = MakeFixture();
    fixture.state->samples = {7};
    fixture.state->duringForward = [begin = first.Action(), end = finalizing.Action()](size_t call, bool) {
        if (call == 1) begin();
        if (call == 3) end();
    };
    auto stream = scheduler.Submit(fixture.session, {1, 2, 3}, Options(1));
    first.AwaitEntry();
    const auto empty = stream->Read(0ms);
    Expect(empty.status == ReadStatus::Timeout && !empty.event,
           "An empty live stream did not report Timeout without an event");
    first.Release();
    // Do not consume token 7: the full ring must still allow the final cache flush.
    finalizing.AwaitEntry();
    Expect(stream->Snapshot().bufferedTokens == 1 && fixture.session->Busy(),
           "Finalization did not retain the full output ring and active lease");
    finalizing.Release();
    const auto result = Drain(stream);
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::Length);
    Expect(result.tokens == std::vector<int>({7}) && result.terminal.promptTokensProcessed == 3 &&
               fixture.state->Context() == std::vector<int>({1, 2, 3, 7}) && !fixture.session->Busy(),
           "Final output token was lost, duplicated, or not cached");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestStopTokenTerminal() {
    Scheduler scheduler(Limits());
    auto fixture = MakeFixture();
    fixture.state->samples = {5, 7};
    auto options = Options(5);
    options.stopTokenIds = {7};
    const auto result = Drain(scheduler.Submit(fixture.session, {1, 2}, options));
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::StopToken);
    Expect(result.tokens == std::vector<int>({5}) &&
               fixture.state->Context() == std::vector<int>({1, 2, 5}),
           "Stop token was published/cached, or an earlier output was lost");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestStopTokenAdmissionLimit() {
    SchedulerOptions limits;
    Expect(limits.maxStopTokenIds == 64, "Default stop-token limit changed unexpectedly");
    Scheduler scheduler(limits);
    auto fixture = MakeFixture();
    fixture.state->samples = {7};
    auto options = Options(2);
    // All IDs are valid and equal: the bound is input size, not unique IDs.
    options.stopTokenIds.assign(limits.maxStopTokenIds + 1, 7);
    ExpectError(ErrorCode::ResourceLimit, [&] { scheduler.Submit(fixture.session, {1, 2}, options); });
    ExpectNoReservations(scheduler);
    Expect(!fixture.session->Busy() && fixture.state->Inputs().empty(),
           "Oversized stop-token list consumed admission or acquired a session lease");
    options.stopTokenIds.pop_back();
    const auto result = Drain(scheduler.Submit(fixture.session, {1, 2}, options));
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::StopToken);
    Expect(result.tokens.empty() && fixture.state->Context() == std::vector<int>({1, 2}),
           "Exact-limit stop-token list failed to stop without publishing/caching the stop ID");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestFairRoundRobinAndSeededReference() {
    const std::vector<int> promptA{1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    const std::vector<int> promptB{2, 4};
    auto optionsA = Options(3, 2, 42);
    auto optionsB = Options(4, 2, 987654);
    optionsA.generation.repetition_penalty = 1.2f;
    optionsB.generation.frequency_penalty = -0.3f;
    auto referenceA = MakeFixture();
    auto referenceB = MakeFixture();
    const auto expectedA = RunDirect(referenceA, promptA, optionsA);
    const auto expectedB = RunDirect(referenceB, promptB, optionsB);

    Scheduler scheduler(Limits());
    ForwardGate initial;
    auto trace = std::make_shared<ExecutionTrace>();
    auto a = MakeFixture(1, trace);
    auto b = MakeFixture(2, trace);
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    optionsA.prefillChunkSize = 99;
    optionsB.prefillChunkSize = 99;
    auto streamA = scheduler.Submit(a.session, promptA, optionsA);
    initial.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, promptB, optionsB);
    Expect(streamA->Id() != streamB->Id(), "Two admitted requests share an ID");
    initial.Release();
    const auto actualA = Drain(streamA);
    const auto actualB = Drain(streamB);
    ExpectEnd(actualA, RequestStatus::Completed, RequestEndReason::Length);
    ExpectEnd(actualB, RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    Expect(actualA.tokens == expectedA && actualB.tokens == expectedB &&
               a.state->Context() == referenceA.state->Context() &&
               b.state->Context() == referenceB.state->Context() &&
               a.state->Histories() == referenceA.state->Histories() &&
               b.state->Histories() == referenceB.state->Histories(),
           "Scheduling/clamped prefill changed fixed-seed tokens, cache, or sampling history");
    const auto entries = trace->Entries();
    Expect(entries.size() == 13 && trace->MaximumInFlight() == 1,
           "Scheduler submitted extra work or overlapping forwards");
    const auto worker = entries.front().thread;
    Expect(worker != std::this_thread::get_id(), "Submit performed inference on the caller thread");
    size_t callsA = 0;
    size_t callsB = 0;
    int previous = 0;
    bool sawB = false;
    bool mixedPhases = false;
    for (const auto& entry : entries) {
        Expect(entry.thread == worker, "Forwards used more than one submission thread");
        if (sawB && callsA < 8 && callsB < 5) {
            Expect(entry.sessionId != previous, "Runnable requests did not alternate at forward boundaries");
        }
        if (entry.sessionId == 1) {
            ++callsA;
            Expect(entry.input.size() == (callsA <= 5 ? 2u : 1u),
                   "Scheduler did not clamp prefill chunks or submitted a multi-token decode");
            if (callsB >= 2 && !entry.sampling && callsA <= 5) mixedPhases = true;
        } else {
            Expect(entry.sessionId == 2, "Trace contains an unknown session");
            if (!sawB) Expect(callsA <= 2, "Newly admitted peer starved behind repeated prefill chunks");
            sawB = true;
            ++callsB;
            Expect(entry.input.size() == (callsB == 1 ? 2u : 1u), "B consumed an invalid step shape");
        }
        previous = entry.sessionId;
    }
    Expect(callsA == 8 && callsB == 5 && mixedPhases,
           "Test did not exercise prefill/decode interleaving and both finalization steps");
    ExpectNoReservations(scheduler);
}

void TestActiveLimitPendingCancelAndDrop() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    limits.maxOutstandingRequests = 3;
    Scheduler scheduler(limits);
    ForwardGate first;
    ForwardGate second;
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    auto c = MakeFixture(3);
    a.state->duringForward = [one = first.Action(), two = second.Action()](size_t call, bool) {
        if (call == 1) one();
        // Allow cancellation maintenance to alternate with runnable work.
        if (call == 3) two();
    };
    auto streamA = scheduler.Submit(a.session, {1, 2, 3, 4, 5, 6, 7, 8}, Options(2));
    first.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, {7}, Options(1));
    auto dropped = scheduler.Submit(c.session, {8}, Options(1));
    const auto held = scheduler.Snapshot();
    Expect(held.activeRequests == 1 && held.queuedRequests == 2 && held.outstandingRequests == 3,
           "Active limit did not keep additional requests pending");
    ExpectError(ErrorCode::Busy, [&] { a.session->Reset(); });
    // Use an otherwise available reservation so Busy is not masked by admission.
    streamB->Cancel();
    streamB->Cancel();
    dropped.reset();
    first.Release();
    second.AwaitEntry();
    const auto cancelled = Drain(streamB);
    ExpectEnd(cancelled, RequestStatus::Cancelled, RequestEndReason::Cancelled);
    Expect(cancelled.tokens.empty() && b.state->Inputs().empty() && c.state->Inputs().empty() &&
               !b.session->Busy() && !c.session->Busy(),
           "Pending cancel/drop submitted work or retained a session lease");
    Expect(scheduler.Snapshot().outstandingRequests == 1,
           "Dropped pending handle or consumed terminal retained admission");
    ExpectError(ErrorCode::Busy, [&] { scheduler.Submit(a.session, {9}, Options(1)); });
    Expect(scheduler.Snapshot().outstandingRequests == 1, "Busy rejection consumed admission");
    auto replacement = scheduler.Submit(c.session, {8}, Options(1));
    streamA->Cancel();
    second.Release();
    const auto cancelledA = Drain(streamA);
    ExpectEnd(cancelledA, RequestStatus::Cancelled, RequestEndReason::Cancelled);
    Expect(cancelledA.tokens.empty() && a.state->Context() == std::vector<int>({1, 2, 3, 4, 5, 6}),
           "Active cancellation changed the completed prefill prefix");
    ExpectEnd(Drain(replacement), RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestCancellationDuringForward() {
    Scheduler scheduler(Limits());
    ForwardGate inFlight;
    auto fixture = MakeFixture();
    fixture.state->duringForward = [hold = inFlight.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto stream = scheduler.Submit(fixture.session, {1, 2}, Options(4));
    inFlight.AwaitEntry();
    stream->Cancel();
    stream->Cancel();
    Expect(fixture.session->Busy() && stream->Read(0ms).status == ReadStatus::Timeout,
           "Cancel published terminal or released session during an unfinished forward");
    inFlight.Release();
    const auto result = Drain(stream);
    ExpectEnd(result, RequestStatus::Cancelled, RequestEndReason::Cancelled);
    Expect(result.tokens.empty() && result.terminal.promptTokensProcessed == 2 &&
               fixture.state->Inputs().size() == 1 && fixture.state->Context() == std::vector<int>({1, 2}) &&
               !fixture.session->Busy(),
           "In-flight cancellation published a sampled token or damaged the consumed prefix");
    fixture.session->Reset();
    ExpectEnd(Drain(scheduler.Submit(fixture.session, {3}, Options(1))),
              RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestBackpressureOnlyCancelsSlowConsumer() {
    auto limits = Limits();
    limits.maxBufferedTokensPerRequest = 1;
    Scheduler scheduler(limits);
    ForwardGate initial;
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    a.state->samples = {11, 12};
    b.state->samples = {21};
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto slow = scheduler.Submit(a.session, {1}, Options(4));
    initial.AwaitEntry();
    auto fast = scheduler.Submit(b.session, {2}, Options(1));
    initial.Release();
    // No reads from slow until its peer has finalized. With two active slots,
    // slow must encounter its full ring before fast's finalization completes.
    const auto fastResult = Drain(fast);
    ExpectEnd(fastResult, RequestStatus::Completed, RequestEndReason::Length);
    const auto blocked = slow->Snapshot();
    Expect(blocked.status == RequestStatus::Cancelled &&
               blocked.endReason == RequestEndReason::OutputBackpressure &&
               blocked.bufferedTokens == 1 && !blocked.terminalConsumed,
           "Full ring did not cancel only the slow request while retaining its output");
    Expect(scheduler.Snapshot().outstandingRequests == 1,
           "Unread backpressure terminal did not retain admission");
    const auto slowResult = Drain(slow);
    ExpectEnd(slowResult, RequestStatus::Cancelled, RequestEndReason::OutputBackpressure);
    Expect(slowResult.tokens == std::vector<int>({11}) && fastResult.tokens == std::vector<int>({21}) &&
               a.state->Inputs().size() == 1 && a.state->Context() == std::vector<int>({1}) &&
               b.state->Context() == std::vector<int>({2, 21}),
           "Backpressure discarded queued tokens, submitted an extra decode, or affected the peer");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestCompletedStreamsRetainAdmission() {
    // Isolate request-count, token-budget and cache-budget enforcement, including
    // size_t addition overflow, so one working bound cannot mask another bug.
    // The last case drops a completed handle with an unread terminal instead.
    for (int dimension = 0; dimension < 5; ++dimension) {
        auto limits = Limits();
        limits.maxActiveRequests = 1;
        limits.maxOutstandingRequests = dimension == 0 || dimension == 4 ? 2 : 8;
        limits.maxOutstandingTokens = dimension == 1 ? 7 : 1024;
        const size_t cacheA = dimension == 3 ? (std::numeric_limits<size_t>::max)() - 3 : 3;
        const size_t cacheB = dimension == 3 ? 3 : 5;
        if (dimension == 2 || dimension == 3) limits.maxSessionCacheBytes = cacheA + cacheB;
        Scheduler scheduler(limits);
        ForwardGate holder;
        auto a = MakeFixture(1, std::make_shared<ExecutionTrace>(), 128, cacheA);
        auto b = MakeFixture(2, std::make_shared<ExecutionTrace>(), 128, cacheB);
        auto c = MakeFixture(3, std::make_shared<ExecutionTrace>(), 128, 1);
        b.state->duringForward = [hold = holder.Action()](size_t call, bool) { if (call == 1) hold(); };
        auto completed = scheduler.Submit(a.session, {1, 2}, Options(2)); // 4 tokens reserved.
        const auto completedId = completed->Id();
        auto running = scheduler.Submit(b.session, {3, 4}, Options(1)); // 3 tokens reserved.
        holder.AwaitEntry(); // maxActive=1 means A's terminal is already published.
        const auto unread = completed->Snapshot();
        const auto full = scheduler.Snapshot();
        Expect(unread.status == RequestStatus::Completed && !unread.terminalConsumed &&
                   unread.bufferedTokens == 2 && full.outstandingRequests == 2 &&
                   full.reservedTokens == 7 && full.reservedSessionCacheBytes == cacheA + cacheB,
               "Completed unread stream did not retain its admission reservations");
        ExpectError(ErrorCode::ResourceLimit, [&] { scheduler.Submit(c.session, {5}, Options(1)); });
        const auto first = completed->Read(0ms);
        Expect(first.status == ReadStatus::Event && first.event && first.event->kind == RequestEventKind::Token,
               "Completed stream did not preserve its first buffered token");
        ExpectError(ErrorCode::ResourceLimit, [&] { scheduler.Submit(c.session, {5}, Options(1)); });
        Expect(!c.session->Busy() && c.state->Inputs().empty(), "Rejected admission acquired a lasting lease");
        if (dimension == 4) {
            completed.reset();
        } else {
            ExpectEnd(Drain(completed, {*first.event}), RequestStatus::Completed, RequestEndReason::Length);
        }
        const auto released = scheduler.Snapshot();
        Expect(released.outstandingRequests == 1 && released.reservedTokens == 3 &&
                   released.reservedSessionCacheBytes == cacheB,
               "Consuming terminal or dropping a completed handle did not return its reservations");
        auto admitted = scheduler.Submit(c.session, {5}, Options(1));
        Expect(admitted->Id() != completedId && admitted->Id() != running->Id(),
               "Admission reused a still-visible request ID");
        holder.Release();
        ExpectEnd(Drain(running), RequestStatus::Completed, RequestEndReason::Length);
        ExpectEnd(Drain(admitted), RequestStatus::Completed, RequestEndReason::Length);
        scheduler.Shutdown();
        ExpectNoReservations(scheduler);
    }
}

void TestOrdinaryFailuresStayLocal() {
    for (Failure failure : {Failure::Ordinary, Failure::InvalidArgument}) {
        auto limits = Limits();
        limits.maxActiveRequests = 1;
        Scheduler scheduler(limits);
        ForwardGate initial;
        auto bad = MakeFixture(1);
        auto good = MakeFixture(2);
        bad.state->failure = failure;
        bad.state->failureMessage.assign(1500, 'x');
        bad.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
        auto failed = scheduler.Submit(bad.session, {1, 2}, Options(2));
        initial.AwaitEntry();
        auto peer = scheduler.Submit(good.session, {3}, Options(1));
        initial.Release();
        const auto error = Drain(failed);
        ExpectEnd(error, RequestStatus::Failed, RequestEndReason::Failed);
        Expect(error.tokens.empty() && !error.terminal.error.empty() &&
                   error.terminal.errorCode != ErrorCode::RuntimeFault,
               "Ordinary exception lost its error or became a runtime fault");
        if (failure == Failure::InvalidArgument) {
            Expect(error.terminal.errorCode == ErrorCode::InvalidArgument, "Typed request error was not preserved");
        }
        ExpectEnd(Drain(peer), RequestStatus::Completed, RequestEndReason::Length);
        const auto state = scheduler.Snapshot();
        Expect(state.accepting && !state.faulted && !bad.session->Busy() && !good.session->Busy(),
               "Ordinary request failure poisoned scheduler or leaked session leases");
        scheduler.Shutdown();
        ExpectNoReservations(scheduler);
    }
}

void TestUtf8ErrorsPreserveTypedProgress() {
    struct ErrorCase { std::string input; std::string expected; };
    const std::string twoByte = u8"é";
    const std::string threeByte = u8"中";
    const std::string fourByte = "\xf0\x9f\x98\x80";
    const std::vector<ErrorCase> cases{
        {std::string(1023, 'x') + twoByte + " tail", std::string(1023, 'x')},
        {std::string(1022, 'x') + threeByte + " tail", std::string(1022, 'x')},
        {std::string(1021, 'x') + fourByte + " tail", std::string(1021, 'x')},
        {std::string(1021, 'x') + threeByte + " tail", std::string(1021, 'x') + threeByte},
        {std::string(1022, 'x') + twoByte, std::string(1022, 'x') + twoByte},
    };
    Scheduler scheduler(Limits());
    for (const auto& errorCase : cases) {
        auto direct = MakeFixture(1);
        direct.state->failure = Failure::InvalidArgument;
        direct.state->failureMessage = errorCase.input;
        auto request = RequestState::Create(direct.session, {1, 2}, Options(2));
        const auto step = request->Step();
        const auto progress = request->Progress();
        const auto snapshot = request->Snapshot();
        Expect(step.status == RequestStatus::Failed && step.becameTerminal &&
                   progress.status == RequestStatus::Failed &&
                   progress.errorCode == ErrorCode::InvalidArgument &&
                   snapshot.errorCode == progress.errorCode &&
                   progress.ErrorMessage() == errorCase.expected && snapshot.error == errorCase.expected &&
                   progress.errorSize == errorCase.expected.size() && progress.errorSize <= 1024,
               "Request error truncation split UTF-8 or lost typed Progress/Snapshot error metadata");

        auto scheduled = MakeFixture(2);
        scheduled.state->failure = Failure::InvalidArgument;
        scheduled.state->failureMessage = errorCase.input;
        const auto result = Drain(scheduler.Submit(scheduled.session, {1, 2}, Options(2)));
        ExpectEnd(result, RequestStatus::Failed, RequestEndReason::Failed);
        Expect(result.terminal.errorCode == progress.errorCode &&
                   result.terminal.error == errorCase.expected && result.tokens.empty(),
               "Scheduler terminal changed the typed error or its complete UTF-8 prefix");
    }
    Expect(scheduler.Snapshot().accepting && !scheduler.Snapshot().faulted,
           "Bounded ordinary errors faulted the scheduler");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestRuntimeFaultStopsAllRequests() {
    Scheduler scheduler(Limits());
    ForwardGate initial;
    auto peer = MakeFixture(1);
    auto faulty = MakeFixture(2);
    auto pending = MakeFixture(3);
    peer.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    faulty.state->failure = Failure::RuntimeFault;
    auto streamPeer = scheduler.Submit(peer.session, {1}, Options(8));
    initial.AwaitEntry();
    auto streamFault = scheduler.Submit(faulty.session, {2}, Options(2));
    auto streamPending = scheduler.Submit(pending.session, {3}, Options(1));
    initial.Release();
    const auto fault = Drain(streamFault);
    const auto affected = Drain(streamPeer);
    const auto queued = Drain(streamPending);
    for (const auto* result : {&fault, &affected, &queued}) {
        ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
        Expect(result->terminal.errorCode == ErrorCode::RuntimeFault && !result->terminal.error.empty(),
               "Runtime fault did not propagate its typed error to all affected requests");
    }
    Expect(!affected.tokens.empty() && queued.tokens.empty() && pending.state->Inputs().empty(),
           "Fault discarded published tokens or submitted work for a pending request");
    const auto stopped = scheduler.Snapshot();
    Expect(stopped.faulted && !stopped.accepting, "Faulted scheduler still accepts work");
    auto rejected = MakeFixture(4);
    ExpectError(ErrorCode::RuntimeFault, [&] { scheduler.Submit(rejected.session, {4}, Options(1)); });
    Expect(!rejected.session->Busy() && !peer.session->Busy() && !faulty.session->Busy() && !pending.session->Busy(),
           "Runtime fault or rejected Submit retained a session lease");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestSubmitSetupFaultStopsScheduler() {
    for (bool invalidLength : {false, true}) {
        auto limits = Limits();
        limits.maxActiveRequests = 1;
        Scheduler scheduler(limits);
        ForwardGate inFlight;
        auto active = MakeFixture(1);
        auto pending = MakeFixture(2);
        auto faulty = MakeFixture(3);
        active.state->duringForward = [hold = inFlight.Action()](size_t call, bool) {
            if (call == 1) hold();
        };
        auto running = scheduler.Submit(active.session, {1, 2}, Options(4));
        inFlight.AwaitEntry();
        auto queued = scheduler.Submit(pending.session, {3}, Options(1));
        const auto before = scheduler.Snapshot();
        // Change only this idle fake after Session construction, so the error
        // is raised during Request creation inside Scheduler::Submit.
        faulty.state->invalidContextLength = invalidLength;
        faulty.state->contextFailure = invalidLength ? Failure::None : Failure::RuntimeFault;
        ExpectError(ErrorCode::RuntimeFault, [&] { scheduler.Submit(faulty.session, {4}, Options(1)); });
        const auto faulted = scheduler.Snapshot();
        Expect(faulted.faulted && !faulted.accepting &&
                   faulted.outstandingRequests == before.outstandingRequests &&
                   faulted.reservedTokens == before.reservedTokens &&
                   faulted.reservedSessionCacheBytes == before.reservedSessionCacheBytes &&
                   !faulty.session->Busy() && faulty.state->Inputs().empty(),
               "Submit setup fault did not stop admission and roll back only its own reservation");
        Expect(active.session->Busy(), "Setup fault released an unfinished peer's session lease");
        inFlight.Release();
        const auto activeResult = Drain(running);
        const auto pendingResult = Drain(queued);
        for (const auto* result : {&activeResult, &pendingResult}) {
            ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
            Expect(result->tokens.empty() && result->terminal.errorCode == ErrorCode::RuntimeFault,
                   "Setup fault failed to suppress in-flight output or propagate the typed error");
        }
        Expect(pending.state->Inputs().empty() && active.state->Inputs().size() == 1 &&
                   !active.session->Busy() && !pending.session->Busy(),
               "Setup fault submitted pending work or failed to release peers at the safe boundary");
        scheduler.Shutdown();
        ExpectNoReservations(scheduler);
    }

    Scheduler scheduler(Limits());
    auto invalid = MakeFixture();
    invalid.state->contextFailure = Failure::InvalidArgument;
    ExpectError(ErrorCode::InvalidArgument, [&] { scheduler.Submit(invalid.session, {1}, Options(1)); });
    Expect(scheduler.Snapshot().accepting && !scheduler.Snapshot().faulted && !invalid.session->Busy(),
           "Ordinary Submit setup error faulted the scheduler or retained the session lease");
    ExpectNoReservations(scheduler);
    invalid.state->contextFailure = Failure::None;
    auto manual = RequestState::Create(invalid.session, {1}, Options(1));
    ExpectError(ErrorCode::Busy, [&] { scheduler.Submit(invalid.session, {2}, Options(1)); });
    Expect(scheduler.Snapshot().accepting && !scheduler.Snapshot().faulted,
           "Busy Submit setup error faulted the scheduler");
    ExpectNoReservations(scheduler);
    manual.reset();
    ExpectEnd(Drain(scheduler.Submit(invalid.session, {2}, Options(1))),
              RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDroppedRunningHandleRetainsSessionUntilSafe() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    Scheduler scheduler(limits);
    ForwardGate inFlight;
    ForwardGate nextRequest;
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    std::weak_ptr<SessionState> weakSession = a.session;
    a.state->duringForward = [hold = inFlight.Action()](size_t call, bool) { if (call == 1) hold(); };
    b.state->duringForward = [hold = nextRequest.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto dropped = scheduler.Submit(a.session, {1, 2}, Options(4));
    inFlight.AwaitEntry();
    auto survivor = scheduler.Submit(b.session, {3}, Options(1));
    a.session.reset();
    dropped.reset();
    Expect(!weakSession.expired() && scheduler.Snapshot().outstandingRequests == 2,
           "Dropping the handle reclaimed an executing request or its reservation early");
    inFlight.Release();
    nextRequest.AwaitEntry();
    Expect(weakSession.expired() && scheduler.Snapshot().outstandingRequests == 1 &&
               a.state->Inputs().size() == 1 && a.state->Context() == std::vector<int>({1, 2}),
           "Abandoned request did not release ownership/admission at the safe boundary");
    nextRequest.Release();
    ExpectEnd(Drain(survivor), RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestLifecycleFaultPreservesCompletedStreamsAndFirstCause() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    Scheduler scheduler(limits);
    ForwardGate decoding;
    auto completedSession = MakeFixture(1);
    auto active = MakeFixture(2);
    auto pending = MakeFixture(3);
    completedSession.state->samples = {11};
    active.state->samples = {21, 22};
    active.state->duringForward = [hold = decoding.Action()](size_t call, bool) {
        if (call == 2) hold();
    };
    // With one active slot this terminal is already published, but unread,
    // when the following session reaches its second forward.
    auto completed = scheduler.Submit(completedSession.session, {1}, Options(1));
    auto running = scheduler.Submit(active.session, {2}, Options(6));
    decoding.AwaitEntry();
    auto queued = scheduler.Submit(pending.session, {3}, Options(1));
    Expect(completed->Snapshot().status == RequestStatus::Completed &&
               !completed->Snapshot().terminalConsumed && running->Snapshot().bufferedTokens == 1,
           "Lifecycle fault fixture did not retain a completed stream and an active token");
    const auto before = scheduler.Snapshot();
    const std::string firstCause = "model.load: synchronous SYCL upload failure";
    std::string borrowedCause = firstCause;
    scheduler.ReportRuntimeFault(borrowedCause);
    borrowedCause.assign("caller reused its error buffer");
    scheduler.ReportRuntimeFault("later device drain failure");
    const auto faulted = scheduler.Snapshot();
    Expect(faulted.faulted && !faulted.accepting && faulted.faultMessage == firstCause &&
               faulted.outstandingRequests == before.outstandingRequests &&
               faulted.reservedTokens == before.reservedTokens &&
               faulted.reservedSessionCacheBytes == before.reservedSessionCacheBytes &&
               active.session->Busy() && pending.session->Busy(),
           "Lifecycle fault lost its first diagnosis, blocked on Forward, or released active ownership early");
    auto rejected = MakeFixture(4);
    ExpectError(ErrorCode::RuntimeFault, [&] { scheduler.Submit(rejected.session, {4}, Options(1)); });
    Expect(!rejected.session->Busy() && rejected.state->Inputs().empty(),
           "Lifecycle fault admitted a new request");

    decoding.Release();
    const auto completedResult = Drain(completed);
    const auto activeResult = Drain(running);
    const auto pendingResult = Drain(queued);
    ExpectEnd(completedResult, RequestStatus::Completed, RequestEndReason::Length);
    Expect(completedResult.tokens == std::vector<int>({11}) &&
               !completedResult.terminal.errorCode && completedResult.terminal.error.empty(),
           "Lifecycle fault rewrote a previously completed unread stream");
    for (const auto* result : {&activeResult, &pendingResult}) {
        ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
        Expect(result->terminal.errorCode == ErrorCode::RuntimeFault &&
                   result->terminal.error == firstCause,
               "Lifecycle broadcast did not preserve the original error on every unfinished request");
    }
    Expect(activeResult.tokens == std::vector<int>({21}) && pendingResult.tokens.empty() &&
               active.state->Inputs().size() == 2 && pending.state->Inputs().empty() &&
               !active.session->Busy() && !pending.session->Busy(),
           "Lifecycle fault lost earlier output, published an in-flight token, or executed pending work");
    scheduler.Shutdown();
    scheduler.ReportRuntimeFault("report after shutdown");
    Expect(scheduler.Snapshot().faultMessage == firstCause, "Repeated report replaced the first diagnosis");
    ExpectNoReservations(scheduler);
}

void TestLifecycleFaultWhileSubmitIsInitializing() {
    Scheduler scheduler(Limits());
    auto fixture = MakeFixture();
    // Release the gate before waiting for this future during exception unwind.
    std::future<std::shared_ptr<ScheduledRequest>> submitting;
    ForwardGate contextQuery;
    fixture.state->duringContextLength = contextQuery.Action();
    submitting = std::async(std::launch::async, [&] {
        return scheduler.Submit(fixture.session, {1}, Options(1));
    });
    contextQuery.AwaitEntry();
    const auto reserved = scheduler.Snapshot();
    Expect(reserved.outstandingRequests == 1 && reserved.queuedRequests == 0 &&
               reserved.activeRequests == 0 && fixture.session->Busy(),
           "Submit did not reach its reserved, unpublished state");
    const std::string diagnosis = "session.create: device initialization failed";
    scheduler.ReportRuntimeFault(diagnosis);
    const auto faulted = scheduler.Snapshot();
    Expect(faulted.faulted && !faulted.accepting && faulted.faultMessage == diagnosis &&
               faulted.outstandingRequests == 1 && fixture.session->Busy() &&
               fixture.state->Inputs().empty(),
           "Lifecycle report waited for initializing Submit or reclaimed its ownership before rollback");
    contextQuery.Release();
    Expect(submitting.wait_for(5s) == std::future_status::ready,
           "Initializing Submit did not finish after a lifecycle fault");
    ExpectError(ErrorCode::RuntimeFault, [&] { submitting.get(); });
    Expect(!fixture.session->Busy() && fixture.state->Inputs().empty(),
           "Submit raced past the lifecycle fault or retained its unpublished session lease");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestReentrantLifecycleFaultBeforeTerminalPublication() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    Scheduler scheduler(limits);
    ForwardGate finalizing;
    auto active = MakeFixture(1);
    auto pending = MakeFixture(2);
    active.state->samples = {31};
    std::shared_ptr<ScheduledRequest> running;
    struct Probe {
        Scheduler* scheduler;
        std::shared_ptr<ScheduledRequest>* stream;
        SessionState* session;
        std::atomic<int> calls{0};
        std::atomic<bool> terminalNotPublished{false};
        std::atomic<bool> finalStepAlreadyEnded{false};
        std::atomic<bool> reportReturned{false};
    } probe{&scheduler, &running, active.session.get()};
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restore;
    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* text, void* userData) {
        if (std::strcmp(text, "SchedulerTests lifecycle fault before terminal") != 0) return;
        auto& probe = *static_cast<Probe*>(userData);
        ++probe.calls;
        // Step has already constructed Completed and released its lease.
        // Its deferred logs run before Scheduler publishes the terminal.
        probe.finalStepAlreadyEnded.store(!probe.session->Busy());
        probe.terminalNotPublished.store((*probe.stream)->Snapshot().status == RequestStatus::Finalizing);
        probe.scheduler->ReportRuntimeFault("session.create: oneDNN initialization failed");
        probe.reportReturned.store(true);
    }, &probe);
    active.state->duringForward = [hold = finalizing.Action()](size_t call, bool) {
        if (call == 2) {
            hold();
            AILA_LOG_INFO("SchedulerTests lifecycle fault before terminal");
        }
    };
    running = scheduler.Submit(active.session, {1}, Options(1));
    finalizing.AwaitEntry();
    auto queued = scheduler.Submit(pending.session, {2}, Options(1));
    finalizing.Release();
    const auto activeResult = Drain(running);
    const auto pendingResult = Drain(queued);
    Expect(probe.calls.load() == 1 && probe.finalStepAlreadyEnded.load() &&
               probe.terminalNotPublished.load() && probe.reportReturned.load(),
           "Lifecycle fault did not occur after Step completion and before terminal publication");
    for (const auto* result : {&activeResult, &pendingResult}) {
        ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
        Expect(result->terminal.errorCode == ErrorCode::RuntimeFault &&
                   result->terminal.error == "session.create: oneDNN initialization failed",
               "Reentrant lifecycle fault failed to retire a still-unpublished terminal");
    }
    Expect(activeResult.tokens == std::vector<int>({31}) && pendingResult.tokens.empty() &&
               pending.state->Inputs().empty() && !active.session->Busy() && !pending.session->Busy(),
           "Reentrant fault discarded buffered output, ran queued work, or retained a session");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestHandlesSurviveSchedulerDestruction() {
    std::shared_ptr<ScheduledRequest> survivor;
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    {
        auto limits = Limits();
        limits.maxActiveRequests = 1;
        Scheduler scheduler(limits);
        ForwardGate held;
        b.state->duringForward = [hold = held.Action()](size_t call, bool) { if (call == 1) hold(); };
        survivor = scheduler.Submit(a.session, {1}, Options(1));
        auto other = scheduler.Submit(b.session, {2}, Options(1));
        held.AwaitEntry();
        Expect(survivor->Snapshot().status == RequestStatus::Completed,
               "Surviving stream was not completed before scheduler destruction");
        other->Cancel();
        held.Release();
        scheduler.Shutdown();
        ExpectEnd(Drain(other), RequestStatus::Cancelled, RequestEndReason::Cancelled);
        Expect(scheduler.Snapshot().outstandingRequests == 1,
               "Shutdown discarded the completed unread stream reservation");
    }
    const auto result = Drain(survivor);
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::Length);
    Expect(result.tokens.size() == 1 && !a.session->Busy() && !b.session->Busy(),
           "Scheduler destruction invalidated a surviving readable stream");
    survivor.reset();
}

void TestReentrantAndConcurrentShutdownWakesReaders() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    Scheduler scheduler(limits);
    ForwardGate beforeLog;
    ForwardGate insideCallback;
    auto a = MakeFixture(1);
    auto b = MakeFixture(2);
    struct Probe {
        Scheduler* scheduler;
        std::function<void()> hold;
        std::atomic<int> calls{0};
        std::atomic<bool> shutdownReturned{false};
    } probe{&scheduler, insideCallback.Action()};
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restore;
    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* text, void* userData) {
        if (std::strcmp(text, "SchedulerTests reentrant shutdown") != 0) return;
        auto& probe = *static_cast<Probe*>(userData);
        ++probe.calls;
        probe.scheduler->Shutdown();
        probe.shutdownReturned.store(true);
        probe.hold();
    }, &probe);
    a.state->duringForward = [hold = beforeLog.Action()](size_t call, bool) {
        if (call == 1) {
            hold();
            AILA_LOG_INFO("SchedulerTests reentrant shutdown");
        }
    };
    auto active = scheduler.Submit(a.session, {1, 2}, Options(6));
    beforeLog.AwaitEntry();
    auto pending = scheduler.Submit(b.session, {3}, Options(1));
    auto activeReader = std::async(std::launch::async, [active] { return Drain(active); });
    auto pendingReader = std::async(std::launch::async, [pending] { return Drain(pending); });
    beforeLog.Release();
    insideCallback.AwaitEntry();
    const auto stopping = scheduler.Snapshot();
    Expect(probe.shutdownReturned.load() && stopping.stopping && !stopping.accepting,
           "Worker-thread Shutdown self-joined or failed to stop admission");
    auto rejected = MakeFixture(3);
    ExpectError(ErrorCode::SchedulerStopped, [&] { scheduler.Submit(rejected.session, {4}, Options(1)); });
    auto firstJoin = std::async(std::launch::async, [&] { scheduler.Shutdown(); });
    auto secondJoin = std::async(std::launch::async, [&] { scheduler.Shutdown(); });
    // Releasing this gate lets the current Step unwind and all joiners/readers finish.
    insideCallback.Release();
    Expect(firstJoin.wait_for(5s) == std::future_status::ready &&
               secondJoin.wait_for(5s) == std::future_status::ready,
           "Concurrent Shutdown callers did not both finish");
    firstJoin.get();
    secondJoin.get();
    Expect(activeReader.wait_for(5s) == std::future_status::ready &&
               pendingReader.wait_for(5s) == std::future_status::ready,
           "Shutdown did not wake both active and pending readers");
    ExpectEnd(activeReader.get(), RequestStatus::Cancelled, RequestEndReason::SchedulerStopped);
    ExpectEnd(pendingReader.get(), RequestStatus::Cancelled, RequestEndReason::SchedulerStopped);
    Expect(probe.calls.load() == 1 && b.state->Inputs().empty() && !a.session->Busy() && !b.session->Busy(),
           "Shutdown repeated callbacks, submitted pending work, or leaked session leases");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDeferredCallbackFaultSuppressesPreparedToken() {
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    Scheduler scheduler(limits);
    ForwardGate beforeLog;
    auto active = MakeFixture(1);
    auto pending = MakeFixture(2);
    auto faulty = MakeFixture(3);
    faulty.state->contextFailure = Failure::RuntimeFault;
    active.state->samples = {11};
    std::shared_ptr<ScheduledRequest> running;
    struct Probe {
        Scheduler* scheduler;
        std::shared_ptr<SessionState> faultySession;
        FakeState* activeState;
        std::shared_ptr<ScheduledRequest>* activeStream;
        std::atomic<int> calls{0};
        std::atomic<bool> sampleAlreadyComputed{false};
        std::atomic<bool> tokenNotYetPublished{false};
        std::atomic<bool> submitFaulted{false};
        std::atomic<bool> unexpectedException{false};
    } probe{&scheduler, faulty.session, active.state.get(), &running};
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restore;
    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* message, void* userData) {
        if (std::strcmp(message, "SchedulerTests fault after prepared Step") != 0) return;
        auto& probe = *static_cast<Probe*>(userData);
        ++probe.calls;
        // Step flushes deferred logs after constructing its Decode result, but
        // Scheduler has not yet published that result's token to this stream.
        probe.sampleAlreadyComputed.store(probe.activeState->Histories().size() == 1);
        probe.tokenNotYetPublished.store((*probe.activeStream)->Snapshot().generatedTokens == 0);
        try {
            probe.scheduler->Submit(probe.faultySession, {4}, Options(1));
            probe.unexpectedException.store(true);
        } catch (const Error& error) {
            probe.submitFaulted.store(error.Code() == ErrorCode::RuntimeFault);
            probe.unexpectedException.store(error.Code() != ErrorCode::RuntimeFault);
        } catch (...) {
            probe.unexpectedException.store(true);
        }
    }, &probe);
    active.state->duringForward = [hold = beforeLog.Action()](size_t call, bool) {
        if (call == 1) {
            hold();
            AILA_LOG_INFO("SchedulerTests fault after prepared Step");
        }
    };
    running = scheduler.Submit(active.session, {1, 2}, Options(4));
    beforeLog.AwaitEntry();
    auto queued = scheduler.Submit(pending.session, {3}, Options(1));
    beforeLog.Release();
    const auto activeResult = Drain(running);
    const auto pendingResult = Drain(queued);
    // The logger swallows callback exceptions; assert all recorded observations here.
    Expect(probe.calls.load() == 1 && probe.sampleAlreadyComputed.load() &&
               probe.tokenNotYetPublished.load() && probe.submitFaulted.load() &&
               !probe.unexpectedException.load(),
           "Deferred callback did not trigger the intended fault after sampling and before publication");
    for (const auto* result : {&activeResult, &pendingResult}) {
        ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
        Expect(result->tokens.empty() && result->terminal.generatedTokens == 0 &&
                   result->terminal.errorCode == ErrorCode::RuntimeFault,
               "Reentrant setup fault published a previously prepared token or lost the peer fault");
    }
    Expect(active.state->Context() == std::vector<int>({1, 2}) && active.state->Inputs().size() == 1 &&
               pending.state->Inputs().empty() && faulty.state->Inputs().empty() &&
               !active.session->Busy() && !pending.session->Busy() && !faulty.session->Busy(),
           "Deferred setup fault submitted pending work or retained session leases");
    Expect(scheduler.Snapshot().faulted && !scheduler.Snapshot().accepting,
           "Deferred setup fault left scheduler admission open");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestLogCallbackDestroysSchedulerOwner() {
    struct Probe {
        std::unique_ptr<Scheduler> owner;
        const char* marker = "SchedulerTests destroy owner from worker";
        std::mutex mutex;
        std::condition_variable changed;
        bool ownerDestroyed = false;
        int calls = 0;
        std::thread::id callbackThread;
    };
    auto probe = std::make_shared<Probe>();
    auto limits = Limits();
    limits.maxActiveRequests = 1;
    probe->owner = std::make_unique<Scheduler>(limits);
    ForwardGate beforeLog;
    auto active = MakeFixture(1);
    auto pending = MakeFixture(2);
    std::weak_ptr<SessionState> weakActive = active.session;
    std::weak_ptr<SessionState> weakPending = pending.session;
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restore;
    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* message, void* userData) {
        auto& probe = *static_cast<Probe*>(userData);
        if (std::strcmp(message, probe.marker) != 0) return;
        // Main has finished all accesses to owner before releasing beforeLog.
        // This destroys Scheduler itself, rather than merely calling Shutdown.
        probe.owner.reset();
        {
            std::lock_guard<std::mutex> lock(probe.mutex);
            ++probe.calls;
            probe.callbackThread = std::this_thread::get_id();
            probe.ownerDestroyed = true;
        }
        probe.changed.notify_all();
    }, probe.get());
    // Retain callback data until the executing backend is done even if a test
    // assertion unwinds. reset() in the callback breaks the owner cycle.
    active.state->duringForward = [hold = beforeLog.Action(), lifetime = probe](size_t call, bool) {
        if (call == 1) {
            hold();
            AILA_LOG_INFO("%s", lifetime->marker);
        }
    };
    auto running = probe->owner->Submit(active.session, {1, 2}, Options(4));
    beforeLog.AwaitEntry();
    auto queued = probe->owner->Submit(pending.session, {3}, Options(1));
    active.session.reset();
    pending.session.reset();
    Expect(!weakActive.expired() && !weakPending.expired(), "Requests did not retain their sessions");
    // No subsequent access to unique_ptr owner occurs on this thread.
    beforeLog.Release();
    {
        std::unique_lock<std::mutex> lock(probe->mutex);
        Expect(probe->changed.wait_for(lock, 5s, [&] { return probe->ownerDestroyed; }),
               "Destroying Scheduler in its own worker callback did not return");
        Expect(probe->calls == 1 && probe->callbackThread != std::this_thread::get_id(),
               "Scheduler was not destroyed exactly once on its submission worker");
    }
    ExpectEnd(Drain(running), RequestStatus::Cancelled, RequestEndReason::SchedulerStopped);
    ExpectEnd(Drain(queued), RequestStatus::Cancelled, RequestEndReason::SchedulerStopped);
    Expect(weakActive.expired() && weakPending.expired() && active.state->Inputs().size() == 1 &&
               pending.state->Inputs().empty(),
           "Worker owner destruction left dangling sessions or executed pending work");
}

void TestShutdownWhileSubmitIsInitializing() {
    Scheduler scheduler(Limits());
    ForwardGate contextQuery;
    auto fixture = MakeFixture();
    // Install after Session construction. Request::Create queries context after
    // Scheduler has reserved admission and the session has acquired its lease.
    fixture.state->duringContextLength = contextQuery.Action();
    auto submitting = std::async(std::launch::async, [&scheduler, session = fixture.session] {
        return scheduler.Submit(session, {1, 2}, Options(4));
    });
    contextQuery.AwaitEntry();
    const auto reserved = scheduler.Snapshot();
    Expect(reserved.outstandingRequests == 1 && reserved.reservedTokens == 6 &&
               reserved.reservedSessionCacheBytes == fixture.session->CacheBytes() &&
               reserved.queuedRequests == 0 && reserved.activeRequests == 0 && fixture.session->Busy(),
           "Context gate did not pause Submit after reservation and before enqueue");
    // Shutdown runs on another thread than Submit and must finish while its
    // context query remains gated. The unqueued Submit owns its reservation.
    scheduler.Shutdown();
    const auto stopped = scheduler.Snapshot();
    Expect(stopped.stopping && !stopped.accepting && stopped.outstandingRequests == 1 &&
               stopped.queuedRequests == 0 && stopped.activeRequests == 0 && fixture.session->Busy() &&
               fixture.state->Inputs().empty(),
           "Shutdown enqueued the initializing request or reclaimed its reservation before rollback");
    contextQuery.Release();
    Expect(submitting.wait_for(5s) == std::future_status::ready, "Initializing Submit did not finish after shutdown");
    ExpectError(ErrorCode::SchedulerStopped, [&] { submitting.get(); });
    ExpectNoReservations(scheduler);
    Expect(!fixture.session->Busy() && fixture.state->Inputs().empty() && fixture.state->Context().empty(),
           "Submit after shutdown did not roll back its session lease without Forward");
    fixture.session->Reset();
}


void TestDecodeBatchConfigurationAndFallback() {
    for (const size_t size : {size_t{0}, size_t{3}, size_t{5}}) {
        auto limits = Limits();
        limits.maxDecodeBatchSize = size;
        ExpectError(ErrorCode::InvalidArgument, [&] { Scheduler rejected(limits); });
    }
    auto large = Limits();
    large.maxActiveRequests = 5;
    large.maxDecodeBatchSize = 5;
    ExpectError(ErrorCode::InvalidArgument, [&] { Scheduler rejected(large); });
    auto limits = Limits();
    limits.maxDecodeBatchSize = 2;
    Scheduler scheduler(limits);
    ForwardGate initial;
    auto first = MakeFixture(1), second = MakeFixture(2), reference = MakeFixture();
    const auto expected = RunDirect(reference, {2}, Options(4));
    first.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto a = scheduler.Submit(first.session, {1}, Options(4));
    initial.AwaitEntry();
    auto b = scheduler.Submit(second.session, {2}, Options(4));
    initial.Release();
    ExpectEnd(Drain(a), RequestStatus::Completed, RequestEndReason::Length);
    const auto result = Drain(b);
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::Length);
    Expect(result.tokens == expected && second.state->Context() == reference.state->Context(),
           "Unsupported batch backend did not fall back to independent scalar execution");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDecodeBatchDynamicMembershipAndPrefillFairness() {
    auto limits = Limits();
    limits.maxActiveRequests = 4;
    limits.maxDecodeBatchSize = 4;
    Scheduler scheduler(limits);
    ForwardGate initial, grouped;
    auto trace = std::make_shared<ExecutionTrace>();
    auto model = std::make_shared<FakeBatch>();
    auto a = MakeFixture(1, trace, 128, 64, model);
    auto b = MakeFixture(2, trace, 128, 64, model);
    auto c = MakeFixture(3, trace, 128, 64, model);
    auto prefill = MakeFixture(4, trace);
    auto replacement = MakeFixture(5, trace, 128, 64, model);
    std::atomic<bool> prefillAdvancedBeforeSecondBatch{false};
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    model->duringBatch = [hold = grouped.Action(), prefillState = prefill.state,
                          &prefillAdvancedBeforeSecondBatch](size_t call) {
        if (call == 1) hold();
        if (call == 2) prefillAdvancedBeforeSecondBatch.store(prefillState->Inputs().size() >= 2);
    };
    const auto optionsA = Options(5, 2, 110), optionsB = Options(4, 2, 220), optionsE = Options(3, 2, 330);
    auto refA = MakeFixture(), refB = MakeFixture(), refE = MakeFixture();
    const auto expectedA = RunDirect(refA, {1}, optionsA);
    const auto expectedB = RunDirect(refB, {2}, optionsB);
    const auto expectedE = RunDirect(refE, {5}, optionsE);
    auto streamA = scheduler.Submit(a.session, {1}, optionsA);
    initial.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, {2}, optionsB);
    auto streamC = scheduler.Submit(c.session, {3}, Options(8));
    auto streamPrefill = scheduler.Submit(prefill.session, {4, 4, 4, 4, 4, 4, 4, 4}, Options(1));
    initial.Release();
    grouped.AwaitEntry();
    Expect(scheduler.Snapshot().activeRequests == 4, "Selected batch rows lost active-slot accounting");
    auto streamE = scheduler.Submit(replacement.session, {5}, optionsE);
    Expect(scheduler.Snapshot().queuedRequests == 1 && replacement.state->Inputs().empty(),
           "A full active set executed a newly submitted replacement early");
    streamC->Cancel();
    Expect(c.session->Busy(), "Cancellation released an in-flight batch row before its boundary");
    grouped.Release();
    const auto resultA = Drain(streamA), resultB = Drain(streamB), resultC = Drain(streamC);
    const auto resultPrefill = Drain(streamPrefill), resultE = Drain(streamE);
    for (const auto* result : {&resultA, &resultB, &resultPrefill, &resultE}) {
        ExpectEnd(*result, RequestStatus::Completed, RequestEndReason::Length);
    }
    ExpectEnd(resultC, RequestStatus::Cancelled, RequestEndReason::Cancelled);
    Expect(resultA.tokens == expectedA && resultB.tokens == expectedB && resultE.tokens == expectedE &&
               a.state->Context() == refA.state->Context() && b.state->Context() == refB.state->Context() &&
               replacement.state->Context() == refE.state->Context(),
           "Removing/adding a batch row changed an independent request sequence or cache");
    const auto groups = model->Groups();
    Expect(groups.size() >= 2 && groups[0] == std::vector<int>({1, 2, 3}) &&
               groups[1] == std::vector<int>({1, 2, 5}) && prefillAdvancedBeforeSecondBatch.load(),
           "Batch membership was not rebuilt or decode overtook the oldest prefill repeatedly");
    Expect(resultC.tokens.size() == 1 && c.state->Context().size() == 2 && c.state->Inputs().size() == 2 &&
               !c.session->Busy() && trace->MaximumInFlight() == 1,
           "Batch cancellation failed to drain exactly its selected input on one submission thread");
    const auto entries = trace->Entries();
    const auto worker = entries.front().thread;
    Expect(worker != std::this_thread::get_id() &&
               std::all_of(entries.begin(), entries.end(), [worker](const auto& entry) { return entry.thread == worker; }),
           "Continuous batching introduced a second submission thread");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDecodeBatchIdentityAndWidthBound() {
    auto limits = Limits();
    limits.maxActiveRequests = 4;
    limits.maxDecodeBatchSize = 2;
    Scheduler scheduler(limits);
    ForwardGate initial;
    auto trace = std::make_shared<ExecutionTrace>();
    auto firstModel = std::make_shared<FakeBatch>(), secondModel = std::make_shared<FakeBatch>();
    auto a = MakeFixture(1, trace, 128, 64, firstModel), b = MakeFixture(2, trace, 128, 64, secondModel);
    auto c = MakeFixture(3, trace, 128, 64, firstModel), d = MakeFixture(4, trace, 128, 64, secondModel);
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto streamA = scheduler.Submit(a.session, {1}, Options(4));
    initial.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, {2}, Options(4));
    auto streamC = scheduler.Submit(c.session, {3}, Options(4));
    auto streamD = scheduler.Submit(d.session, {4}, Options(4));
    initial.Release();
    for (const auto& stream : {streamA, streamB, streamC, streamD}) {
        ExpectEnd(Drain(stream), RequestStatus::Completed, RequestEndReason::Length);
    }
    const auto one = firstModel->Groups(), two = secondModel->Groups();
    Expect(!one.empty() && !two.empty(), "Compatible rows from two models never reached their own batch entries");
    for (const auto& group : one) {
        Expect(group.size() == 2 && (group == std::vector<int>({1, 3}) || group == std::vector<int>({3, 1})),
               "First batch exceeded its cap or mixed model identity");
    }
    for (const auto& group : two) {
        Expect(group.size() == 2 && (group == std::vector<int>({2, 4}) || group == std::vector<int>({4, 2})),
               "Second batch exceeded its cap or mixed model identity");
    }
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDecodeBatchBackpressureAndFullRingFinalization() {
    auto limits = Limits();
    limits.maxActiveRequests = 3;
    limits.maxDecodeBatchSize = 3;
    limits.maxBufferedTokensPerRequest = 1;
    Scheduler scheduler(limits);
    ForwardGate initial, grouped;
    auto model = std::make_shared<FakeBatch>();
    auto trace = std::make_shared<ExecutionTrace>();
    auto slow = MakeFixture(1, trace, 128, 64, model);
    auto a = MakeFixture(2, trace, 128, 64, model), b = MakeFixture(3, trace, 128, 64, model);
    slow.state->samples = {11, 12};
    a.state->samples = {21};
    b.state->samples = {31};
    slow.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    model->duringBatch = [hold = grouped.Action()](size_t call) { if (call == 1) hold(); };
    auto blocked = scheduler.Submit(slow.session, {1}, Options(4));
    initial.AwaitEntry();
    auto streamA = scheduler.Submit(a.session, {2}, Options(1));
    auto streamB = scheduler.Submit(b.session, {3}, Options(1));
    initial.Release();
    grouped.AwaitEntry();
    Expect(streamA->Snapshot().bufferedTokens == 1 && streamB->Snapshot().bufferedTokens == 1 &&
               streamA->Snapshot().status == RequestStatus::Finalizing &&
               streamB->Snapshot().status == RequestStatus::Finalizing,
           "Batch test did not retain full rings on both finalizing rows");
    grouped.Release();
    const auto resultA = Drain(streamA), resultB = Drain(streamB), resultSlow = Drain(blocked);
    ExpectEnd(resultA, RequestStatus::Completed, RequestEndReason::Length);
    ExpectEnd(resultB, RequestStatus::Completed, RequestEndReason::Length);
    ExpectEnd(resultSlow, RequestStatus::Cancelled, RequestEndReason::OutputBackpressure);
    Expect(resultSlow.tokens == std::vector<int>({11}) && slow.state->Inputs().size() == 1 &&
               resultA.tokens == std::vector<int>({21}) && resultB.tokens == std::vector<int>({31}) &&
               a.state->Context() == std::vector<int>({2, 21}) && b.state->Context() == std::vector<int>({3, 31}) &&
               model->Groups() == std::vector<std::vector<int>>({{2, 3}}),
           "Backpressure submitted an extra row, discarded output, or cancelled grouped finalization");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDecodeBatchSharedErrorScope() {
    for (const auto failure : {Failure::Ordinary, Failure::InvalidArgument, Failure::RuntimeFault}) {
        auto limits = Limits();
        limits.maxActiveRequests = 3;
        limits.maxDecodeBatchSize = 3;
        Scheduler scheduler(limits);
        ForwardGate initial;
        auto trace = std::make_shared<ExecutionTrace>();
        auto model = std::make_shared<FakeBatch>();
        auto a = MakeFixture(1, trace, 128, 64, model), b = MakeFixture(2, trace, 128, 64, model);
        auto independent = MakeFixture(3, trace), pending = MakeFixture(4, trace);
        b.state->failure = failure;
        b.state->failOnCall = 2; // First row has prepared a sample when the second row fails.
        a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
        auto streamA = scheduler.Submit(a.session, {1}, Options(4));
        initial.AwaitEntry();
        auto streamB = scheduler.Submit(b.session, {2}, Options(4));
        auto streamIndependent = scheduler.Submit(independent.session, {3}, Options(3));
        auto streamPending = scheduler.Submit(pending.session, {4}, Options(2));
        initial.Release();
        const auto resultA = Drain(streamA), resultB = Drain(streamB);
        const auto resultIndependent = Drain(streamIndependent), resultPending = Drain(streamPending);
        const auto reason = failure == Failure::RuntimeFault ? RequestEndReason::RuntimeFault : RequestEndReason::Failed;
        ExpectEnd(resultA, RequestStatus::Failed, reason);
        ExpectEnd(resultB, RequestStatus::Failed, reason);
        Expect(resultA.tokens.size() == 1 && resultB.tokens.size() == 1,
               "Shared failure exposed the first row's partially prepared sample");
        if (failure == Failure::RuntimeFault) {
            ExpectEnd(resultIndependent, RequestStatus::Failed, RequestEndReason::RuntimeFault);
            ExpectEnd(resultPending, RequestStatus::Failed, RequestEndReason::RuntimeFault);
            Expect(scheduler.Snapshot().faulted && !scheduler.Snapshot().accepting &&
                       pending.state->Inputs().empty() && resultPending.tokens.empty(),
                   "Batch RuntimeFault did not block admission and pending device work");
            for (const auto* result : {&resultA, &resultB, &resultIndependent, &resultPending}) {
                Expect(result->terminal.errorCode == ErrorCode::RuntimeFault &&
                           result->terminal.error == b.state->failureMessage,
                       "Batch fault lost the device domain's first failure");
            }
        } else {
            ExpectEnd(resultIndependent, RequestStatus::Completed, RequestEndReason::Length);
            ExpectEnd(resultPending, RequestStatus::Completed, RequestEndReason::Length);
            Expect(!scheduler.Snapshot().faulted && scheduler.Snapshot().accepting &&
                       resultIndependent.tokens.size() == 3 && resultPending.tokens.size() == 2,
                   "Ordinary batch error escaped into unrelated model requests");
            const auto code = failure == Failure::Ordinary ? std::optional<ErrorCode>{}
                                                           : std::optional<ErrorCode>{ErrorCode::InvalidArgument};
            Expect(resultA.terminal.errorCode == code && resultB.terminal.errorCode == code,
                   "Shared ordinary batch error lost its typed classification");
        }
        scheduler.Shutdown();
        ExpectNoReservations(scheduler);
    }
}

void TestDecodeBatchReentrantShutdownCancelsAllRunningRows() {
    auto limits = Limits();
    limits.maxActiveRequests = 2;
    limits.maxDecodeBatchSize = 2;
    Scheduler scheduler(limits);
    ForwardGate initial;
    auto trace = std::make_shared<ExecutionTrace>();
    auto model = std::make_shared<FakeBatch>();
    auto a = MakeFixture(1, trace, 128, 64, model), b = MakeFixture(2, trace, 128, 64, model);
    auto pending = MakeFixture(3, trace);
    std::atomic<bool> shutdownReturned{false};
    model->duringBatch = [&](size_t call) {
        if (call == 1) {
            scheduler.Shutdown();
            shutdownReturned.store(true);
        }
    };
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    auto streamA = scheduler.Submit(a.session, {1}, Options(4));
    initial.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, {2}, Options(4));
    auto streamPending = scheduler.Submit(pending.session, {3}, Options(2));
    initial.Release();
    const auto resultA = Drain(streamA), resultB = Drain(streamB), resultPending = Drain(streamPending);
    for (const auto* result : {&resultA, &resultB, &resultPending}) {
        ExpectEnd(*result, RequestStatus::Cancelled, RequestEndReason::SchedulerStopped);
    }
    Expect(shutdownReturned.load() && resultA.tokens.size() == 1 && resultB.tokens.size() == 1 &&
               a.state->Context().size() == 2 && b.state->Context().size() == 2 && pending.state->Inputs().empty(),
           "Reentrant shutdown self-joined, missed a running batch row, or ran pending work");
    scheduler.Shutdown();
    ExpectNoReservations(scheduler);
}

void TestDecodeBatchDeferredFaultSuppressesAllPreparedRows() {
    for (const int newTokens : {1, 3}) {
        auto limits = Limits();
        limits.maxDecodeBatchSize = 2;
        Scheduler scheduler(limits);
        ForwardGate initial;
        auto trace = std::make_shared<ExecutionTrace>();
        auto model = std::make_shared<FakeBatch>();
        auto a = MakeFixture(1, trace, 128, 64, model), b = MakeFixture(2, trace, 128, 64, model);
        a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
        struct Probe {
            Scheduler* scheduler;
            size_t calls = 0;
            bool returned = false;
        } probe{&scheduler};
        struct RestoreLogging {
            aila::LogLevel level = aila::get_log_level();
            ~RestoreLogging() {
                aila::set_log_callback(nullptr);
                aila::set_log_level(level);
            }
        } restore;
        aila::set_log_level(aila::LogLevel::Info);
        aila::set_log_callback([](int, const char* message, void* userData) {
            if (std::strcmp(message, "SchedulerTests deferred batch fault") != 0) return;
            auto& probe = *static_cast<Probe*>(userData);
            ++probe.calls;
            probe.scheduler->ReportRuntimeFault("fake batch lifecycle fault");
            probe.returned = true;
        }, &probe);
        model->afterBatch = [](size_t call) { if (call == 1) AILA_LOG_INFO("SchedulerTests deferred batch fault"); };
        auto streamA = scheduler.Submit(a.session, {1}, Options(newTokens));
        initial.AwaitEntry();
        auto streamB = scheduler.Submit(b.session, {2}, Options(newTokens));
        initial.Release();
        const auto resultA = Drain(streamA), resultB = Drain(streamB);
        for (const auto* result : {&resultA, &resultB}) {
            ExpectEnd(*result, RequestStatus::Failed, RequestEndReason::RuntimeFault);
            Expect(result->tokens.size() == 1 && result->terminal.error == "fake batch lifecycle fault",
                   "Deferred batch fault exposed prepared output or an unpublished successful terminal");
        }
        Expect(probe.calls == 1 && probe.returned && !a.session->Busy() && !b.session->Busy(),
               "Deferred batch lifecycle report did not retire every row safely");
        scheduler.Shutdown();
        ExpectNoReservations(scheduler);
    }
}

void TestDecodeBatchTerminalReleasesActiveSlotBeforeCredit() {
    auto limits = Limits();
    limits.maxDecodeBatchSize = 2;
    limits.maxOutstandingRequests = 2;
    limits.maxOutstandingTokens = 6;
    limits.maxSessionCacheBytes = 128;
    Scheduler scheduler(limits);
    ForwardGate initial, destruction;
    auto trace = std::make_shared<ExecutionTrace>();
    auto model = std::make_shared<FakeBatch>();
    auto a = MakeFixture(1, trace, 128, 64, model), b = MakeFixture(2, trace, 128, 64, model);
    auto replacement = MakeFixture(3, trace);
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    b.state->duringDestruction = destruction.Action();
    auto streamA = scheduler.Submit(a.session, {1}, Options(2));
    initial.AwaitEntry();
    auto streamB = scheduler.Submit(b.session, {2}, Options(2));
    // Leave the request as the last Session owner. Its completion must destroy
    // the backend before publishing B's terminal, after publishing A's terminal.
    b.session.reset();
    initial.Release();
    destruction.AwaitEntry();
    Expect(model->Groups() == std::vector<std::vector<int>>({{1, 2}, {1, 2}}),
           "Accounting test did not reach grouped decode and finalization");
    const auto resultA = Drain(streamA);
    ExpectEnd(resultA, RequestStatus::Completed, RequestEndReason::Length);
    const auto held = scheduler.Snapshot();
    Expect(held.outstandingRequests == 1 && held.queuedRequests == 0 && held.activeRequests == 1 &&
               held.reservedTokens == 3 && held.reservedSessionCacheBytes == 64,
           "Consuming a completed batch row released its credit before its active slot");
    // Reuse all three kinds of returned admission while the same batch still
    // owns B. The queued replacement must not execute during B's destruction.
    auto streamReplacement = scheduler.Submit(replacement.session, {3}, Options(2));
    const auto admitted = scheduler.Snapshot();
    Expect(admitted.outstandingRequests == 2 && admitted.queuedRequests == 1 && admitted.activeRequests == 1 &&
               admitted.reservedTokens == 6 && admitted.reservedSessionCacheBytes == 128 &&
               replacement.state->Inputs().empty(),
           "Replacing a consumed batch row broke admission or execution accounting");
    destruction.Release();
    ExpectEnd(Drain(streamB), RequestStatus::Completed, RequestEndReason::Length);
    ExpectEnd(Drain(streamReplacement), RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    Expect(resultA.tokens.size() == 2 && !b.state->destructionCallbackFailed.load(),
           "Batch accounting regression did not complete its controlled destruction");
    ExpectNoReservations(scheduler);
}

void TestDroppedDecodeBatchRowRetainsSessionUntilSafe() {
    auto limits = Limits();
    limits.maxDecodeBatchSize = 2;
    Scheduler scheduler(limits);
    ForwardGate initial, grouped;
    auto trace = std::make_shared<ExecutionTrace>();
    auto model = std::make_shared<FakeBatch>();
    auto a = MakeFixture(1, trace, 128, 64, model), b = MakeFixture(2, trace, 128, 64, model);
    std::weak_ptr<SessionState> weakA = a.session;
    auto reference = MakeFixture();
    const auto expected = RunDirect(reference, {2}, Options(4));
    a.state->duringForward = [hold = initial.Action()](size_t call, bool) { if (call == 1) hold(); };
    model->duringBatch = [hold = grouped.Action()](size_t call) { if (call == 1) hold(); };
    auto dropped = scheduler.Submit(a.session, {1}, Options(4));
    initial.AwaitEntry();
    auto peer = scheduler.Submit(b.session, {2}, Options(4));
    initial.Release();
    grouped.AwaitEntry();
    a.session.reset();
    dropped.reset();
    Expect(!weakA.expired() && scheduler.Snapshot().outstandingRequests == 2,
           "Dropping an executing batch row released its session or admission too early");
    grouped.Release();
    const auto result = Drain(peer);
    ExpectEnd(result, RequestStatus::Completed, RequestEndReason::Length);
    scheduler.Shutdown();
    Expect(weakA.expired() && result.tokens == expected && b.state->Context() == reference.state->Context(),
           "Dropped batch row leaked ownership or changed its continuing peer");
    ExpectNoReservations(scheduler);
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"configuration, request validation and rejected admission", TestValidationAndAdmissionFailures},
        {"thinking early length, peer isolation and reservation release", TestThinkingBudgetEarlyLengthAndPeer},
        {"Read timeout/closed and finalization with a full ring", TestReadAndFinalizationWithFullRing},
        {"stop token terminal and output ordering", TestStopTokenTerminal},
        {"prefill admission, cancellation, peer progress and recovery", TestPrefillAdmissionCancellationAndPeer},
        {"stop-token admission counts valid repeated IDs", TestStopTokenAdmissionLimit},
        {"fair single-thread scheduling and fixed-seed reference", TestFairRoundRobinAndSeededReference},
        {"active limit, pending cancel and dropped pending handle", TestActiveLimitPendingCancelAndDrop},
        {"cancellation during Forward and session reuse", TestCancellationDuringForward},
        {"backpressure cancels only the slow consumer", TestBackpressureOnlyCancelsSlowConsumer},
        {"unread terminal retains request/token/cache admission", TestCompletedStreamsRetainAdmission},
        {"ordinary and typed request errors remain local", TestOrdinaryFailuresStayLocal},
        {"UTF-8 error bound preserves Progress and terminal types", TestUtf8ErrorsPreserveTypedProgress},
        {"RuntimeFault stops active and pending requests", TestRuntimeFaultStopsAllRequests},
        {"Submit setup faults roll back admission and stop peers", TestSubmitSetupFaultStopsScheduler},
        {"lifecycle fault preserves completed streams and broadcasts first cause", TestLifecycleFaultPreservesCompletedStreamsAndFirstCause},
        {"lifecycle fault races unpublished Submit and rolls back admission", TestLifecycleFaultWhileSubmitIsInitializing},
        {"reentrant lifecycle fault precedes terminal publication", TestReentrantLifecycleFaultBeforeTerminalPublication},
        {"dropping a running handle preserves lifetime until safe", TestDroppedRunningHandleRetainsSessionUntilSafe},
        {"readable handles outlive Scheduler", TestHandlesSurviveSchedulerDestruction},
        {"reentrant/concurrent Shutdown wakes readers", TestReentrantAndConcurrentShutdownWakesReaders},
        {"deferred callback fault suppresses a prepared token", TestDeferredCallbackFaultSuppressesPreparedToken},
        {"worker log callback destroys Scheduler owner safely", TestLogCallbackDestroysSchedulerOwner},
        {"Shutdown races Submit after reserve but before enqueue", TestShutdownWhileSubmitIsInitializing},
        {"decode batch limits and unsupported scalar fallback", TestDecodeBatchConfigurationAndFallback},
        {"continuous batch membership and prefill fairness", TestDecodeBatchDynamicMembershipAndPrefillFairness},
        {"batch identity isolation and width bound", TestDecodeBatchIdentityAndWidthBound},
        {"batch backpressure preserves full-ring finalization", TestDecodeBatchBackpressureAndFullRingFinalization},
        {"batch shared errors and device-domain fault scope", TestDecodeBatchSharedErrorScope},
        {"reentrant shutdown cancels every running batch row", TestDecodeBatchReentrantShutdownCancelsAllRunningRows},
        {"deferred fault suppresses all prepared batch rows", TestDecodeBatchDeferredFaultSuppressesAllPreparedRows},
        {"batch terminal releases its active slot before its admission credit", TestDecodeBatchTerminalReleasesActiveSlotBeforeCredit},
        {"dropped batch row retains its session until safe", TestDroppedDecodeBatchRowRetainsSessionUntilSafe},
    };
    size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << std::endl;
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << std::endl;
        }
    }
    std::cout << "AilaSchedulerTests: " << tests.size() - failures << " passed, " << failures << " failed\n";
    return failures == 0 ? 0 : 1;
}
