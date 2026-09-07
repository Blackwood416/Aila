#include "inference/Request.hpp"
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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using namespace std::chrono_literals;

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

template <typename Callable>
void ExpectError(ErrorCode code, Callable&& callable) {
    try {
        callable();
    } catch (const Error& error) {
        Expect(error.Code() == code, "Wrong error code: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected an inference Error");
}

struct FakeModel {
    // Represents model resources retained by a session, independently of a
    // caller's model handle. No device runtime is linked into this test.
    int marker = 19;
};

struct FakeState {
    int capacity = 128;
    int vocabSize = 64;
    int truncateGranularity = 1;
    std::vector<int> context;
    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> samplingHistories;
    std::vector<GenerationConfig> samplingConfigs;
    std::vector<int> samples;
    size_t samplesTaken = 0;
    size_t capturedLogits = 0;
    int resets = 0;
    std::optional<int> contextOverride;
    std::optional<int> truncateResultOverride;
    bool failContext = false;
    bool failReset = false;
    bool failTruncate = false;
    bool failBeforeForward = false;
    bool failAfterForward = false;
    bool throwUnknown = false;
    bool incompleteConsume = false;
    bool invalidLogitsSize = false;
    std::function<void()> duringForward;
};

class FakeBackend final : public SessionBackend {
public:
    FakeBackend(std::shared_ptr<FakeState> state, std::shared_ptr<FakeModel> model)
        : state(std::move(state)), model(std::move(model)) {}

    int ContextLength() const override {
        if (state->failContext) throw std::runtime_error("fake context query failed");
        return state->contextOverride.value_or(static_cast<int>(state->context.size()));
    }
    int Capacity() const override { return state->capacity; }
    int VocabSize() const override { return state->vocabSize; }
    size_t CacheBytes() const override {
        return static_cast<size_t>(state->capacity) * sizeof(int);
    }
    void Reset() override {
        if (state->failReset) throw std::runtime_error("fake reset failed");
        state->context.clear();
        ++state->resets;
    }
    int Truncate(int length) override {
        if (state->failTruncate) throw std::runtime_error("fake truncate failed");
        const int actual = length / state->truncateGranularity * state->truncateGranularity;
        state->context.resize(static_cast<size_t>(actual));
        return state->truncateResultOverride.value_or(actual);
    }
    ExecutionResult Forward(const std::vector<int>& tokenIds,
                            const SamplingInput* sampling,
                            bool captureLogits) override {
        if (state->failBeforeForward) throw std::runtime_error("fake forward failed before consume");
        state->inputs.push_back(tokenIds);
        if (!state->incompleteConsume) {
            state->context.insert(state->context.end(), tokenIds.begin(), tokenIds.end());
        }
        if (state->duringForward) state->duringForward();
        if (state->failAfterForward) throw std::runtime_error("fake forward failed after consume");
        if (state->throwUnknown) throw 42;

        uint32_t hash = static_cast<uint32_t>(model->marker);
        for (const int tokenId : state->context) hash = hash * 33u + static_cast<uint32_t>(tokenId);
        ExecutionResult result;
        if (sampling) {
            state->samplingHistories.push_back(sampling->generatedIds);
            state->samplingConfigs.push_back(sampling->config);
            // Includes all consumed context and prior outputs so interleaving
            // catches both cache and penalty-history contamination.
            for (const int tokenId : sampling->generatedIds) {
                hash = hash * 17u + static_cast<uint32_t>(tokenId);
            }
            if (sampling->config.do_sample) hash ^= sampling->rng();
            result.tokenId = state->samplesTaken < state->samples.size()
                                 ? state->samples[state->samplesTaken]
                                 : static_cast<int>(hash % static_cast<uint32_t>(state->vocabSize));
            ++state->samplesTaken;
        }
        if (captureLogits) {
            ++state->capturedLogits;
            const size_t count = state->invalidLogitsSize ? 1 : static_cast<size_t>(state->vocabSize);
            result.logits.resize(count);
            for (size_t i = 0; i < count; ++i) {
                result.logits[i] = static_cast<float>((hash + i) % 101u);
            }
        }
        return result;
    }

private:
    std::shared_ptr<FakeState> state;
    std::shared_ptr<FakeModel> model;
};

struct Fixture {
    std::shared_ptr<FakeState> state;
    std::shared_ptr<SessionState> session;
};

Fixture MakeFixture(int capacity = 128, int vocabSize = 64,
                    std::shared_ptr<FakeModel> model = std::make_shared<FakeModel>()) {
    auto state = std::make_shared<FakeState>();
    state->capacity = capacity;
    state->vocabSize = vocabSize;
    auto session = std::make_shared<SessionState>(std::make_unique<FakeBackend>(state, std::move(model)));
    return {std::move(state), std::move(session)};
}

RequestOptions Options(int maxNewTokens = 4, int chunkSize = 2, uint64_t seed = 42) {
    RequestOptions options;
    options.generation.max_new_tokens = maxNewTokens;
    options.generation.use_fixed_seed = true;
    options.generation.sampling_seed = seed;
    options.prefillChunkSize = chunkSize;
    return options;
}

std::vector<int> Complete(const std::shared_ptr<RequestState>& request) {
    std::vector<int> tokens;
    int terminals = 0;
    for (int i = 0; i < 1000; ++i) {
        const auto step = request->Step();
        if (step.tokenId) tokens.push_back(*step.tokenId);
        if (step.becameTerminal) ++terminals;
        if (IsTerminal(step.status)) {
            Expect(step.status == RequestStatus::Completed,
                   "Unexpected terminal: " + request->Snapshot().error);
            Expect(terminals == 1, "Completion did not publish exactly one terminal");
            return tokens;
        }
    }
    throw std::runtime_error("Request did not terminate within the test bound");
}

void TestSessionMutations() {
    auto fixture = MakeFixture(10, 16);
    Expect(fixture.session->Capacity() == 10 && fixture.session->VocabSize() == 16,
           "Session metadata was not retained");
    Expect(fixture.session->CacheBytes() == 10 * sizeof(int), "Wrong cache reservation");
    Expect(fixture.session->ForwardLogits({1, 2, 3}).size() == 16, "Missing captured logits");
    Expect(fixture.session->ContextLength() == 3, "Forward did not consume input");
    Expect(fixture.state->capturedLogits == 1 && fixture.state->samplesTaken == 0,
           "ForwardLogits sampled a token");
    Expect(fixture.session->Truncate(2) == 2, "Truncate returned wrong length");
    Expect(fixture.state->context == std::vector<int>({1, 2}), "Truncate changed retained prefix");
    fixture.session->Reset();
    Expect(fixture.session->ContextLength() == 0 && !fixture.session->Busy(), "Reset left session busy");

    ExpectError(ErrorCode::InvalidArgument, [&] { fixture.session->ForwardLogits({}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { fixture.session->ForwardLogits({-1}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { fixture.session->ForwardLogits({16}); });
    ExpectError(ErrorCode::ContextOverflow, [&] { fixture.session->ForwardLogits(std::vector<int>(11, 1)); });
    ExpectError(ErrorCode::InvalidArgument, [&] { fixture.session->Truncate(-1); });
    ExpectError(ErrorCode::InvalidArgument, [&] { fixture.session->Truncate(1); });
    Expect(!fixture.session->Busy() && fixture.session->ContextLength() == 0,
           "Rejected mutation leaked a lease or changed context");

    fixture.session->ForwardLogits({1, 2, 3, 4});
    fixture.state->truncateGranularity = 2;
    Expect(fixture.session->Truncate(3) == 2 && fixture.session->ContextLength() == 2,
           "Earlier snapshot restore length was not returned");
}

void TestBackendMetadataAndFailures() {
    ExpectError(ErrorCode::InvalidArgument, [] { SessionState session(nullptr); });
    ExpectError(ErrorCode::InvalidArgument, [] { MakeFixture(0, 16); });
    ExpectError(ErrorCode::InvalidArgument, [] { MakeFixture(16, 0); });
    auto state = std::make_shared<FakeState>();
    state->contextOverride = -1;
    ExpectError(ErrorCode::RuntimeFault, [&] {
        SessionState session(std::make_unique<FakeBackend>(state, std::make_shared<FakeModel>()));
    });

    auto fixture = MakeFixture();
    fixture.state->incompleteConsume = true;
    ExpectError(ErrorCode::RuntimeFault, [&] { fixture.session->ForwardLogits({1}); });
    fixture.state->incompleteConsume = false;
    fixture.state->invalidLogitsSize = true;
    ExpectError(ErrorCode::RuntimeFault, [&] { fixture.session->ForwardLogits({1}); });
    fixture.state->invalidLogitsSize = false;
    fixture.state->truncateResultOverride = 1;
    ExpectError(ErrorCode::RuntimeFault, [&] { fixture.session->Truncate(0); });
    fixture.state->truncateResultOverride.reset();
    Expect(!fixture.session->Busy(), "Backend contract failure leaked a lease");

    for (bool FakeState::* flag : {&FakeState::failReset, &FakeState::failTruncate,
                                  &FakeState::failBeforeForward}) {
        fixture.state.get()->*flag = true;
        bool failed = false;
        try {
            if (flag == &FakeState::failReset) fixture.session->Reset();
            else if (flag == &FakeState::failTruncate) fixture.session->Truncate(0);
            else fixture.session->ForwardLogits({1});
        } catch (const std::runtime_error&) {
            failed = true;
        }
        Expect(failed && !fixture.session->Busy(), "Mutation exception leaked its lease");
        fixture.state.get()->*flag = false;
    }
    fixture.session->Reset();
}

void TestChunkedGenerationAndFinalization() {
    auto fixture = MakeFixture(8);
    auto options = Options(3, 2);
    options.generation.do_sample = false;
    auto request = RequestState::Create(fixture.session, {1, 2, 3, 4, 5}, options);
    Expect(request->Snapshot().status == RequestStatus::Queued && fixture.session->Busy(),
           "New request did not acquire session");
    for (size_t processed : {size_t{2}, size_t{4}}) {
        const auto step = request->Step();
        Expect(step.status == RequestStatus::Prefill && !step.tokenId && !step.becameTerminal,
               "Nonfinal prefill chunk emitted output");
        Expect(request->Snapshot().promptTokensProcessed == processed &&
                   fixture.session->ContextLength() == static_cast<int>(processed),
               "Prefill progress and cache length disagree");
    }
    std::vector<int> output;
    for (int i = 0; i < 3; ++i) {
        const auto step = request->Step();
        Expect(step.tokenId.has_value() && !step.becameTerminal, "Missing generated token");
        output.push_back(*step.tokenId);
        Expect(step.status == (i == 2 ? RequestStatus::Finalizing : RequestStatus::Decode),
               "Unexpected generation phase");
        Expect(fixture.session->ContextLength() == 5 + i, "Decode consumed wrong number of tokens");
    }
    Expect(fixture.session->Busy(), "Lease released before final token was cached");
    const auto final = request->Step();
    Expect(final.status == RequestStatus::Completed && final.becameTerminal && !final.tokenId,
           "Finalizing did not finish without an extra sample");
    Expect(fixture.state->samplesTaken == 3, "Finalizing advanced the sampler");
    std::vector<int> expected{1, 2, 3, 4, 5};
    expected.insert(expected.end(), output.begin(), output.end());
    Expect(fixture.state->context == expected && fixture.session->ContextLength() == 8,
           "Completed cache does not contain every published output token");
    Expect(fixture.state->inputs == std::vector<std::vector<int>>({{1, 2}, {3, 4}, {5},
                                                                 {output[0]}, {output[1]}, {output[2]}}),
           "A step submitted more than one prefill chunk or decode token");
    Expect(!fixture.session->Busy(), "Completion leaked session lease");
    const size_t calls = fixture.state->inputs.size();
    request->Cancel();
    const auto again = request->Step();
    Expect(again.status == RequestStatus::Completed && !again.becameTerminal && !again.tokenId,
           "Repeated terminal step or late cancellation changed completion");
    Expect(fixture.state->inputs.size() == calls, "Terminal request submitted more work");
}

void TestStopTokens() {
    for (const std::vector<int>& samples : {std::vector<int>{7}, std::vector<int>{5, 7}}) {
        auto fixture = MakeFixture();
        fixture.state->samples = samples;
        auto options = Options(8, 4);
        options.stopTokenIds = {7, 7};
        auto request = RequestState::Create(fixture.session, {1, 2}, options);
        const auto output = Complete(request);
        const std::vector<int> expectedOutput = samples.size() == 1 ? std::vector<int>{}
                                                                  : std::vector<int>{5};
        Expect(output == expectedOutput && request->Snapshot().generatedIds == expectedOutput,
               "Stop token was published");
        std::vector<int> expectedContext{1, 2};
        expectedContext.insert(expectedContext.end(), expectedOutput.begin(), expectedOutput.end());
        Expect(fixture.state->context == expectedContext, "Stop token was cached or output was lost");
    }
}

void TestIndependentInterleavingAndRng() {
    const std::vector<int> promptA{1, 3, 5, 7, 9};
    const std::vector<int> promptB{2, 4, 6, 8, 10, 12, 14};
    auto optionsA = Options(8, 2, 42);
    auto optionsB = Options(6, 3, 1234);
    optionsA.generation.repetition_penalty = 1.1f;
    optionsA.generation.presence_penalty = 0.3f;
    optionsB.generation.frequency_penalty = -0.2f;
    auto referenceA = MakeFixture();
    auto referenceB = MakeFixture();
    const auto expectedA = Complete(RequestState::Create(referenceA.session, promptA, optionsA));
    const auto expectedB = Complete(RequestState::Create(referenceB.session, promptB, optionsB));

    auto model = std::make_shared<FakeModel>();
    auto fixtureA = MakeFixture(128, 64, model);
    auto fixtureB = MakeFixture(128, 64, model);
    auto requestA = RequestState::Create(fixtureA.session, promptA, optionsA);
    auto requestB = RequestState::Create(fixtureB.session, promptB, optionsB);
    for (int i = 0; i < 100; ++i) {
        if (!IsTerminal(requestA->Snapshot().status)) requestA->Step();
        if (!IsTerminal(requestB->Snapshot().status)) requestB->Step();
        if (IsTerminal(requestA->Snapshot().status) && IsTerminal(requestB->Snapshot().status)) break;
    }
    Expect(requestA->Snapshot().status == RequestStatus::Completed &&
               requestB->Snapshot().status == RequestStatus::Completed,
           "Interleaved requests did not complete");
    Expect(requestA->Snapshot().generatedIds == expectedA && requestB->Snapshot().generatedIds == expectedB,
           "Interleaving changed request-local token sequences or RNG");
    Expect(fixtureA.state->context == referenceA.state->context &&
               fixtureB.state->context == referenceB.state->context,
           "Interleaving changed session cache");
    Expect(fixtureA.state->samplingHistories == referenceA.state->samplingHistories &&
               fixtureB.state->samplingHistories == referenceB.state->samplingHistories,
           "Sampling received another request's output history");
    Expect(fixtureA.state->samplingConfigs.front().repetition_penalty == 1.1f &&
               fixtureB.state->samplingConfigs.front().frequency_penalty == -0.2f,
           "Sampling config was not forwarded");

    auto sameLowSeed = MakeFixture();
    optionsA.generation.sampling_seed += uint64_t{1} << 32;
    Expect(Complete(RequestState::Create(sameLowSeed.session, promptA, optionsA)) == expectedA,
           "Fixed seed does not retain legacy low-32-bit semantics");
}

void TestBusyLeaseAndAbandonment() {
    auto fixture = MakeFixture();
    auto request = RequestState::Create(fixture.session, {1, 2, 3}, Options());
    ExpectError(ErrorCode::Busy, [&] { RequestState::Create(fixture.session, {4}, Options()); });
    ExpectError(ErrorCode::Busy, [&] { fixture.session->Reset(); });
    ExpectError(ErrorCode::Busy, [&] { fixture.session->Truncate(-1); });
    ExpectError(ErrorCode::Busy, [&] { fixture.session->ForwardLogits({}); });
    request->Step();
    request.reset();
    Expect(!fixture.session->Busy() && fixture.session->ContextLength() == 2,
           "Abandoned request failed to release its lease or preserve consumed prefix");
    auto replacement = RequestState::Create(fixture.session, {5}, Options(1));
    Complete(replacement);
    Expect(fixture.session->ContextLength() == 4, "Replacement request did not append to existing context");
    fixture.session->Reset();
    Expect(fixture.session->ContextLength() == 0, "Completed request still prevents reset");
}

void TestRequestValidation() {
    auto fixture = MakeFixture(16, 16);
    ExpectError(ErrorCode::InvalidArgument, [] { RequestState::Create(nullptr, {1}, Options()); });
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {}, Options()); });
    for (int id : {-1, 16}) {
        ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {id}, Options()); });
        auto stopOptions = Options();
        stopOptions.stopTokenIds = {id};
        ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {1}, stopOptions); });
    }
    const float nan = (std::numeric_limits<float>::quiet_NaN)();
    const float inf = (std::numeric_limits<float>::infinity)();
    const std::vector<std::function<void(RequestOptions&)>> invalid{
        [](auto& o) { o.prefillChunkSize = 0; },
        [](auto& o) { o.prefillChunkSize = -1; },
        [](auto& o) { o.generation.max_new_tokens = 0; },
        [](auto& o) { o.generation.max_new_tokens = -1; },
        [nan](auto& o) { o.generation.temperature = nan; },
        [inf](auto& o) { o.generation.temperature = inf; },
        [](auto& o) { o.generation.temperature = -1.0f; },
        [](auto& o) { o.generation.temperature = 0.0f; },
        [nan](auto& o) { o.generation.top_p = nan; },
        [inf](auto& o) { o.generation.top_p = inf; },
        [](auto& o) { o.generation.top_p = 0.0f; },
        [](auto& o) { o.generation.top_p = 1.01f; },
        [](auto& o) { o.generation.top_k = -1; },
        [nan](auto& o) { o.generation.repetition_penalty = nan; },
        [inf](auto& o) { o.generation.repetition_penalty = inf; },
        [](auto& o) { o.generation.repetition_penalty = 0.0f; },
        [](auto& o) { o.generation.repetition_penalty = -1.0f; },
        [inf](auto& o) { o.generation.presence_penalty = inf; },
        [nan](auto& o) { o.generation.presence_penalty = nan; },
        [inf](auto& o) { o.generation.frequency_penalty = -inf; },
        [nan](auto& o) { o.generation.frequency_penalty = nan; },
        [](auto& o) { o.generation.thinking_budget_tokens = 0; },
        [](auto& o) { o.generation.thinking_budget_tokens = 5; },
        [](auto& o) { o.generation.thinking_budget_tokens = -2; },
    };
    for (const auto& edit : invalid) {
        auto options = Options();
        edit(options);
        ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {1}, options); });
        Expect(!fixture.session->Busy() && fixture.state->inputs.empty(),
               "Validation performed inference or leaked lease");
    }

    auto valid = Options(1);
    valid.generation.do_sample = false;
    valid.generation.temperature = 0.0f;
    valid.generation.top_p = 1.0f;
    valid.generation.top_k = 0;
    valid.generation.repetition_penalty = 0.5f;
    valid.generation.presence_penalty = -0.3f;
    valid.generation.frequency_penalty = -0.4f;
    Complete(RequestState::Create(fixture.session, {1}, valid));
}

void TestCapacityAndCreateExceptions() {
    auto fixture = MakeFixture(8);
    ExpectError(ErrorCode::ContextOverflow, [&] { RequestState::Create(fixture.session, {1, 2}, Options(7)); });
    fixture.session->ForwardLogits({1, 2, 3, 4, 5, 6});
    ExpectError(ErrorCode::ContextOverflow, [&] { RequestState::Create(fixture.session, {7}, Options(2)); });
    Expect(!fixture.session->Busy(), "Capacity validation after acquisition leaked lease");
    Complete(RequestState::Create(fixture.session, {7}, Options(1)));
    Expect(fixture.session->ContextLength() == 8, "Exact-capacity request failed to retain final token");

    auto huge = MakeFixture((std::numeric_limits<int>::max)());
    ExpectError(ErrorCode::ContextOverflow, [&] {
        RequestState::Create(huge.session, {1}, Options((std::numeric_limits<int>::max)()));
    });
    Expect(!huge.session->Busy(), "Integer boundary admission leaked lease");

    fixture.session->Reset();
    fixture.state->failContext = true;
    bool failed = false;
    try { RequestState::Create(fixture.session, {1}, Options(1)); }
    catch (const std::runtime_error&) { failed = true; }
    Expect(failed && !fixture.session->Busy(), "Context query exception leaked lease");
    fixture.state->failContext = false;
    fixture.state->contextOverride = 9;
    ExpectError(ErrorCode::RuntimeFault, [&] { RequestState::Create(fixture.session, {1}, Options(1)); });
    Expect(!fixture.session->Busy(), "Invalid context length leaked lease");
    fixture.state->contextOverride.reset();
    Complete(RequestState::Create(fixture.session, {1}, Options(1)));
}

void TestCancellationBoundaries() {
    auto fixture = MakeFixture();
    auto request = RequestState::Create(fixture.session, {1, 2, 3, 4}, Options());
    request->Cancel();
    request->Cancel();
    Expect(fixture.session->Busy(), "Cancel released a live request before its step boundary");
    const auto cancelled = request->Step();
    Expect(cancelled.status == RequestStatus::Cancelled && cancelled.becameTerminal && !cancelled.tokenId,
           "Queued cancellation did not terminate once");
    Expect(fixture.state->inputs.empty() && request->Snapshot().promptTokensProcessed == 0,
           "Queued cancellation submitted work");
    Expect(!request->Step().becameTerminal && !fixture.session->Busy(), "Repeated cancel published another terminal");

    request = RequestState::Create(fixture.session, {1, 2, 3, 4}, Options());
    request->Step();
    request->Cancel();
    Expect(request->Step().status == RequestStatus::Cancelled, "Chunk-boundary cancellation failed");
    Expect(fixture.state->context == std::vector<int>({1, 2}) &&
               request->Snapshot().promptTokensProcessed == 2,
           "Chunk cancellation changed consumed prefix");

    fixture.session->Reset();
    request = RequestState::Create(fixture.session, {5}, Options(1));
    Expect(request->Step().status == RequestStatus::Finalizing, "Expected pending output token");
    request->Cancel();
    Expect(request->Step().status == RequestStatus::Cancelled && fixture.session->ContextLength() == 1,
           "Cancellation submitted the pending output token");
    Expect(request->Snapshot().generatedIds.size() == 1, "Cancellation erased already published output");
}

class ForwardGate final {
public:
    void Enter() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        condition.notify_all();
        condition.wait(lock, [&] { return released; });
    }
    bool AwaitEntry() {
        std::unique_lock<std::mutex> lock(mutex);
        return condition.wait_for(lock, 5s, [&] { return entered; });
    }
    void Release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }

private:
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool released = false;
};

void TestCancellationDuringExecutionAndConcurrentStep() {
    auto fixture = MakeFixture();
    auto request = RequestState::Create(fixture.session, {1, 2}, Options(4, 2));
    ForwardGate gate;
    fixture.state->duringForward = [&] { gate.Enter(); };
    auto running = std::async(std::launch::async, [request] { return request->Step(); });
    std::future<RequestSnapshot> snapshot;
    try {
        Expect(gate.AwaitEntry(), "Forward did not reach execution gate");
        ExpectError(ErrorCode::Busy, [&] { request->Step(); });
        ExpectError(ErrorCode::Busy, [&] { fixture.session->Reset(); });
        ExpectError(ErrorCode::Busy, [&] { RequestState::Create(fixture.session, {3}, Options()); });
        snapshot = std::async(std::launch::async, [request] { return request->Snapshot(); });
        Expect(snapshot.wait_for(20ms) == std::future_status::timeout,
               "Snapshot exposed an in-progress state mutation");
        request->Cancel();
    } catch (...) {
        gate.Release();
        running.wait();
        if (snapshot.valid()) snapshot.wait();
        throw;
    }
    gate.Release();
    Expect(running.wait_for(5s) == std::future_status::ready, "In-flight cancellation did not finish");
    const auto step = running.get();
    const auto view = snapshot.get();
    Expect(step.status == RequestStatus::Cancelled && step.becameTerminal && !step.tokenId,
           "Cancelled in-flight forward published a sampled token");
    Expect(view.status == RequestStatus::Cancelled && view.promptTokensProcessed == 2 && view.generatedIds.empty(),
           "Snapshot does not reflect consumed input and suppressed output");
    Expect(fixture.state->context == std::vector<int>({1, 2}) && !fixture.session->Busy(),
           "In-flight cancellation lost cache prefix or leaked lease");
}

void TestStepReentry() {
    auto fixture = MakeFixture();
    auto request = RequestState::Create(fixture.session, {1, 2}, Options());
    bool reentered = false;
    fixture.state->duringForward = [&] {
        ExpectError(ErrorCode::Busy, [&] { request->Step(); });
        ExpectError(ErrorCode::Busy, [&] { fixture.session->ForwardLogits({3}); });
        request->Cancel();
        reentered = true;
    };
    const auto step = request->Step();
    Expect(reentered && step.status == RequestStatus::Cancelled && step.becameTerminal,
           "Reentrant Step did not report Busy without disturbing outer step");
}

void TestFailuresStayRequestLocal() {
    auto model = std::make_shared<FakeModel>();
    auto other = MakeFixture(128, 64, model);
    auto otherRequest = RequestState::Create(other.session, {4, 5}, Options(3));
    for (int mode = 0; mode < 4; ++mode) {
        auto fixture = MakeFixture(128, 64, model);
        fixture.state->failBeforeForward = mode == 0;
        fixture.state->failAfterForward = mode == 1;
        fixture.state->throwUnknown = mode == 2;
        if (mode == 3) fixture.state->samples = {64};
        auto request = RequestState::Create(fixture.session, {1, 2}, Options());
        const auto failed = request->Step();
        Expect(failed.status == RequestStatus::Failed && failed.becameTerminal && !failed.tokenId,
               "Backend exception did not become a Failed terminal");
        Expect(!request->Snapshot().error.empty() && !fixture.session->Busy(),
               "Failure lost error details or session lease");
        Expect(fixture.session->ContextLength() == (mode == 0 ? 0 : 2),
               "Failure erased consumed cache or invented progress");
        const auto again = request->Step();
        Expect(again.status == RequestStatus::Failed && !again.becameTerminal,
               "Failed request published another terminal");
        fixture.state->failBeforeForward = false;
        fixture.state->failAfterForward = false;
        fixture.state->throwUnknown = false;
        fixture.state->samples.clear();
        Complete(RequestState::Create(fixture.session, {3}, Options(1)));
    }
    Expect(Complete(otherRequest).size() == 3, "Ordinary backend failure invalidated another session");
}

void TestCancelledSessionDoesNotAffectPeer() {
    auto expected = MakeFixture();
    const auto options = Options(6, 2, 9876);
    const auto expectedTokens = Complete(RequestState::Create(expected.session, {2, 4, 6, 8}, options));

    auto model = std::make_shared<FakeModel>();
    auto a = MakeFixture(128, 64, model);
    auto b = MakeFixture(128, 64, model);
    auto requestA = RequestState::Create(a.session, {1, 3, 5}, Options(8, 2, 42));
    auto requestB = RequestState::Create(b.session, {2, 4, 6, 8}, options);
    requestA->Step();
    requestB->Step();
    requestA->Step();
    requestA->Cancel();
    Expect(requestA->Step().status == RequestStatus::Cancelled, "A did not cancel");
    requestA.reset();
    a.session.reset();
    Expect(Complete(requestB) == expectedTokens && b.state->context == expected.state->context,
           "Cancelling and destroying A changed B output or cache");
    auto replacement = MakeFixture(128, 64, model);
    Expect(Complete(RequestState::Create(replacement.session, {2, 4, 6, 8}, options)) == expectedTokens,
           "New session could not reuse retained model resources");
}

void TestStrongOwnershipAndSnapshotCopy() {
    auto model = std::make_shared<FakeModel>();
    std::weak_ptr<FakeModel> weakModel = model;
    auto fixture = MakeFixture(128, 64, model);
    std::weak_ptr<SessionState> weakSession = fixture.session;
    auto request = RequestState::Create(fixture.session, {1}, Options(1));
    fixture.session.reset();
    model.reset();
    Expect(!weakSession.expired() && !weakModel.expired(), "Request did not retain its session/model");
    request->Step();
    auto snapshot = request->Snapshot();
    snapshot.generatedIds.push_back(-1);
    snapshot.error = "caller mutation";
    Expect(request->Snapshot().generatedIds.size() == 1 && request->Snapshot().error.empty(),
           "Snapshot exposes request's mutable state");
    Expect(request->Step().status == RequestStatus::Completed, "Retained request could not finish");
    Expect(!weakSession.expired(), "Terminal handle lost its retained session before destruction");
    request.reset();
    Expect(weakSession.expired() && weakModel.expired(), "Request/session retained model resources after destruction");
}

void TestDeferredLogCallbackRetainsRequest() {
    auto fixture = MakeFixture();
    auto request = RequestState::Create(fixture.session, {1, 2, 3}, Options(4, 2));
    struct CallbackProbe {
        std::shared_ptr<RequestState>* callerRequest;
        std::weak_ptr<RequestState> weakRequest;
        SessionState* session;
        const char* marker = "RequestStateTests deferred Step lifetime";
        int calls = 0;
        bool forwardFinished = false;
        bool deliveredAfterForward = false;
        bool aliveAfterReset = false;
        bool aliveAfterReentry = false;
        bool sessionBusy = false;
        bool reentryBusy = false;
        bool unexpectedException = false;
    } probe{&request, request, fixture.session.get()};
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            // The suite otherwise uses the default callback.
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restoreLogging;

    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* message, void* userData) {
        auto& probe = *static_cast<CallbackProbe*>(userData);
        if (std::strcmp(message, probe.marker) != 0) return;
        ++probe.calls;
        probe.deliveredAfterForward = probe.forwardFinished;
        probe.callerRequest->reset();
        // Check before weak.lock(): the callback must not be what keeps Step alive.
        probe.aliveAfterReset = !probe.weakRequest.expired();
        probe.sessionBusy = probe.session->Busy();
        if (auto activeRequest = probe.weakRequest.lock()) {
            try {
                activeRequest->Step();
            } catch (const Error& error) {
                probe.reentryBusy = error.Code() == ErrorCode::Busy;
                probe.unexpectedException = !probe.reentryBusy;
            } catch (...) {
                probe.unexpectedException = true;
            }
        }
        probe.aliveAfterReentry = !probe.weakRequest.expired();
    }, &probe);
    fixture.state->duringForward = [&probe] {
        AILA_LOG_INFO("%s", probe.marker);
        probe.forwardFinished = true;
    };

    // Do not copy request here: the callback releases its last external owner.
    const auto step = request->Step();
    // Deferred logging swallows callback exceptions, so assert only after Step.
    Expect(probe.calls == 1 && probe.deliveredAfterForward,
           "Forward log was not deferred until Step unwound");
    Expect(!request && probe.aliveAfterReset && probe.aliveAfterReentry,
           "Request died while its deferred log callback was still running");
    Expect(probe.reentryBusy && !probe.unexpectedException,
           "Deferred log callback's recursive Step did not report Busy");
    Expect(probe.sessionBusy && probe.weakRequest.expired() && !fixture.session->Busy(),
           "Step did not retain then release the abandoned request's session lease");
    Expect(step.status == RequestStatus::Prefill && !step.tokenId && !step.becameTerminal &&
               fixture.state->context == std::vector<int>({1, 2}),
           "Deferred log callback changed the outer Step result or consumed prefix");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"session mutations and actual truncate length", TestSessionMutations},
        {"backend metadata and mutation exceptions", TestBackendMetadataAndFailures},
        {"chunked generation and final token retention", TestChunkedGenerationAndFinalization},
        {"stop tokens are neither published nor cached", TestStopTokens},
        {"interleaving preserves cache, RNG and sampling history", TestIndependentInterleavingAndRng},
        {"busy lease and abandoned request", TestBusyLeaseAndAbandonment},
        {"request input and numerical validation", TestRequestValidation},
        {"capacity overflow and Create exception release", TestCapacityAndCreateExceptions},
        {"cancellation at queued, chunk and finalize boundaries", TestCancellationBoundaries},
        {"in-flight cancellation, concurrent Step and snapshots", TestCancellationDuringExecutionAndConcurrentStep},
        {"same-thread Step reentry returns Busy", TestStepReentry},
        {"backend failures stay request-local", TestFailuresStayRequestLocal},
        {"cancel and destroy A while B continues", TestCancelledSessionDoesNotAffectPeer},
        {"strong ownership and snapshot value semantics", TestStrongOwnershipAndSnapshotCopy},
        {"deferred log callback retains request and Step guard", TestDeferredLogCallbackRetainsRequest},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << "AilaRequestStateTests: " << tests.size() - static_cast<size_t>(failures)
              << " passed, " << failures << " failed\n";
    return failures == 0 ? 0 : 1;
}
