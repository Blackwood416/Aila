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
    bool batchEnabled = false;
    std::vector<size_t> batchWidths;
    std::function<void()> duringBatch;
    std::optional<ErrorCode> batchFailure;
    bool ordinaryBatchFailure = false;
    bool invalidBatchResultCount = false;
};

struct FakeState {
    int capacity = 128;
    int vocabSize = 64;
    int truncateGranularity = 1;
    std::vector<int> context;
    std::vector<std::vector<int>> inputs;
    std::vector<std::vector<int>> samplingHistories;
    std::vector<GenerationConfig> samplingConfigs;
    std::vector<bool> hostSampling;
    std::vector<bool> legacyPrefillLogits;
    // Prompt-phase entry points only; decode stays on plain Forward.
    std::vector<int> prefillRemaining;
    std::vector<std::vector<int>> prefillLookahead;
    int prefillForwards = 0;
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
    bool windowEnabled = false;
    bool windowAccepts = true;
    int windows = 0;
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
    const void* DecodeBatchKey() const noexcept override {
        return model->batchEnabled ? model.get() : nullptr;
    }
    bool SupportsDecodeWindow() const noexcept override { return state->windowEnabled; }
    std::optional<DecodeWindowResult> ForwardWindow(int pending, const DecodeWindowSelector& selector,
                                                    int /*maxWidth*/ = 0) override {
        ++state->windows;
        state->context.push_back(pending);
        const int first = state->samples.at(state->samplesTaken++);
        if (state->duringForward) state->duringForward();
        if (state->windowAccepts && selector(first)) {
            state->context.push_back(first);
            const int bonus = state->samples.at(state->samplesTaken++);
            return DecodeWindowResult{first, bonus, 2, int(state->context.size()), true};
        }
        return DecodeWindowResult{first, std::nullopt, 1, int(state->context.size()), false};
    }
    std::vector<ExecutionResult> ForwardDecodeBatch(const std::vector<BackendDecodeInput>& rows) override {
        std::vector<ExecutionResult> results;
        for (const auto& row : rows) {
            auto* peer = dynamic_cast<FakeBackend*>(row.backend);
            Expect(peer && peer->model == model, "Fake decode batch mixed model identities");
        }
        model->batchWidths.push_back(rows.size());
        if (model->duringBatch) model->duringBatch();
        if (model->batchFailure) throw Error(*model->batchFailure, "fake shared batch failure");
        if (model->ordinaryBatchFailure) throw std::runtime_error("fake ordinary batch failure");
        for (const auto& row : rows) {
            results.push_back(row.backend->Forward({row.tokenId}, row.sampling, row.captureLogits));
        }
        if (model->invalidBatchResultCount) results.pop_back();
        return results;
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
            state->hostSampling.push_back(sampling->forceHost);
            state->legacyPrefillLogits.push_back(sampling->legacyPrefillLogits);
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

    ExecutionResult ForwardPrefill(const std::vector<int>& tokenIds,
                                   const SamplingInput* sampling,
                                   bool captureLogits,
                                   int prefillRemaining,
                                   const int* nextTokens, int nextCount) override {
        ++state->prefillForwards;
        state->prefillRemaining.push_back(prefillRemaining);
        state->prefillLookahead.push_back(nextCount ? std::vector<int>(nextTokens, nextTokens + nextCount) : std::vector<int>{});
        return Forward(tokenIds, sampling, captureLogits);
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

void TestPrefillOnlyAndCancelledReplay() {
    const std::vector<int> prefix{1, 2, 3, 4, 5};
    auto options = Options(0, 2);
    options.prefillOnly = true;
    auto fixture = MakeFixture();
    auto replay = RequestState::Create(fixture.session, prefix, options);
    Expect(fixture.session->Busy(), "Prefill did not acquire the Session");
    ExpectError(ErrorCode::Busy, [&] { fixture.session->Reset(); });
    Expect(Complete(replay).empty() && fixture.state->context == prefix &&
               fixture.state->inputs == std::vector<std::vector<int>>({{1, 2}, {3, 4}, {5}}) &&
               fixture.state->samplesTaken == 0 && fixture.state->capturedLogits == 0,
           "Prefill emitted tokens, sampled, captured logits, or changed the chunk boundaries");
    Expect(replay->Snapshot().endReason == RequestEndReason::Length &&
               replay->Snapshot().promptTokensProcessed == prefix.size() && !fixture.session->Busy(),
           "Prefill completion did not report/retain exactly the prompt");
    auto direct = MakeFixture();
    direct.session->ForwardLogits(prefix);
    Expect(Complete(RequestState::Create(fixture.session, {6}, Options())) ==
               Complete(RequestState::Create(direct.session, {6}, Options())),
           "Prefill changed the following request's context, RNG or penalties");
    auto exact = MakeFixture(5);
    Expect(Complete(RequestState::Create(exact.session, prefix, options)).empty() && exact.session->ContextLength() == 5,
           "Prefill incorrectly reserved an extra generated token at capacity");
    fixture = MakeFixture();
    replay = RequestState::Create(fixture.session, prefix, options);
    replay->Cancel();
    Expect(replay->Step().status == RequestStatus::Cancelled && fixture.state->inputs.empty(),
           "Queued prefill cancellation entered the backend");
    replay = RequestState::Create(fixture.session, prefix, options);
    fixture.state->duringForward = [&] { replay->Cancel(); };
    Expect(replay->Step().status == RequestStatus::Cancelled &&
               fixture.state->context == std::vector<int>({1, 2}) && !fixture.session->Busy(),
           "Active prefill cancellation lost its completed prefix or Session lease");
    fixture.state->duringForward = {};
    Expect(Complete(RequestState::Create(fixture.session, {3, 4, 5}, options)).empty() &&
               fixture.state->context == prefix && fixture.state->samplesTaken == 0,
           "Cancelled prefill could not replay only its missing suffix");
    for (int invalid = 0; invalid < 4; ++invalid) {
        auto bad = options;
        if (invalid == 0) bad.generation.max_new_tokens = 1;
        if (invalid == 1) bad.stopTokenIds = {2};
        if (invalid == 2) bad.generation.thinking_budget_tokens = 0;
        if (invalid == 3) bad.thinkingBudget = aila::generation::ThinkingBudgetOptions{1, 10, 11, true, {11}};
        ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {1}, bad); });
        Expect(!fixture.session->Busy(), "Invalid prefill acquired the Session");
    }
}

RequestOptions ThinkingOptions(int maximum, uint32_t budget, bool opens = true) {
    auto options = Options(maximum, 2);
    options.thinkingBudget = aila::generation::ThinkingBudgetOptions{budget, 10, 11, opens, {11, 3}};
    return options;
}

void TestSingleCloseAndHostSamplingPolicy() {
    auto fixture = MakeFixture();
    fixture.state->samples = {20, 21, 10, 22, 23, 24, 25, 26};
    auto options = ThinkingOptions(10, 2);
    options.thinkingBudget->closeOnce = true;
    options.forceHostSampling = true;
    options.legacyPrefillLogits = true;
    auto request = RequestState::Create(fixture.session, {1}, options);
    Expect(Complete(request) == std::vector<int>({20, 21, 11, 3, 10, 22, 23, 24, 25, 26}),
           "One-shot policy forced a second close after the model reopened thinking");
    Expect(fixture.state->samplesTaken == 8 &&
               fixture.state->hostSampling == std::vector<bool>(8, true) &&
               fixture.state->legacyPrefillLogits == std::vector<bool>(8, true),
           "One-shot policy consumed RNG for forced tokens or lost a sampling policy");
    auto peer = MakeFixture();
    peer.state->samples = {20, 21, 10, 22};
    auto regular = RequestState::Create(peer.session, {1}, ThinkingOptions(8, 2));
    Expect(Complete(regular) == std::vector<int>({20, 21, 11, 3, 10, 11, 3, 22}) &&
               peer.state->hostSampling == std::vector<bool>(4, false) &&
               peer.state->legacyPrefillLogits == std::vector<bool>(4, false),
           "Compatibility policies changed ordinary Request defaults");
}

void TestThinkingBudgetClosesAndAccounts() {
    auto fixture = MakeFixture();
    fixture.state->samples = {20, 21, 22, 23, 24, 25};
    auto request = RequestState::Create(fixture.session, {1, 2, 4}, ThinkingOptions(8, 2));
    Expect(Complete(request) == std::vector<int>({20, 21, 11, 3, 22, 23, 24, 25}),
           "Thinking close sequence was not emitted at the budget boundary");
    Expect(fixture.state->samplesTaken == 6 && fixture.state->samplingHistories[2] == std::vector<int>({20, 21, 11, 3}),
           "Forced close burned RNG or was missing from the penalty history");
    Expect(fixture.state->context == std::vector<int>({1, 2, 4, 20, 21, 11, 3, 22, 23, 24, 25}),
           "Forced close/final token did not enter the session cache");
    Expect(request->Progress().endReason == RequestEndReason::Length && !fixture.session->Busy(),
           "Budgeted completion lost its terminal reason or lease");
    fixture.state->samples.push_back(26);
    Expect(Complete(RequestState::Create(fixture.session, {5}, Options(1))) == std::vector<int>({26}),
           "Session could not continue after a forced close");
}

void TestThinkingBudgetZeroNaturalAndReopened() {
    auto zero = MakeFixture();
    zero.state->samples = {20};
    Expect(Complete(RequestState::Create(zero.session, {1, 2, 4}, ThinkingOptions(3, 0))) ==
               std::vector<int>({11, 3, 20}) && zero.state->samplesTaken == 1,
           "Zero budget must close an open prompt before sampling");
    auto natural = MakeFixture();
    natural.state->samples = {20, 11, 21, 10, 22, 23};
    Expect(Complete(RequestState::Create(natural.session, {1}, ThinkingOptions(8, 2))) ==
               std::vector<int>({20, 11, 21, 10, 22, 11, 3, 23}),
           "Natural close/reopen reset the per-request thinking count");
    auto reopened = MakeFixture();
    reopened.state->samples = {10, 10, 20};
    Expect(Complete(RequestState::Create(reopened.session, {1}, ThinkingOptions(7, 0, false))) ==
               std::vector<int>({10, 11, 3, 10, 11, 3, 20}),
           "Model-opened thinking escaped a zero or already exhausted budget");
}

void TestThinkingBudgetLengthAndCancellation() {
    auto shortOutput = MakeFixture();
    shortOutput.state->samples = {20, 21, 22};
    auto request = RequestState::Create(shortOutput.session, {1}, ThinkingOptions(3, 2));
    Expect(Complete(request) == std::vector<int>({20, 21}), "Insufficient close space emitted a partial marker sequence");
    Expect(request->Snapshot().endReason == RequestEndReason::Length && shortOutput.state->samplesTaken == 2 &&
               shortOutput.state->context == std::vector<int>({1, 20, 21}),
           "Early length finish sampled extra output or failed to flush the last token");
    auto none = MakeFixture();
    auto noOutput = RequestState::Create(none.session, {1, 2, 4}, ThinkingOptions(1, 0));
    Expect(Complete(noOutput).empty() && noOutput->Progress().endReason == RequestEndReason::Length &&
               none.state->samplesTaken == 0 && none.state->context == std::vector<int>({1, 2, 4}),
           "No-room budget did not consume its prompt without sampling");
    auto cancelled = MakeFixture();
    auto cancel = RequestState::Create(cancelled.session, {1}, ThinkingOptions(4, 0));
    Expect(cancel->Step().tokenId == 11, "Missing first forced marker");
    cancel->Cancel();
    const auto end = cancel->Step();
    Expect(end.becameTerminal && end.status == RequestStatus::Cancelled && !end.tokenId &&
               cancelled.state->samplesTaken == 0 && cancelled.state->inputs.size() == 1 && !cancelled.session->Busy(),
           "Cancel during forced close submitted more work or leaked the Session");
}

void TestThinkingBudgetValidationAndIsolation() {
    auto fixture = MakeFixture();
    auto invalid = [&](RequestOptions options) {
        ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::Create(fixture.session, {1}, options); });
        Expect(!fixture.session->Busy() && fixture.state->inputs.empty(), "Invalid thinking options acquired or executed the Session");
    };
    auto options = ThinkingOptions(8, 2);
    options.thinkingBudget->closeTokenIds.clear(); invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->closeTokenIds = {3, 11}; invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->closeTokenIds = {11, 64}; invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->closeTokenIds = {11, 10}; invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->closeTokenIds = {11, 11}; invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->closeTokenIds.resize(17, 3); invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->openTokenId = 11; invalid(options);
    options = ThinkingOptions(8, 2); options.stopTokenIds = {11}; invalid(options);
    options = ThinkingOptions(8, 2); options.stopTokenIds = {3}; invalid(options);
    options = ThinkingOptions(8, 2); options.generation.thinking_budget_tokens = 3; invalid(options);
    options = ThinkingOptions(8, 2); options.thinkingBudget->maxTokens = UINT32_MAX; invalid(options);

    const auto make = [] {
        auto result = MakeFixture();
        result.state->samples = {20, 21, 22, 23, 24, 25, 26, 27};
        return result;
    };
    auto a = make(), b = make(), referenceA = make(), referenceB = make();
    const auto optionsA = ThinkingOptions(8, 2), optionsB = ThinkingOptions(7, 0);
    const auto expectedA = Complete(RequestState::Create(referenceA.session, {1, 2, 4}, optionsA));
    const auto expectedB = Complete(RequestState::Create(referenceB.session, {6}, optionsB));
    auto requestA = RequestState::Create(a.session, {1, 2, 4}, optionsA);
    auto requestB = RequestState::Create(b.session, {6}, optionsB);
    for (int i = 0; i < 32; ++i) {
        if (!IsTerminal(requestA->Progress().status)) requestA->Step();
        if (!IsTerminal(requestB->Progress().status)) requestB->Step();
    }
    Expect(requestA->Snapshot().generatedIds == expectedA && requestB->Snapshot().generatedIds == expectedB &&
               a.state->samplingHistories == referenceA.state->samplingHistories &&
               b.state->samplingHistories == referenceB.state->samplingHistories,
           "Interleaved requests contaminated their thinking state or penalty history");
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


struct BatchedRun {
    std::vector<std::vector<int>> tokens;
    std::vector<size_t> terminals;
};

BatchedRun CompleteBatched(const std::vector<std::shared_ptr<RequestState>>& requests) {
    BatchedRun output{std::vector<std::vector<int>>(requests.size()), std::vector<size_t>(requests.size())};
    const auto observe = [&](size_t index, const StepResult& step) {
        if (step.tokenId) output.tokens[index].push_back(*step.tokenId);
        if (step.becameTerminal) ++output.terminals[index];
        Expect(!step.tokenId || !IsTerminal(step.status), "Batched terminal also published a token");
    };
    for (size_t turn = 0; turn < 128; ++turn) {
        std::vector<std::shared_ptr<RequestState>> decode;
        std::vector<size_t> indices;
        for (size_t i = 0; i < requests.size(); ++i) {
            auto status = requests[i]->Progress().status;
            if (status == RequestStatus::Queued || status == RequestStatus::Prefill) {
                observe(i, requests[i]->Step());
                status = requests[i]->Progress().status;
            }
            if (!IsTerminal(status)) {
                if (status == RequestStatus::Decode || status == RequestStatus::Finalizing) {
                    indices.push_back(i);
                    decode.push_back(requests[i]);
                }
            }
        }
        if (!decode.empty()) {
            const auto results = RequestState::StepDecodeBatch(decode);
            Expect(results.size() == decode.size(), "Decode results lost their row mapping");
            for (size_t i = 0; i < results.size(); ++i) observe(indices[i], results[i]);
        }
        if (std::all_of(requests.begin(), requests.end(), [](const auto& request) {
                return IsTerminal(request->Progress().status);
            })) {
            for (size_t i = 0; i < requests.size(); ++i) {
                Expect(requests[i]->Progress().status == RequestStatus::Completed && output.terminals[i] == 1 &&
                           output.tokens[i] == requests[i]->Snapshot().generatedIds,
                       "Decode batch lost output, failed, or produced more than one terminal");
                const auto repeated = RequestState::StepDecodeBatch({requests[i]});
                Expect(!repeated.front().becameTerminal && !repeated.front().tokenId,
                       "Repeated batched step reopened a terminal request");
            }
            return output;
        }
    }
    throw std::runtime_error("Decode batch did not finish within the CPU turn bound");
}

void TestDecodeBatchMatchesIndependentRows() {
    for (const size_t width : {size_t{1}, size_t{2}, size_t{4}}) {
        auto model = std::make_shared<FakeModel>();
        model->batchEnabled = true;
        std::vector<Fixture> fixtures;
        std::vector<Fixture> references;
        std::vector<std::shared_ptr<RequestState>> requests;
        std::vector<std::vector<int>> expected;
        for (size_t i = 0; i < width; ++i) {
            auto options = Options(static_cast<int>(i * 2 + 1), static_cast<int>(i + 1), 700 + i);
            options.generation.do_sample = i % 2 != 0;
            options.generation.repetition_penalty = 1.1f + static_cast<float>(i) * 0.1f;
            options.forceHostSampling = i % 2 != 0;
            options.legacyPrefillLogits = i >= 2;
            std::vector<int> prompt(i + 1, static_cast<int>(i + 2));
            fixtures.push_back(MakeFixture(128, 64, model));
            references.push_back(MakeFixture());
            expected.push_back(Complete(RequestState::Create(references.back().session, prompt, options)));
            requests.push_back(RequestState::Create(fixtures.back().session, prompt, options));
        }
        const auto output = CompleteBatched(requests);
        Expect(output.tokens == expected, "Batched independent RNG/configuration changed full output sequences");
        for (size_t i = 0; i < width; ++i) {
            Expect(fixtures[i].state->context == references[i].state->context &&
                       fixtures[i].state->samplingHistories == references[i].state->samplingHistories &&
                       fixtures[i].state->hostSampling == references[i].state->hostSampling &&
                       fixtures[i].state->legacyPrefillLogits == references[i].state->legacyPrefillLogits &&
                       !fixtures[i].session->Busy(),
                   "Batch rows contaminated cache/history/sampling policy or leaked a lease");
        }
        if (width == 1) {
            Expect(model->batchWidths.empty(), "B=1 invoked a changed batch execution path");
        } else {
            Expect(!model->batchWidths.empty() &&
                       *std::max_element(model->batchWidths.begin(), model->batchWidths.end()) <= width,
                   "Multiple ready rows never reached the shared batch entry");
        }
    }
}

void TestDecodeBatchThinkingStopAndFinalization() {
    auto model = std::make_shared<FakeModel>();
    model->batchEnabled = true;
    std::vector<RequestOptions> settings{ThinkingOptions(8, 2), ThinkingOptions(3, 2), Options(5), Options(1)};
    settings[0].forceHostSampling = true;
    settings[0].legacyPrefillLogits = true;
    settings[2].stopTokenIds = {22};
    std::vector<Fixture> fixtures, references;
    std::vector<std::shared_ptr<RequestState>> requests;
    std::vector<std::vector<int>> expected;
    for (size_t i = 0; i < settings.size(); ++i) {
        fixtures.push_back(MakeFixture(128, 64, model));
        references.push_back(MakeFixture());
        fixtures.back().state->samples = references.back().state->samples = {20, 21, 22, 23, 24, 25};
        const std::vector<int> prompt{static_cast<int>(i + 1)};
        expected.push_back(Complete(RequestState::Create(references.back().session, prompt, settings[i])));
        requests.push_back(RequestState::Create(fixtures.back().session, prompt, settings[i]));
    }
    const auto output = CompleteBatched(requests);
    Expect(output.tokens == expected && output.tokens[1] == std::vector<int>({20, 21}) &&
               output.tokens[2] == std::vector<int>({20, 21}),
           "Batch thinking close/early length/stop handling changed output");
    for (size_t i = 0; i < fixtures.size(); ++i) {
        Expect(fixtures[i].state->context == references[i].state->context &&
                   fixtures[i].state->samplingHistories == references[i].state->samplingHistories &&
                   fixtures[i].state->samplesTaken == references[i].state->samplesTaken,
               "Forced close, stop, or finalization sampled extra RNG or consumed wrong tokens");
    }
    Expect(requests[2]->Progress().endReason == RequestEndReason::StopToken &&
               requests[1]->Progress().endReason == RequestEndReason::Length,
           "Mixed batch rows lost independent terminal reasons");
    Expect(std::find(model->batchWidths.begin(), model->batchWidths.end(), size_t{4}) != model->batchWidths.end(),
           "Test did not combine decode with a finalizing row");
}

void TestDecodeBatchValidationAndCancelledRows() {
    auto model = std::make_shared<FakeModel>();
    model->batchEnabled = true;
    auto a = MakeFixture(128, 64, model), b = MakeFixture(128, 64, model);
    auto first = RequestState::Create(a.session, {1}, Options(3));
    auto second = RequestState::Create(b.session, {2}, Options(3));
    first->Step();
    const auto before = a.state->context;
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({first, nullptr}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({first, first}); });
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({first, second}); });
    Expect(a.state->context == before && b.state->context.empty() && a.session->Busy() && b.session->Busy(),
           "Invalid batch partially advanced a valid row or released its lease");
    second->Step();

    auto differentModel = std::make_shared<FakeModel>();
    differentModel->batchEnabled = true;
    auto c = MakeFixture(128, 64, differentModel);
    auto third = RequestState::Create(c.session, {3}, Options(3));
    third->Step();
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({first, third}); });
    auto unsupported = MakeFixture();
    auto noBatch = RequestState::Create(unsupported.session, {4}, Options(2));
    noBatch->Step();
    ExpectError(ErrorCode::InvalidArgument, [&] { RequestState::StepDecodeBatch({first, noBatch}); });

    first->Cancel();
    const auto secondBefore = b.state->context.size();
    const auto results = RequestState::StepDecodeBatch({first, second});
    Expect(results[0].status == RequestStatus::Cancelled && results[0].becameTerminal && !results[0].tokenId &&
               a.state->context == before && b.state->context.size() == secondBefore + 1 &&
               results[1].tokenId && model->batchWidths.empty(),
           "A cancelled row submitted pending work or prevented the B=1 peer fallback");
    const auto repeated = RequestState::StepDecodeBatch({first, second});
    Expect(!repeated[0].becameTerminal && repeated[0].status == RequestStatus::Cancelled &&
               repeated[1].tokenId,
           "Terminal row reopened or blocked its continuing peer");
    Complete(second);
    third->Cancel(); third->Step();
    noBatch->Cancel(); noBatch->Step();
}

void TestDecodeBatchConcurrentCancellationAndReentry() {
    auto model = std::make_shared<FakeModel>();
    model->batchEnabled = true;
    auto a = MakeFixture(128, 64, model), b = MakeFixture(128, 64, model), reference = MakeFixture();
    const auto optionsA = Options(4, 1, 100), optionsB = Options(5, 1, 200);
    const auto expectedB = Complete(RequestState::Create(reference.session, {2}, optionsB));
    auto first = RequestState::Create(a.session, {1}, optionsA);
    auto second = RequestState::Create(b.session, {2}, optionsB);
    first->Step(); second->Step();
    ForwardGate gate;
    model->duringBatch = [&] { gate.Enter(); };
    auto executing = std::async(std::launch::async, [&] {
        return RequestState::StepDecodeBatch({first, second});
    });
    std::future<RequestSnapshot> firstSnapshot, secondSnapshot;
    try {
        Expect(gate.AwaitEntry(), "Batch did not reach the held shared Forward");
        ExpectError(ErrorCode::Busy, [&] { first->Step(); });
        ExpectError(ErrorCode::Busy, [&] { second->Step(); });
        ExpectError(ErrorCode::Busy, [&] { RequestState::StepDecodeBatch({second, first}); });
        ExpectError(ErrorCode::Busy, [&] { b.session->Reset(); });
        firstSnapshot = std::async(std::launch::async, [first] { return first->Snapshot(); });
        secondSnapshot = std::async(std::launch::async, [second] { return second->Snapshot(); });
        Expect(firstSnapshot.wait_for(20ms) == std::future_status::timeout &&
                   secondSnapshot.wait_for(20ms) == std::future_status::timeout,
               "A batch snapshot observed mutable in-flight row state");
        first->Cancel();
    } catch (...) {
        gate.Release();
        executing.wait();
        if (firstSnapshot.valid()) firstSnapshot.wait();
        if (secondSnapshot.valid()) secondSnapshot.wait();
        throw;
    }
    gate.Release();
    const auto results = executing.get();
    model->duringBatch = {};
    Expect(results[0].status == RequestStatus::Cancelled && results[0].becameTerminal && !results[0].tokenId &&
               a.state->context.size() == 2 && !a.session->Busy(),
           "In-flight batch cancellation failed to drain input or suppressed the wrong row");
    Expect(firstSnapshot.get().status == RequestStatus::Cancelled &&
               secondSnapshot.get().status == results[1].status && results[1].tokenId,
           "Batch snapshots did not describe the completed boundary");
    Complete(second);
    Expect(second->Snapshot().generatedIds == expectedB && b.state->context == reference.state->context,
           "Cancelled peer/reentrant Step changed another row's RNG, history, or cache");
}

void TestDecodeBatchSharedFailuresAndInvalidResults() {
    for (int mode = 0; mode < 4; ++mode) {
        auto model = std::make_shared<FakeModel>();
        model->batchEnabled = true;
        auto a = MakeFixture(128, 64, model), b = MakeFixture(128, 64, model);
        auto first = RequestState::Create(a.session, {1}, Options(4));
        auto second = RequestState::Create(b.session, {2}, Options(4));
        first->Step(); second->Step();
        const auto firstHistory = first->Snapshot().generatedIds, secondHistory = second->Snapshot().generatedIds;
        if (mode == 0) model->ordinaryBatchFailure = true;
        if (mode == 1) model->batchFailure = ErrorCode::ResourceLimit;
        if (mode == 2) model->batchFailure = ErrorCode::RuntimeFault;
        if (mode == 3) model->invalidBatchResultCount = true;
        const auto result = RequestState::StepDecodeBatch({first, second});
        for (size_t row = 0; row < result.size(); ++row) {
            const auto& request = row == 0 ? first : second;
            Expect(result[row].status == RequestStatus::Failed && result[row].becameTerminal &&
                       !result[row].tokenId && !request->Progress().ErrorMessage().empty(),
                   "Shared batch failure did not terminate every submitted row once");
            const auto expectedCode = mode == 0 ? std::optional<ErrorCode>{}
                : mode == 1 ? std::optional<ErrorCode>{ErrorCode::ResourceLimit}
                            : std::optional<ErrorCode>{ErrorCode::RuntimeFault};
            Expect(request->Progress().errorCode == expectedCode,
                   "Batch failure lost ordinary/resource/runtime error classification");
        }
        Expect(!a.session->Busy() && !b.session->Busy() &&
                   first->Snapshot().generatedIds == firstHistory && second->Snapshot().generatedIds == secondHistory,
               "Failed batch leaked a lease or exposed a partially prepared sample");
        const auto repeated = RequestState::StepDecodeBatch({first, second});
        Expect(!repeated[0].becameTerminal && !repeated[1].becameTerminal,
               "Repeated failed batch published duplicate terminals");
    }
}

void TestDecodeBatchMalformedLastRowRejectsWholeGroup() {
    for (const bool invalidSample : {true, false}) {
        const std::string label = invalidSample ? "Last-row invalid sample: " : "Last-row incomplete consume: ";
        auto model = std::make_shared<FakeModel>();
        model->batchEnabled = true;
        std::vector<Fixture> fixtures;
        std::vector<std::shared_ptr<RequestState>> requests;
        std::vector<RequestSnapshot> before;
        for (size_t row = 0; row < MaxDecodeBatchSize; ++row) {
            fixtures.push_back(MakeFixture(128, 64, model));
            auto& fixture = fixtures.back();
            fixture.state->samples = {static_cast<int>(10 + row), static_cast<int>(20 + row)};
            requests.push_back(RequestState::Create(fixture.session, {static_cast<int>(row + 1)}, Options(4)));
            const auto prefill = requests.back()->Step();
            Expect(prefill.status == RequestStatus::Decode && prefill.tokenId == fixture.state->samples[0] &&
                       fixture.state->inputs.size() == 1 && fixture.state->samplesTaken == 1,
                   label + "fixture did not reach decode after exactly one prefill");
            before.push_back(requests.back()->Snapshot());
        }
        auto& last = *fixtures.back().state;
        if (invalidSample) last.samples[1] = last.vocabSize;
        else last.incompleteConsume = true;

        const auto result = RequestState::StepDecodeBatch(requests);
        Expect(result.size() == requests.size() && model->batchWidths == std::vector<size_t>{requests.size()},
               label + "call did not execute exactly one complete batch");
        for (size_t row = 0; row < requests.size(); ++row) {
            const auto& fixture = fixtures[row];
            const auto snapshot = requests[row]->Snapshot();
            const size_t expectedContext = !invalidSample && row + 1 == requests.size() ? 1 : 2;
            Expect(fixture.state->inputs.size() == 2 && fixture.state->samplesTaken == 2 &&
                       fixture.state->inputs.back() == std::vector<int>{before[row].generatedIds.back()} &&
                       fixture.state->context.size() == expectedContext,
                   label + "failure was not discovered after every row prepared its result");
            Expect(result[row].status == RequestStatus::Failed && result[row].becameTerminal && !result[row].tokenId &&
                       snapshot.status == RequestStatus::Failed && snapshot.endReason == RequestEndReason::RuntimeFault &&
                       snapshot.errorCode == ErrorCode::RuntimeFault && !snapshot.error.empty(),
                   label + "a submitted row escaped the shared RuntimeFault terminal");
            Expect(snapshot.generatedIds == before[row].generatedIds &&
                       snapshot.promptTokensProcessed == before[row].promptTokensProcessed && !fixture.session->Busy(),
                   label + "a prepared token was published or a failed request retained its lease");
        }

        last.incompleteConsume = false;
        last.samples.clear();
        // Keep the failed requests alive while new requests acquire every lease.
        // A repeated terminal step must not release a replacement's ownership.
        std::vector<std::shared_ptr<RequestState>> replacements;
        for (const auto& fixture : fixtures) {
            replacements.push_back(RequestState::Create(fixture.session, {7}, Options(1)));
        }
        const auto repeated = RequestState::StepDecodeBatch(requests);
        Expect(repeated.size() == requests.size() && model->batchWidths.size() == 1,
               label + "repeated terminal step submitted another batch");
        for (size_t row = 0; row < requests.size(); ++row) {
            const auto& fixture = fixtures[row];
            Expect(repeated[row].status == RequestStatus::Failed && !repeated[row].becameTerminal && !repeated[row].tokenId &&
                       requests[row]->Snapshot().generatedIds == before[row].generatedIds && fixture.session->Busy(),
                   label + "repeated failure published output, repeated a terminal, or released a replacement lease");
            replacements[row]->Cancel();
            const auto cancelled = replacements[row]->Step();
            Expect(cancelled.status == RequestStatus::Cancelled && cancelled.becameTerminal && !cancelled.tokenId &&
                       !fixture.session->Busy() && fixture.state->inputs.size() == 2 && fixture.state->samplesTaken == 2,
                   label + "replacement lease could not be released without extra backend work");
        }
    }
}

void TestDecodeBatchDeferredCallbacksRetainEveryRow() {
    auto model = std::make_shared<FakeModel>();
    model->batchEnabled = true;
    auto a = MakeFixture(128, 64, model), b = MakeFixture(128, 64, model);
    std::vector<std::shared_ptr<RequestState>> requests{
        RequestState::Create(a.session, {1}, Options(3)),
        RequestState::Create(b.session, {2}, Options(3))};
    requests[0]->Step(); requests[1]->Step();
    struct Probe {
        std::vector<std::shared_ptr<RequestState>>* requests;
        std::vector<std::weak_ptr<RequestState>> weak;
        size_t calls = 0;
        size_t retained = 0;
        size_t busy = 0;
        size_t snapshots = 0;
    } probe{&requests, {requests[0], requests[1]}};
    struct RestoreLogging {
        aila::LogLevel level = aila::get_log_level();
        ~RestoreLogging() {
            aila::set_log_callback(nullptr);
            aila::set_log_level(level);
        }
    } restore;
    aila::set_log_level(aila::LogLevel::Info);
    aila::set_log_callback([](int, const char* message, void* userData) {
        if (std::strcmp(message, "RequestStateTests deferred batch") != 0) return;
        auto& probe = *static_cast<Probe*>(userData);
        ++probe.calls;
        probe.requests->clear();
        for (const auto& weak : probe.weak) {
            if (auto request = weak.lock()) {
                ++probe.retained;
                if (request->Snapshot().status == RequestStatus::Decode) ++probe.snapshots;
                try { request->Step(); }
                catch (const Error& error) { if (error.Code() == ErrorCode::Busy) ++probe.busy; }
            }
        }
    }, &probe);
    model->duringBatch = [] { AILA_LOG_INFO("RequestStateTests deferred batch"); };
    const auto result = RequestState::StepDecodeBatch(requests);
    Expect(result.size() == 2 && result[0].tokenId && result[1].tokenId && probe.calls == 1 &&
               probe.retained == 2 && probe.busy == 2 && probe.snapshots == 2 &&
               probe.weak[0].expired() && probe.weak[1].expired() && !a.session->Busy() && !b.session->Busy(),
           "Batch callback observed locks, lost ownership, or reentered a completed step guard");
}

void TestSessionBatchLogitsAndAdmission() {
    auto model = std::make_shared<FakeModel>();
    model->batchEnabled = true;
    auto a = MakeFixture(8, 64, model), b = MakeFixture(8, 64, model);
    auto refA = MakeFixture(8), refB = MakeFixture(8);
    a.session->ForwardLogits({1, 2}); refA.session->ForwardLogits({1, 2});
    b.session->ForwardLogits({3}); refB.session->ForwardLogits({3});
    const auto result = SessionState::ForwardDecodeBatchLogits({b.session, a.session}, {5, 4});
    Expect(result == std::vector<std::vector<float>>({refB.session->ForwardLogits({5}), refA.session->ForwardLogits({4})}) &&
               a.state->context == refA.state->context && b.state->context == refB.state->context,
           "Batch diagnostic logits did not preserve independent positions and input row order");
    const auto calls = model->batchWidths.size();
    const auto beforeA = a.state->context, beforeB = b.state->context;
    ExpectError(ErrorCode::InvalidArgument, [&] {
        SessionState::ForwardDecodeBatchLogits({a.session, a.session}, {1, 2});
    });
    ExpectError(ErrorCode::InvalidArgument, [&] {
        SessionState::ForwardDecodeBatchLogits({a.session, b.session}, {1, 64});
    });
    auto busy = RequestState::Create(b.session, {1}, Options(1));
    ExpectError(ErrorCode::Busy, [&] {
        SessionState::ForwardDecodeBatchLogits({a.session, b.session}, {1, 2});
    });
    Expect(!a.session->Busy() && b.session->Busy() && a.state->context == beforeA && b.state->context == beforeB &&
               model->batchWidths.size() == calls,
           "Rejected batch leaked an earlier lease or consumed a valid row before validation finished");
    busy->Cancel(); busy->Step();
    b.session->ForwardLogits(std::vector<int>(6, 1));
    ExpectError(ErrorCode::ContextOverflow, [&] {
        SessionState::ForwardDecodeBatchLogits({a.session, b.session}, {1, 2});
    });
    Expect(a.state->context == beforeA && b.state->context.size() == 8 && !a.session->Busy() && !b.session->Busy(),
           "Capacity rejection partially advanced the other diagnostic row");
    const auto scalar = SessionState::ForwardDecodeBatchLogits({a.session}, {6});
    Expect(scalar.front() == refA.session->ForwardLogits({6}) && model->batchWidths.size() == calls,
           "Diagnostic B=1 did not use the original scalar Forward path");
}

void TestDecodeWindowCommit() {
    for (bool accepts : {false, true}) for (int maximum : {1, 2, 3, 4, 5}) {
        auto f = MakeFixture(); f.state->windowEnabled = true; f.state->windowAccepts = accepts;
        f.state->samples = {10, 11, 12, 13, 14, 15};
        auto options = Options(maximum); options.generation.do_sample = false;
        auto request = RequestState::Create(f.session, {1, 2}, options);
        const auto generated = Complete(request);
        Expect(generated == std::vector<int>(f.state->samples.begin(), f.state->samples.begin() + maximum), "Window output order");
        std::vector<int> expected{1, 2}; expected.insert(expected.end(), generated.begin(), generated.end());
        Expect(f.state->context == expected, "Window consumed a token twice or retained a suffix");
        Expect((f.state->windows == 0) == (maximum < 3), "Window used at short output tail");
    }
    for (int stop : {11, 12}) {
        auto f = MakeFixture(); f.state->windowEnabled = true; f.state->samples = {10, 11, 12, 13};
        auto options = Options(4); options.generation.do_sample = false; options.stopTokenIds = {stop};
        auto request = RequestState::Create(f.session, {1, 2}, options);
        const auto generated = Complete(request);
        Expect(generated.size() == size_t(stop - 10), "Window stop publication");
        std::vector<int> expected{1, 2}; expected.insert(expected.end(), generated.begin(), generated.end());
        Expect(f.state->context == expected, "Window stop cached an unpublished token");
    }
    for (bool during : {false, true}) {
        auto f = MakeFixture(); f.state->windowEnabled = true; f.state->samples = {10, 11, 12, 13};
        auto options = Options(4); options.generation.do_sample = false;
        auto request = RequestState::Create(f.session, {1, 2}, options);
        request->Step();
        if (during) f.state->duringForward = [&] { request->Cancel(); };
        request->Step(); request->Cancel(); request->Step();
        Expect(f.state->context.empty() && request->Progress().status == RequestStatus::Cancelled, "Cancelled MTP session not reset");
    }
    auto f = MakeFixture(); f.state->windowEnabled = true; f.state->samples = {10, 11, 12, 13};
    auto options = Options(4); options.generation.do_sample = false;
    auto request = RequestState::Create(f.session, {1, 2}, options);
    request->Step(); request->Step(1);
    Expect(f.state->windows == 0, "Window ignored output backpressure budget");
}

void TestPrefillChunkPhaseRemaining() {
    // Each prompt chunk reports the queued prompt suffix; 0 marks the final
    // slice so a phase-aware backend can close prefill. Decode and the final
    // pending-token flush stay on the plain entry point.
    auto chunked = MakeFixture();
    auto options = Options(2, 2);
    options.generation.do_sample = false;
    chunked.state->samples = {10, 11, 12, 13};
    Complete(RequestState::Create(chunked.session, {1, 2, 3, 4, 5}, options));
    Expect(chunked.state->prefillRemaining == std::vector<int>({3, 1, 0}),
           "Prefill chunks did not report the remaining prompt count");
    Expect(chunked.state->prefillLookahead == std::vector<std::vector<int>>({{3, 4}, {5}, {}}),
           "Lookahead must contain exactly the next chunk without consuming it");
    Expect(chunked.state->prefillForwards == 3,
           "Decode or finalize flush used the prompt-phase entry point");
    Expect(chunked.state->inputs[0] == std::vector<int>({1, 2}) &&
               chunked.state->inputs[1] == std::vector<int>({3, 4}) &&
               chunked.state->inputs[2] == std::vector<int>({5}),
           "Prefill chunk boundaries shifted");

    auto single = MakeFixture();
    auto whole = Options(2, 8);
    whole.generation.do_sample = false;
    single.state->samples = {10, 11, 12, 13};
    Complete(RequestState::Create(single.session, {1, 2, 3, 4, 5}, whole));
    Expect(single.state->prefillRemaining == std::vector<int>({0}) &&
               single.state->prefillForwards == 1,
           "Single-chunk prefill did not report remaining zero once");
    Expect(single.state->prefillLookahead == std::vector<std::vector<int>>({{}}),
           "A final prefill chunk must not carry lookahead");

    auto replay = MakeFixture();
    auto prefillOnly = Options(0, 2);
    prefillOnly.prefillOnly = true;
    Complete(RequestState::Create(replay.session, {1, 2, 3, 4, 5}, prefillOnly));
    Expect(replay.state->prefillRemaining == std::vector<int>({3, 1, 0}) &&
               replay.state->samplesTaken == 0,
           "Prefill-only replay skipped or misordered its phase values");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, void (*)()>> tests{
        {"prefill chunk remaining phase metadata", TestPrefillChunkPhaseRemaining},
        {"decode window prefix, stop, tail and cancellation", TestDecodeWindowCommit},
        {"session mutations and actual truncate length", TestSessionMutations},
        {"prefill without sampling, capacity and cancelled replay", TestPrefillOnlyAndCancelledReplay},
        {"thinking budget closes, accounts and continues", TestThinkingBudgetClosesAndAccounts},
        {"single reasoning close and explicit host sampler policy", TestSingleCloseAndHostSamplingPolicy},
        {"thinking zero budget, natural close and reopen", TestThinkingBudgetZeroNaturalAndReopened},
        {"thinking close length boundary and cancellation", TestThinkingBudgetLengthAndCancellation},
        {"thinking validation and request isolation", TestThinkingBudgetValidationAndIsolation},
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
        {"decode batches preserve independent rows and B=1", TestDecodeBatchMatchesIndependentRows},
        {"batch thinking, stop and finalization remain row-local", TestDecodeBatchThinkingStopAndFinalization},
        {"batch validation and cancelled-row B=1 fallback", TestDecodeBatchValidationAndCancelledRows},
        {"batch in-flight cancellation, snapshots and reentry", TestDecodeBatchConcurrentCancellationAndReentry},
        {"batch failures and invalid result count retire all rows", TestDecodeBatchSharedFailuresAndInvalidResults},
        {"malformed last batch row suppresses every prepared token", TestDecodeBatchMalformedLastRowRejectsWholeGroup},
        {"deferred batch callbacks retain all rows and guards", TestDecodeBatchDeferredCallbacksRetainEveryRow},
        {"batch logits row order and transactional admission", TestSessionBatchLogitsAndAdmission},
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
