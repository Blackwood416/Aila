#include "inference/Model.hpp"
#include "inference/Scheduler.hpp"
#include "profile/Profiling.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int>;

constexpr int kCapacity = 512;
constexpr int kGenerated = 16;
constexpr int kContinuationGenerated = 8;
constexpr auto kRequestTimeout = 120s;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_error(ErrorCode code, Function&& operation, const std::string& label) {
    try {
        operation();
    } catch (const Error& error) {
        expect(error.Code() == code, label + ": wrong error code: " + error.what());
        return;
    }
    throw std::runtime_error(label + ": expected an inference error");
}

class ScopedLogLevel final {
public:
    explicit ScopedLogLevel(aila::LogLevel level) : previous(aila::get_log_level()) {
        aila::set_log_level(level);
    }
    ~ScopedLogLevel() { aila::set_log_level(previous); }
    ScopedLogLevel(const ScopedLogLevel&) = delete;
    ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;

private:
    aila::LogLevel previous;
};

// Each use deliberately grows the execution slot beyond its previous shape.
// Request::Step flushes this log after releasing its device/session locks, so
// the test can enqueue other real sessions before the worker's next quantum.
// The gate owns no model/session/request handles and does no GPU work.
class PrefillGate final {
public:
    PrefillGate() : previousLevel(aila::get_log_level()) {
        aila::set_log_level(aila::LogLevel::Info);
        aila::set_log_callback(&Callback, this);
    }
    ~PrefillGate() {
        Release();
        // This standalone fixture installed no earlier custom callback.
        aila::set_log_callback(nullptr, nullptr);
        aila::set_log_level(previousLevel);
    }
    PrefillGate(const PrefillGate&) = delete;
    PrefillGate& operator=(const PrefillGate&) = delete;

    bool AwaitEntry() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, kRequestTimeout, [&] { return entered; });
    }
    void Release() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        changed.notify_all();
    }
    void Check() const {
        std::lock_guard<std::mutex> lock(mutex);
        expect(entered && !timedOut, "Prefill gate was not entered or timed out");
    }

private:
    static void Callback(int, const char* message, void* userData) noexcept {
        if (!message || !std::strstr(message, "[Qwen3Bnb4] Runtime buffers resized")) return;
        auto& gate = *static_cast<PrefillGate*>(userData);
        std::unique_lock<std::mutex> lock(gate.mutex);
        if (gate.entered) return;
        gate.entered = true;
        gate.changed.notify_all();
        if (!gate.changed.wait_for(lock, kRequestTimeout, [&] { return gate.released; })) {
            gate.timedOut = true;
        }
    }

    aila::LogLevel previousLevel;
    mutable std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
    bool timedOut = false;
};

// Declare after Scheduler, so an assertion/exception opens the gate before
// Scheduler's destructor joins a worker that could still be inside Callback.
class ReleaseGateOnExit final {
public:
    explicit ReleaseGateOnExit(PrefillGate& value) : gate(value) {}
    ~ReleaseGateOnExit() { gate.Release(); }

private:
    PrefillGate& gate;
};

Tokens repeat_tokens(const Tokens& source, size_t count) {
    expect(!source.empty(), "Tokenizer returned an empty fixture");
    Tokens result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(source[i % source.size()]);
    return result;
}

RequestOptions request_options(bool sampling, bool penalties, uint64_t seed,
                               int chunkSize, int generated = kGenerated) {
    RequestOptions options;
    options.prefillChunkSize = chunkSize;
    options.generation.max_new_tokens = generated;
    options.generation.do_sample = sampling;
    options.generation.use_fixed_seed = true;
    options.generation.sampling_seed = seed;
    options.generation.temperature = 0.8f;
    options.generation.top_k = 32;
    options.generation.top_p = 0.9f;
    if (penalties) {
        options.generation.repetition_penalty = 1.15f;
        options.generation.presence_penalty = 0.2f;
        options.generation.frequency_penalty = 0.1f;
    }
    // No stop IDs: every successful request must finalize its exact length,
    // even if a sampled token happens to be a model EOS token.
    return options;
}

SchedulerOptions scheduler_options(size_t active, size_t ringCapacity, int chunkSize) {
    SchedulerOptions options;
    options.maxActiveRequests = active;
    options.maxOutstandingRequests = active;
    options.maxPromptTokens = kCapacity;
    options.maxGeneratedTokens = 128;
    options.maxOutstandingTokens = active * kCapacity;
    options.maxPrefillChunkSize = chunkSize;
    options.maxBufferedTokensPerRequest = ringCapacity;
    return options;
}

void expect_idle(const Scheduler& scheduler, const std::string& label) {
    const auto snapshot = scheduler.Snapshot();
    expect(!snapshot.faulted && snapshot.outstandingRequests == 0 &&
               snapshot.queuedRequests == 0 && snapshot.activeRequests == 0 &&
               snapshot.reservedTokens == 0 && snapshot.reservedSessionCacheBytes == 0,
           label + ": requests or reservations remain after shutdown/consumption");
}

struct StreamObservation {
    Tokens tokens;
    RequestEvent terminal;
    bool ended = false;
    Clock::time_point submitted{};
    Clock::time_point terminalRead{};
    std::vector<double> tokenReadMs;
};

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

bool read_one(const std::shared_ptr<ScheduledRequest>& handle, StreamObservation& observed,
              std::chrono::milliseconds timeout, const std::string& label,
              bool recordTiming = false) {
    auto read = handle->Read(timeout);
    const auto consumedAt = Clock::now();
    if (read.status == ReadStatus::Timeout) return false;
    expect(read.status == ReadStatus::Event && read.event.has_value(),
           label + ": stream closed before its terminal event");
    const auto& event = *read.event;
    expect(!observed.ended && event.requestId == handle->Id() &&
               event.sequence == observed.tokens.size() + 1,
           label + ": duplicate event or invalid stream routing/sequence");
    if (event.kind == RequestEventKind::Token) {
        expect(event.tokenId >= 0 && !IsTerminal(event.status) &&
                   event.endReason == RequestEndReason::None && !event.errorCode && event.error.empty(),
               label + ": malformed token event");
        observed.tokens.push_back(event.tokenId);
        expect(event.generatedTokens == observed.tokens.size(), label + ": wrong token event count");
        if (recordTiming) observed.tokenReadMs.push_back(milliseconds(consumedAt - observed.submitted));
    } else {
        expect(IsTerminal(event.status) && event.generatedTokens == observed.tokens.size() &&
                   event.tokenId == -1 && event.error.size() <= 1024,
               label + ": malformed terminal event");
        observed.terminal = std::move(*read.event);
        observed.terminalRead = consumedAt;
        observed.ended = true;
    }
    return true;
}

void expect_closed(const std::shared_ptr<ScheduledRequest>& handle, const std::string& label) {
    expect(handle->Read(0ms).status == ReadStatus::Closed,
           label + ": terminal event was published more than once");
    const auto snapshot = handle->Snapshot();
    expect(snapshot.terminalConsumed && snapshot.bufferedTokens == 0,
           label + ": consumed stream retained buffered tokens");
}

std::vector<StreamObservation> drain_all(
    const std::vector<std::shared_ptr<ScheduledRequest>>& handles, const std::string& label) {
    std::vector<StreamObservation> observations(handles.size());
    const auto deadline = Clock::now() + kRequestTimeout;
    size_t remaining = handles.size();
    while (remaining != 0) {
        bool progressed = false;
        for (size_t i = 0; i < handles.size(); ++i) {
            if (observations[i].ended) continue;
            progressed |= read_one(handles[i], observations[i], 0ms, label + " " + std::to_string(i));
            if (observations[i].ended) --remaining;
        }
        expect(Clock::now() < deadline, label + ": timed out waiting for request completion");
        if (!progressed && remaining != 0) {
            for (size_t i = 0; i < handles.size(); ++i) {
                if (observations[i].ended) continue;
                read_one(handles[i], observations[i], 10ms, label + " " + std::to_string(i));
                if (observations[i].ended) --remaining;
                break;
            }
        }
    }
    for (size_t i = 0; i < handles.size(); ++i) expect_closed(handles[i], label);
    return observations;
}

StreamObservation drain_timed(const std::shared_ptr<ScheduledRequest>& handle,
                              Clock::time_point submitted) {
    StreamObservation observed;
    observed.submitted = submitted;
    observed.tokens.reserve(128);
    observed.tokenReadMs.reserve(128);
    const auto deadline = submitted + kRequestTimeout;
    while (!observed.ended) {
        read_one(handle, observed, 100ms, "Benchmark stream", true);
        expect(Clock::now() < deadline, "Benchmark stream timed out");
    }
    expect_closed(handle, "Benchmark stream");
    return observed;
}

void expect_tokens(const Tokens& actual, const Tokens& expected, const std::string& label) {
    if (actual == expected) return;
    size_t index = 0;
    while (index < actual.size() && index < expected.size() && actual[index] == expected[index]) ++index;
    throw std::runtime_error(label + ": token mismatch at " + std::to_string(index) +
                             " (actual size " + std::to_string(actual.size()) +
                             ", expected size " + std::to_string(expected.size()) + ")");
}

void expect_completed(const StreamObservation& observed,
                      const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                      const RequestOptions& options, int originalLength, const std::string& label) {
    expect(observed.ended && observed.terminal.status == RequestStatus::Completed &&
               observed.terminal.endReason == RequestEndReason::Length &&
               !observed.terminal.errorCode && observed.terminal.error.empty(),
           label + ": did not complete successfully: " + observed.terminal.error);
    expect(observed.tokens.size() == static_cast<size_t>(options.generation.max_new_tokens) &&
               observed.terminal.promptTokensProcessed == prompt.size(),
           label + ": incomplete prompt or generation");
    expect(!session->Busy() && session->ContextLength() ==
               originalLength + static_cast<int>(prompt.size() + observed.tokens.size()),
           label + ": final token was not committed or session lease remains held");
}

Tokens run_independent(const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                       const RequestOptions& options, const std::string& label) {
    const int originalLength = session->ContextLength();
    auto request = RequestState::Create(session, prompt, options);
    Tokens output;
    int terminalEvents = 0;
    for (int i = 0; i < 2 * kCapacity; ++i) {
        const auto step = request->Step();
        if (step.tokenId) output.push_back(*step.tokenId);
        if (step.becameTerminal) ++terminalEvents;
        if (IsTerminal(step.status)) break;
    }
    const auto snapshot = request->Snapshot();
    expect(snapshot.status == RequestStatus::Completed && terminalEvents == 1 && snapshot.error.empty(),
           label + ": independent request failed: " + snapshot.error);
    expect(output.size() == static_cast<size_t>(options.generation.max_new_tokens) &&
               snapshot.generatedIds == output && snapshot.promptTokensProcessed == prompt.size(),
           label + ": independent request returned an incomplete trace");
    expect(!session->Busy() && session->ContextLength() ==
               originalLength + static_cast<int>(prompt.size() + output.size()),
           label + ": independent request left an invalid cache/lease");
    return output;
}

void expect_logits(const std::vector<float>& actual, const std::vector<float>& expected,
                   const std::string& label) {
    expect(actual.size() == expected.size() && !actual.empty(), label + ": invalid logits dimensions");
    for (size_t i = 0; i < actual.size(); ++i) {
        // BF16 logits convert losslessly to float. Identical per-session step
        // shapes must produce identical values even when other sessions run.
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
            std::ostringstream message;
            message << label << ": logit " << i << " differs (" << actual[i]
                    << " versus " << expected[i] << ")";
            throw std::runtime_error(message.str());
        }
    }
}

struct TestCase {
    Tokens prompt;
    Tokens continuation;
    Tokens probe;
    RequestOptions initialOptions;
    RequestOptions continuationOptions;
};

std::array<TestCase, 4> make_cases(Tokenizer& tokenizer) {
    const std::array<const char*, 4> texts{{
        "Explain how a small robot sorts red and blue blocks. ",
        "A gardener records rainfall and the growth of young trees each morning. ",
        "Describe a careful experiment comparing two ways to store solar energy. ",
        "A librarian maps a collection of historical travel diaries to their cities. "
    }};
    const std::array<size_t, 4> lengths{{192, 144, 65, 257}};
    const std::array<int, 4> chunks{{96, 48, 16, 128}};
    const std::array<size_t, 4> continuationLengths{{11, 19, 7, 23}};
    const std::array<int, 4> continuationChunks{{5, 7, 3, 11}};
    const std::array<uint64_t, 4> seeds{{19, 113, 277, 691}};
    const auto continuation = tokenizer.encode(" Continue with a concrete example and explain the result. ");
    const auto probe = repeat_tokens(tokenizer.encode(" Therefore"), 1);
    std::array<TestCase, 4> cases;
    for (size_t i = 0; i < cases.size(); ++i) {
        cases[i].prompt = repeat_tokens(tokenizer.encode(texts[i]), lengths[i]);
        cases[i].continuation = repeat_tokens(continuation, continuationLengths[i]);
        cases[i].probe = probe;
        cases[i].initialOptions = request_options(i != 0, i == 2, seeds[i], chunks[i]);
        cases[i].continuationOptions = request_options(
            i != 0, i == 2, seeds[i] + 1000, continuationChunks[i], kContinuationGenerated);
    }
    return cases;
}

struct References {
    std::array<Tokens, 4> initial;
    size_t cacheBytes = 0;
};

References test_four_sessions(const std::shared_ptr<DeviceRuntime>& runtime,
                             const std::shared_ptr<LoadedModel>& model,
                             const std::array<TestCase, 4>& cases) {
    std::cout << "[RUN] Four scheduled sessions versus independent greedy/seed/penalty requests" << std::endl;
    const size_t residentBefore = runtime->Memory().currentBytes;
    std::array<std::shared_ptr<SessionState>, 4> sessions;
    References references;
    for (auto& session : sessions) {
        session = model->CreateSession();
        if (!references.cacheBytes) references.cacheBytes = session->CacheBytes();
        expect(session->CacheBytes() == references.cacheBytes &&
                   session->CacheBytes() == model->SessionReservationBytes(), "Unexpected per-session KV size");
    }
    expect(model->ActiveSessions() == sessions.size() &&
               runtime->Memory().currentBytes == residentBefore + sessions.size() * references.cacheBytes,
           "Four sessions duplicated model/workspace resources");

    PrefillGate gate;
    Scheduler scheduler(scheduler_options(4, 64, 128));
    ReleaseGateOnExit releaseGate(gate);
    std::vector<std::shared_ptr<ScheduledRequest>> handles;
    handles.push_back(scheduler.Submit(sessions[0], cases[0].prompt, cases[0].initialOptions));
    expect(gate.AwaitEntry(), "First real prefill did not enter its gate");
    for (size_t i = 1; i < sessions.size(); ++i) {
        handles.push_back(scheduler.Submit(sessions[i], cases[i].prompt, cases[i].initialOptions));
    }
    const auto queued = scheduler.Snapshot();
    expect(queued.outstandingRequests == 4 && queued.activeRequests == 1 && queued.queuedRequests == 3,
           "Fixture failed to queue four sessions before resuming the first request");
    gate.Release();
    const auto initial = drain_all(handles, "Four initial streams");
    for (size_t i = 0; i < sessions.size(); ++i) {
        expect_completed(initial[i], sessions[i], cases[i].prompt, cases[i].initialOptions, 0,
                         "Initial session " + std::to_string(i));
    }
    handles.clear();
    for (size_t i = 0; i < sessions.size(); ++i) {
        handles.push_back(scheduler.Submit(sessions[i], cases[i].continuation, cases[i].continuationOptions));
    }
    const auto continued = drain_all(handles, "Four appended streams");
    for (size_t i = 0; i < sessions.size(); ++i) {
        expect_completed(continued[i], sessions[i], cases[i].continuation, cases[i].continuationOptions,
                         static_cast<int>(cases[i].prompt.size()) + kGenerated,
                         "Continued session " + std::to_string(i));
    }
    scheduler.Shutdown();
    expect_idle(scheduler, "Four-session scheduler");
    gate.Check();

    std::array<std::vector<float>, 4> scheduledLogits;
    for (size_t i = 0; i < sessions.size(); ++i) {
        scheduledLogits[i] = sessions[i]->ForwardLogits(cases[i].probe);
        expect(scheduledLogits[i].size() == static_cast<size_t>(model->VocabSize()), "Invalid probe logits size");
    }
    for (size_t i = 0; i < sessions.size(); ++i) {
        const auto label = "Independent session " + std::to_string(i);
        sessions[i]->Reset();
        references.initial[i] = run_independent(sessions[i], cases[i].prompt, cases[i].initialOptions, label);
        expect_tokens(initial[i].tokens, references.initial[i], label + " initial output");
        const auto appended = run_independent(sessions[i], cases[i].continuation, cases[i].continuationOptions, label);
        expect_tokens(continued[i].tokens, appended, label + " appended output");
        expect_logits(scheduledLogits[i], sessions[i]->ForwardLogits(cases[i].probe), label + " appended cache");
    }
    return references;
}

void test_cancel_or_drop(const std::shared_ptr<DeviceRuntime>& runtime,
                         const std::shared_ptr<LoadedModel>& model,
                         const std::array<TestCase, 4>& cases, const References& references,
                         bool drop, int firstChunk) {
    const std::string label = drop ? "Drop A while B continues" : "Cancel A while B continues";
    std::cout << "[RUN] " << label << std::endl;
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const std::weak_ptr<SessionState> weakA = a;
    const auto aPrompt = repeat_tokens(cases[0].prompt, static_cast<size_t>(firstChunk + 64));
    const auto aOptions = request_options(true, true, 441, firstChunk, 64);
    PrefillGate gate;
    Scheduler scheduler(scheduler_options(2, 128, firstChunk));
    ReleaseGateOnExit releaseGate(gate);
    auto first = scheduler.Submit(a, aPrompt, aOptions);
    expect(gate.AwaitEntry(), label + ": real prefill did not reach its gate");
    auto second = scheduler.Submit(b, cases[1].prompt, cases[1].initialOptions);
    const auto memoryAtGate = runtime->Memory();
    if (drop) first.reset();
    else first->Cancel();
    a.reset();
    expect(!weakA.expired() && model->ActiveSessions() == 2 &&
               scheduler.Snapshot().outstandingRequests == 2,
           label + ": live execution resources/reservation were released inside the step");
    gate.Release();

    if (drop) {
        const auto observed = drain_all({second}, label);
        expect_completed(observed[0], b, cases[1].prompt, cases[1].initialOptions, 0, label + " B");
        expect_tokens(observed[0].tokens, references.initial[1], label + " B output");
    } else {
        const auto observed = drain_all({first, second}, label);
        expect(observed[0].terminal.status == RequestStatus::Cancelled &&
                   observed[0].terminal.endReason == RequestEndReason::Cancelled &&
                   observed[0].terminal.promptTokensProcessed == static_cast<size_t>(firstChunk) &&
                   observed[0].tokens.empty(), label + ": cancellation submitted more work or lost progress");
        expect_completed(observed[1], b, cases[1].prompt, cases[1].initialOptions, 0, label + " B");
        expect_tokens(observed[1].tokens, references.initial[1], label + " B output");
    }
    scheduler.Shutdown();
    gate.Check();
    expect_idle(scheduler, label);
    expect(weakA.expired() && model->ActiveSessions() == 1 &&
               runtime->Memory().currentBytes == memoryAtGate.currentBytes - references.cacheBytes,
           label + ": A did not release exactly its KV cache after execution ended");
}

void test_output_backpressure(const std::shared_ptr<LoadedModel>& model,
                              const std::array<TestCase, 4>& cases, const References& references) {
    std::cout << "[RUN] A full output ring cancels only its slow consumer and preserves terminal delivery" << std::endl;
    constexpr size_t ringSize = 4;
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const auto aPrompt = repeat_tokens(cases[0].prompt, 384);
    const auto aOptions = request_options(false, false, 41, 320);
    auto bOptions = cases[1].initialOptions;
    bOptions.generation.max_new_tokens = static_cast<int>(ringSize);
    PrefillGate gate;
    Scheduler scheduler(scheduler_options(2, ringSize, 320));
    ReleaseGateOnExit releaseGate(gate);
    auto slow = scheduler.Submit(a, aPrompt, aOptions);
    expect(gate.AwaitEntry(), "Backpressure fixture did not reach its real prefill gate");
    auto fast = scheduler.Submit(b, cases[1].prompt, bOptions);
    gate.Release();
    // B fits exactly in its ring, including successful Finalizing. This
    // avoids turning a delayed test-reader thread into a spurious B failure.
    const auto fastTrace = drain_all({fast}, "Unaffected B during backpressure");
    expect_completed(fastTrace[0], b, cases[1].prompt, bOptions, 0, "Backpressure B");
    expect_tokens(fastTrace[0].tokens,
                  Tokens(references.initial[1].begin(), references.initial[1].begin() + ringSize),
                  "Backpressure B output");
    const auto deadline = Clock::now() + kRequestTimeout;
    while (!IsTerminal(slow->Snapshot().status)) {
        expect(Clock::now() < deadline, "Slow stream did not reach a terminal state");
        // A timed read on the already consumed fast stream is not a wait, so
        // use a small host delay solely for this terminal-state observation.
        std::this_thread::sleep_for(1ms);
    }
    const auto beforeRead = slow->Snapshot();
    expect(beforeRead.status == RequestStatus::Cancelled &&
               beforeRead.endReason == RequestEndReason::OutputBackpressure &&
               beforeRead.errorCode == ErrorCode::ResourceLimit &&
               beforeRead.bufferedTokens == ringSize && !beforeRead.terminalConsumed,
           "Full output ring did not preserve its bounded tokens and separate terminal");
    expect(!a->Busy() && a->ContextLength() == static_cast<int>(aPrompt.size() + ringSize - 1),
           "Backpressure executed another token-producing/finalizing step");
    expect(scheduler.Snapshot().outstandingRequests == 1,
           "An unread terminal released its stream reservation early");
    const auto slowTrace = drain_all({slow}, "Slow terminal after backpressure");
    expect(slowTrace[0].tokens.size() == ringSize &&
               slowTrace[0].terminal.endReason == RequestEndReason::OutputBackpressure,
           "Slow stream lost buffered tokens or its terminal event");
    scheduler.Shutdown();
    gate.Check();
    expect_idle(scheduler, "Backpressure scheduler");
}

void test_repeated_reclamation(const std::shared_ptr<DeviceRuntime>& runtime,
                               const std::shared_ptr<LoadedModel>& model,
                               const Tokens& source, size_t cacheBytes) {
    std::cout << "[RUN] Repeated scheduled request/session destruction returns all KV allocations" << std::endl;
    expect(model->ActiveSessions() == 0, "Earlier scheduler fixture retained a session");
    const auto resident = runtime->Memory().currentBytes;
    for (int iteration = 0; iteration < 3; ++iteration) {
        std::array<std::shared_ptr<SessionState>, 4> sessions;
        std::array<std::weak_ptr<SessionState>, 4> weakSessions;
        for (size_t i = 0; i < sessions.size(); ++i) {
            sessions[i] = model->CreateSession();
            weakSessions[i] = sessions[i];
        }
        expect(model->ActiveSessions() == 4 && runtime->Memory().currentBytes == resident + 4 * cacheBytes,
               "Session recreation duplicated shared resources");
        const auto atLimit = runtime->Memory();
        expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Four-session count limit");
        const auto afterRejection = runtime->Memory();
        expect(afterRejection.currentBytes == atLimit.currentBytes && afterRejection.peakBytes == atLimit.peakBytes,
               "Rejected fifth session allocated device memory");
        Scheduler scheduler(scheduler_options(4, 4, 8));
        std::vector<std::shared_ptr<ScheduledRequest>> handles;
        for (size_t i = 0; i < sessions.size(); ++i) {
            handles.push_back(scheduler.Submit(sessions[i], repeat_tokens(source, 16),
                                              request_options(false, false, i, 8, 2)));
            sessions[i].reset();
        }
        const auto observed = drain_all(handles, "Repeated reclamation");
        for (const auto& trace : observed) {
            expect(trace.terminal.status == RequestStatus::Completed && trace.tokens.size() == 2,
                   "Reclamation request did not complete");
        }
        scheduler.Shutdown();
        expect_idle(scheduler, "Reclamation scheduler");
        for (const auto& session : weakSessions) expect(session.expired(), "Terminal stream retained its session");
        expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == resident,
               "Repeated scheduler/session teardown leaked tracked KV or other device allocations");
    }
}

void test_model_cache_budget(const std::shared_ptr<DeviceRuntime>& runtime,
                             const std::string& modelDirectory, size_t cacheBytes,
                             size_t emptyRuntimeBytes) {
    std::cout << "[RUN] Model KV byte budget rejects a second session before allocation, below the count limit" << std::endl;
    expect(runtime->Memory().currentBytes == emptyRuntimeBytes,
           "The original model must be fully released before the budget-model load");
    expect(cacheBytes > 1 && cacheBytes <= std::numeric_limits<size_t>::max() / 2,
           "Invalid cache size for byte-budget fixture");
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 4;
    options.maxSessionCacheBytes = 2 * cacheBytes - 1;
    auto model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, options);
    expect(model->SessionReservationBytes() == cacheBytes, "Reload changed the expected BF16 KV size");
    const auto resident = runtime->Memory().currentBytes;
    auto first = model->CreateSession();
    expect(model->ActiveSessions() == 1 && first->CacheBytes() == cacheBytes,
           "Byte budget did not admit its first session");
    const auto before = runtime->Memory();
    expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Nonzero KV byte limit");
    const auto after = runtime->Memory();
    expect(model->ActiveSessions() == 1 && model->ActiveSessions() < options.maxSessions &&
               before.currentBytes == after.currentBytes && before.peakBytes == after.peakBytes &&
               first->ContextLength() == 0,
           "Byte-budget rejection allocated memory, changed context, or was actually a count-limit failure");
    first.reset();
    expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == resident,
           "Byte-budget session was not reclaimed");
    auto replacement = model->CreateSession();
    expect(replacement->CacheBytes() == cacheBytes && model->ActiveSessions() == 1,
           "Releasing the first session did not return its byte reservation");
    replacement.reset();
    model.reset();
    expect(runtime->Memory().currentBytes == emptyRuntimeBytes, "Budget-model teardown retained device allocations");
}

void run_correctness(const std::string& modelDirectory) {
    ScopedLogLevel quiet(aila::LogLevel::Error);
    Tokenizer tokenizer;
    expect(tokenizer.load(modelDirectory), "Could not load model tokenizer");
    const auto cases = make_cases(tokenizer);
    auto runtime = DeviceRuntime::Create();
    std::cout << "[INFO] Device: " << runtime->DeviceName() << std::endl;
    const size_t emptyRuntimeBytes = runtime->Memory().currentBytes;
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 4;
    auto model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, options);
    const auto references = test_four_sessions(runtime, model, cases);
    expect(model->ActiveSessions() == 0, "Four-session fixture retained session ownership");
    test_cancel_or_drop(runtime, model, cases, references, false, 192);
    test_cancel_or_drop(runtime, model, cases, references, true, 256);
    test_output_backpressure(model, cases, references);
    test_repeated_reclamation(runtime, model, cases[0].prompt, references.cacheBytes);
    expect(model->ActiveSessions() == 0, "A completed scheduler retained sessions before reload");
    model.reset();
    test_model_cache_budget(runtime, modelDirectory, references.cacheBytes, emptyRuntimeBytes);
}

std::string environment_value(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t size = 0;
    const auto error = _dupenv_s(&value, &size, name);
    const std::unique_ptr<char, decltype(&std::free)> owned(value, &std::free);
    expect(error == 0, std::string("Could not read environment variable ") + name);
    return owned ? owned.get() : "";
#else
    const char* value = std::getenv(name);
    return value ? value : "";
#endif
}

struct BenchmarkRun {
    size_t concurrency = 0;
    int repetition = 0;
    Clock::time_point started;
    double elapsedMs = 0.0;
    double tokensPerSecond = 0.0;
    MemorySnapshot memory;
    std::vector<StreamObservation> streams;
};

BenchmarkRun measure_run(const std::shared_ptr<DeviceRuntime>& runtime,
                         const std::shared_ptr<LoadedModel>& model,
                         const std::array<Tokens, 4>& prompts, size_t concurrency, int repetition) {
    std::vector<std::shared_ptr<SessionState>> sessions;
    sessions.reserve(concurrency);
    for (size_t i = 0; i < concurrency; ++i) sessions.push_back(model->CreateSession());
    // Declare futures before Scheduler. Exception cleanup first cancels/joins
    // the scheduler, then joins any reader futures still awaiting a terminal.
    std::vector<std::future<StreamObservation>> readers;
    readers.reserve(concurrency);
    Scheduler scheduler(scheduler_options(concurrency, 256, 256));
    BenchmarkRun result;
    result.concurrency = concurrency;
    result.repetition = repetition;
    result.streams.reserve(concurrency);
    for (size_t i = 0; i < concurrency; ++i) {
        const auto options = request_options(false, false, 113 + i, 256, 128);
        const auto submitted = Clock::now();
        if (i == 0) result.started = submitted;
        auto handle = scheduler.Submit(sessions[i], prompts[i], options);
        readers.push_back(std::async(std::launch::async, [handle, submitted] {
            return drain_timed(handle, submitted);
        }));
    }
    auto lastRead = result.started;
    size_t tokenCount = 0;
    for (size_t i = 0; i < concurrency; ++i) {
        auto observed = readers[i].get();
        expect_completed(observed, sessions[i], prompts[i],
                         request_options(false, false, 113 + i, 256, 128), 0, "Benchmark request");
        expect(observed.tokenReadMs.size() == 128, "Benchmark lost actual token consumption timestamps");
        lastRead = std::max(lastRead, observed.terminalRead);
        tokenCount += observed.tokens.size();
        result.streams.push_back(std::move(observed));
    }
    scheduler.Shutdown();
    expect_idle(scheduler, "Benchmark scheduler");
    result.elapsedMs = milliseconds(lastRead - result.started);
    expect(result.elapsedMs > 0.0, "Benchmark clock did not advance");
    result.tokensPerSecond = 1000.0 * static_cast<double>(tokenCount) / result.elapsedMs;
    // Taken with exactly N idle sessions still alive. These are Context-
    // tracked USM figures, not a total VRAM or driver-memory measurement.
    result.memory = runtime->Memory();
    return result;
}

std::vector<double> token_intervals(const StreamObservation& stream) {
    std::vector<double> intervals;
    if (stream.tokenReadMs.size() > 1) {
        intervals.reserve(stream.tokenReadMs.size() - 1);
        for (size_t i = 1; i < stream.tokenReadMs.size(); ++i) {
            intervals.push_back(stream.tokenReadMs[i] - stream.tokenReadMs[i - 1]);
        }
    }
    return intervals;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t rank = static_cast<size_t>(std::ceil(quantile * static_cast<double>(values.size())));
    return values[std::min(values.size() - 1, std::max(size_t{1}, rank) - 1)];
}

void write_json_string(std::ostream& output, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    output << '"';
    for (const unsigned char value : text) {
        if (value == '"') output << "\\\"";
        else if (value == '\\') output << "\\\\";
        else if (value < 0x20) output << "\\u00" << hex[value >> 4] << hex[value & 15];
        else output << static_cast<char>(value);
    }
    output << '"';
}

template <typename Value>
void write_json_array(std::ostream& output, const std::vector<Value>& values) {
    output << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) output << ',';
        output << values[i];
    }
    output << ']';
}

void write_benchmark(const std::filesystem::path& path, const std::string& modelDirectory,
                     const std::string& device, const std::array<Tokens, 4>& prompts,
                     const std::vector<BenchmarkRun>& runs) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    expect(output.is_open(), "Could not open benchmark JSON: " + path.string());
    output << std::fixed << std::setprecision(6);
    output << "{\n  \"schema\":\"aila.qwen3.scheduler.benchmark.v1\",\n  \"model_directory\":";
    write_json_string(output, std::filesystem::absolute(modelDirectory).u8string());
    output << ",\n  \"device\":";
    write_json_string(output, device);
    output << ",\n  \"clock\":\"steady_clock\","
              "\n  \"measurement\":\"Each request starts immediately before Submit; token timestamps are taken immediately after its reader thread returns from Read.\","
              "\n  \"throughput_scope\":\"All published tokens divided by last terminal Read minus first Submit start, including prefill and finalization.\","
              "\n  \"percentile_method\":\"nearest_rank\","
              "\n  \"context_tokens\":512,\n  \"prompt_tokens\":256,\n  \"generated_tokens\":128,"
              "\n  \"prefill_chunk_tokens\":256,\n  \"max_buffered_tokens_per_request\":256,"
              "\n  \"warmup_runs_per_concurrency\":1,\n  \"measured_runs_per_concurrency\":3,"
              "\n  \"sampling\":\"greedy\",\n  \"kv_dtype\":\"bf16\","
              "\n  \"memory_scope\":\"Context-tracked USM with N idle sessions alive; excludes oneDNN/driver allocations. Peak is process-wide.\","
              "\n  \"environment\":{";
    const std::array<const char*, 8> environmentNames{{
        "AILA_KV_QUANT", "AILA_ATTN_JM", "AILA_BNB4_GEMM_XMX", "AILA_FUSE_RESIDUAL_ADD",
        "AILA_PROFILE_Q3_DECODE", "AILA_PROFILE_Q3_PREFILL", "AILA_PROFILE_Q3_HOST_ONLY",
        "ONEAPI_DEVICE_SELECTOR"
    }};
    for (size_t i = 0; i < environmentNames.size(); ++i) {
        if (i) output << ',';
        write_json_string(output, environmentNames[i]);
        output << ':';
        const auto value = environment_value(environmentNames[i]);
        if (value.empty()) output << "null";
        else write_json_string(output, value);
    }
    output << "},\n  \"prompt_token_ids\":[";
    for (size_t i = 0; i < prompts.size(); ++i) {
        if (i) output << ',';
        write_json_array(output, prompts[i]);
    }
    output << "],\n  \"results\":[\n";
    bool firstGroup = true;
    for (const size_t concurrency : {size_t{1}, size_t{2}, size_t{4}}) {
        if (!firstGroup) output << ",\n";
        firstGroup = false;
        std::vector<double> ttft;
        std::vector<double> intervals;
        double totalElapsedMs = 0.0;
        size_t totalTokens = 0;
        output << "    {\"concurrency\":" << concurrency << ",\"runs\":[";
        bool firstRun = true;
        for (const auto& run : runs) {
            if (run.concurrency != concurrency) continue;
            if (!firstRun) output << ',';
            firstRun = false;
            totalElapsedMs += run.elapsedMs;
            output << "{\"repetition\":" << run.repetition << ",\"elapsed_ms\":" << run.elapsedMs
                   << ",\"overall_tokens_per_second\":" << run.tokensPerSecond
                   << ",\"tracked_device_current_bytes\":" << run.memory.currentBytes
                   << ",\"tracked_device_process_peak_bytes\":" << run.memory.peakBytes
                   << ",\"requests\":[";
            for (size_t i = 0; i < run.streams.size(); ++i) {
                if (i) output << ',';
                const auto& stream = run.streams[i];
                const auto gaps = token_intervals(stream);
                ttft.push_back(stream.tokenReadMs.front());
                intervals.insert(intervals.end(), gaps.begin(), gaps.end());
                totalTokens += stream.tokens.size();
                output << "{\"request_index\":" << i
                       << ",\"submit_offset_ms\":" << milliseconds(stream.submitted - run.started)
                       << ",\"ttft_ms\":" << stream.tokenReadMs.front()
                       << ",\"completion_ms\":" << milliseconds(stream.terminalRead - stream.submitted)
                       << ",\"inter_token_ms_p50\":" << percentile(gaps, 0.50)
                       << ",\"inter_token_ms_p95\":" << percentile(gaps, 0.95)
                       << ",\"generated_tokens\":" << stream.tokens.size()
                       << ",\"token_read_ms_from_submit\":";
                write_json_array(output, stream.tokenReadMs);
                output << ",\"generated_token_ids\":";
                write_json_array(output, stream.tokens);
                output << '}';
            }
            output << "]}";
        }
        expect(totalElapsedMs > 0.0, "Benchmark group has no measured duration");
        output << "],\"aggregate\":{\"ttft_ms_p50\":" << percentile(ttft, 0.50)
               << ",\"ttft_ms_p95\":" << percentile(ttft, 0.95)
               << ",\"inter_token_ms_p50\":" << percentile(intervals, 0.50)
               << ",\"inter_token_ms_p95\":" << percentile(intervals, 0.95)
               << ",\"overall_tokens_per_second\":" << 1000.0 * static_cast<double>(totalTokens) / totalElapsedMs
               << ",\"total_generated_tokens\":" << totalTokens << "}}";
    }
    output << "\n  ]\n}\n";
    output.close();
    expect(output.good(), "Failed to write complete benchmark JSON: " + path.string());
}

void run_benchmark(const std::string& modelDirectory, const std::filesystem::path& outputPath) {
    ScopedLogLevel quiet(aila::LogLevel::Error);
    Tokenizer tokenizer;
    expect(tokenizer.load(modelDirectory), "Could not load benchmark tokenizer");
    const auto cases = make_cases(tokenizer);
    std::array<Tokens, 4> prompts;
    for (size_t i = 0; i < prompts.size(); ++i) prompts[i] = repeat_tokens(cases[i].prompt, 256);
    auto runtime = DeviceRuntime::Create();
    const auto device = runtime->DeviceName();
    std::cout << "[INFO] Benchmark device: " << device << std::endl;
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 4;
    auto model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, options);
    std::vector<BenchmarkRun> runs;
    runs.reserve(9);
    for (const size_t concurrency : {size_t{1}, size_t{2}, size_t{4}}) {
        std::cout << "[BENCH] sessions=" << concurrency << " warmup 1/1" << std::endl;
        measure_run(runtime, model, prompts, concurrency, 0);
        expect(model->ActiveSessions() == 0, "Benchmark warmup retained session resources");
        for (int repetition = 1; repetition <= 3; ++repetition) {
            auto run = measure_run(runtime, model, prompts, concurrency, repetition);
            std::cout << "[BENCH] sessions=" << concurrency << " repeat=" << repetition
                      << " elapsed_ms=" << run.elapsedMs << " overall_tok_s=" << run.tokensPerSecond << std::endl;
            runs.push_back(std::move(run));
            expect(model->ActiveSessions() == 0, "Benchmark run retained session resources");
        }
    }
    write_benchmark(outputPath, modelDirectory, device, prompts, runs);
    std::cout << "[INFO] Benchmark JSON: " << std::filesystem::absolute(outputPath).string() << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string modelDirectory;
        std::string benchmarkPath;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "Usage: Qwen3SchedulerTests [--model <Qwen3 NF4 directory>] [--benchmark <output.json>]\n"
                             "Fallback environment variable: AILA_QWEN3_TEST_MODEL\n"
                             "Requires BF16 KV (AILA_KV_QUANT=0). Default: correctness only.\n"
                             "--benchmark runs performance only: 1/2/4 sessions, prompt 256, generate 128,\n"
                             "one warmup and three measured runs per concurrency.\n";
                return 0;
            }
            if ((argument != "--model" && argument != "--benchmark") || i + 1 >= argc) {
                throw std::runtime_error("Expected --model <directory> or --benchmark <output.json>");
            }
            auto& destination = argument == "--model" ? modelDirectory : benchmarkPath;
            expect(destination.empty(), "Duplicate command-line argument: " + argument);
            destination = argv[++i];
            expect(!destination.empty(), "Empty command-line argument: " + argument);
        }
        if (modelDirectory.empty()) modelDirectory = environment_value("AILA_QWEN3_TEST_MODEL");
        if (modelDirectory.empty()) {
            std::cout << "[SKIP] Specify --model <directory> or AILA_QWEN3_TEST_MODEL for the real GPU scheduler fixture.\n";
            return 77;
        }
        expect(std::filesystem::is_directory(modelDirectory), "Model directory does not exist: " + modelDirectory);
        if (benchmarkPath.empty()) {
            run_correctness(modelDirectory);
            std::cout << "Qwen3 scheduler correctness tests passed." << std::endl;
        } else {
            run_benchmark(modelDirectory, benchmarkPath);
            std::cout << "Qwen3 scheduler benchmark completed." << std::endl;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3 scheduler tests failed: " << error.what() << std::endl;
        return 1;
    }
}
