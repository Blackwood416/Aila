#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "inference/Session.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using Tokens = std::vector<int>;
using Plan = std::vector<Tokens>;
using LogitsTrace = std::vector<std::vector<float>>;

constexpr int kCapacity = 512;
constexpr int kNewTokens = 8;
constexpr int kMaxSteps = 2 * kCapacity;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

enum class LogProbePhase : size_t { Load, Session, Growth, Request, Count };

struct LogReadProbe {
    // Borrow the caller's handles: registering a log probe must not add an
    // owner that keeps a Session alive during later reclamation tests.
    const std::shared_ptr<DeviceRuntime>& runtime;
    const std::shared_ptr<LoadedModel>& model;
    const std::shared_ptr<SessionState>& session;
    const std::shared_ptr<RequestState>& request;
    LogProbePhase phase = LogProbePhase::Load;
    static constexpr size_t phaseCount = static_cast<size_t>(LogProbePhase::Count);
    std::array<size_t, phaseCount> calls{};
    std::array<size_t, phaseCount> modelReads{};
    std::array<size_t, phaseCount> sessionReads{};
    std::array<size_t, phaseCount> requestReads{};
    std::array<size_t, phaseCount> runtimeGrowthLogs{};
    std::exception_ptr failure;
    bool invalidMetadata = false;

    static void Callback(int, const char* message, void* userData) noexcept {
        auto& probe = *static_cast<LogReadProbe*>(userData);
        const size_t index = static_cast<size_t>(probe.phase);
        ++probe.calls[index];
        if (message && std::strstr(message, "Runtime buffers resized")) {
            ++probe.runtimeGrowthLogs[index];
        }
        try {
            const auto memory = probe.runtime->Memory();
            if (memory.peakBytes < memory.currentBytes || probe.runtime->DeviceName().empty()) {
                probe.invalidMetadata = true;
            }
            if (probe.model) {
                if (probe.model->ActiveSessions() > 2) probe.invalidMetadata = true;
                ++probe.modelReads[index];
            }
            if (probe.session) {
                const int length = probe.session->ContextLength();
                if (length < 0 || length > probe.session->Capacity()) probe.invalidMetadata = true;
                ++probe.sessionReads[index];
            }
            if (probe.request) {
                const auto snapshot = probe.request->Snapshot();
                if (snapshot.generatedIds.size() > kNewTokens) probe.invalidMetadata = true;
                ++probe.requestReads[index];
            }
        } catch (...) {
            // Record getter failures without exercising callback-throw policy.
            if (!probe.failure) probe.failure = std::current_exception();
        }
    }

    void Check(const std::string& label) const {
        if (failure) {
            try {
                std::rethrow_exception(failure);
            } catch (const std::exception& error) {
                throw std::runtime_error(label + ": log callback getter failed: " + error.what());
            } catch (...) {
                throw std::runtime_error(label + ": log callback getter failed");
            }
        }
        expect(!invalidMetadata, label + ": log callback read invalid metadata");
    }
};

class ScopedReadLogCallback {
public:
    explicit ScopedReadLogCallback(LogReadProbe& probe)
        : originalLevel(aila::get_log_level()) {
        // This standalone program has installed no earlier custom handler.
        // Restore that known default and the captured level on every exit.
        aila::set_log_level(aila::LogLevel::Info);
        aila::set_log_callback(&LogReadProbe::Callback, &probe);
    }

    ~ScopedReadLogCallback() { Restore(); }
    ScopedReadLogCallback(const ScopedReadLogCallback&) = delete;
    ScopedReadLogCallback& operator=(const ScopedReadLogCallback&) = delete;

    void Restore() {
        if (!installed) return;
        aila::set_log_callback(nullptr, nullptr);
        aila::set_log_level(originalLevel);
        installed = false;
    }

    aila::LogLevel OriginalLevel() const { return originalLevel; }

private:
    aila::LogLevel originalLevel;
    bool installed = true;
};

class ScopedKvQuantOverride {
public:
    explicit ScopedKvQuantOverride(bool enabled)
        : original(aila::env::g_kv_quant_override) {
        // This fixture submits all work on one thread. Avoid changing the
        // process environment while exercising the effective configuration.
        aila::env::g_kv_quant_override = enabled;
    }
    ~ScopedKvQuantOverride() { aila::env::g_kv_quant_override = original; }
    ScopedKvQuantOverride(const ScopedKvQuantOverride&) = delete;
    ScopedKvQuantOverride& operator=(const ScopedKvQuantOverride&) = delete;

private:
    bool original;
};

template <typename Function>
void expect_error(ErrorCode expectedCode, Function&& operation,
                  const std::string& label) {
    try {
        operation();
    } catch (const Error& error) {
        expect(error.Code() == expectedCode,
               label + ": wrong inference error: " + error.what());
        return;
    }
    throw std::runtime_error(label + ": expected an inference error");
}

template <typename Function>
void expect_quantized_kv_rejection(Function&& operation, const std::string& label) {
    try {
        operation();
    } catch (const Error& error) {
        expect(error.Code() == ErrorCode::InvalidArgument,
               label + ": quantized KV rejection used the wrong error code");
        const std::string message = error.what();
        expect(message.find("BF16") != std::string::npos &&
               message.find("AILA_KV_QUANT=0") != std::string::npos,
               label + ": rejection did not explain the supported KV configuration");
        return;
    }
    throw std::runtime_error(label + ": quantized KV was not rejected");
}

void test_reject_quantized_load(const std::shared_ptr<DeviceRuntime>& runtime,
                                const std::string& modelDirectory,
                                const ModelOptions& options) {
    std::cout << "[RUN] Quantized KV is rejected before shared-model allocation" << std::endl;
    const auto before = runtime->Memory();
    {
        ScopedKvQuantOverride quantized(true);
        expect_quantized_kv_rejection(
            [&] { LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, options); },
            "Quantized model load");
        const auto after = runtime->Memory();
        expect(after.currentBytes == before.currentBytes && after.peakBytes == before.peakBytes,
               "Rejected model load allocated device resources");
    }
    // The caller now performs the ordinary BF16 load and the full inference
    // fixture, proving that this admission error did not poison the runtime.
}

void test_reject_quantized_session(const std::shared_ptr<DeviceRuntime>& runtime,
                                   const std::shared_ptr<LoadedModel>& model) {
    std::cout << "[RUN] Quantized KV enabled after load is rejected without consuming a session slot" << std::endl;
    const auto before = runtime->Memory();
    const size_t activeBefore = model->ActiveSessions();
    {
        ScopedKvQuantOverride quantized(true);
        expect_quantized_kv_rejection([&] { model->CreateSession(); }, "Quantized session creation");
        const auto after = runtime->Memory();
        expect(model->ActiveSessions() == activeBefore &&
               after.currentBytes == before.currentBytes && after.peakBytes == before.peakBytes,
               "Rejected session creation changed the active count or allocated device resources");
    }
    // The following runtime-sharing and interleaving cases create more BF16
    // sessions and execute them alongside the already existing session.
}

Tokens repeat_tokens(const Tokens& source, size_t count) {
    expect(!source.empty(), "Tokenizer returned an empty fixture");
    Tokens result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(source[i % source.size()]);
    return result;
}

Tokens slice(const Tokens& tokens, size_t start, size_t count) {
    expect(start <= tokens.size() && count <= tokens.size() - start,
           "Invalid test fixture slice");
    return Tokens(tokens.begin() + start, tokens.begin() + start + count);
}

Plan make_plan(const Tokens& prompt, size_t firstChunk, const Tokens& continuation) {
    Plan plan{slice(prompt, 0, firstChunk),
              slice(prompt, firstChunk, prompt.size() - firstChunk)};
    for (int token : continuation) plan.push_back({token});
    return plan;
}

size_t token_count(const Plan& plan) {
    size_t count = 0;
    for (const auto& step : plan) count += step.size();
    return count;
}

void check_logits(const std::vector<float>& logits, int vocabSize,
                  const std::string& label) {
    expect(logits.size() == static_cast<size_t>(vocabSize), label + ": wrong logits size");
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) {
            throw std::runtime_error(label + ": non-finite logit at " + std::to_string(i));
        }
    }
}

void expect_logits(const std::vector<float>& actual,
                   const std::vector<float>& expected, const std::string& label) {
    expect(actual.size() == expected.size(), label + ": logits sizes differ");
    for (size_t i = 0; i < actual.size(); ++i) {
        // These are BF16 logits converted losslessly to float. Both executions
        // use identical prefill/decode shapes, so require elementwise equality.
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
            std::ostringstream message;
            message << label << ": logit " << i << " differs: expected "
                    << expected[i] << ", got " << actual[i];
            throw std::runtime_error(message.str());
        }
    }
}

std::vector<float> forward(const std::shared_ptr<SessionState>& session,
                           const Tokens& tokens, const std::string& label) {
    const int before = session->ContextLength();
    auto logits = session->ForwardLogits(tokens);
    expect(session->ContextLength() == before + static_cast<int>(tokens.size()),
           label + ": wrong context advance");
    expect(!session->Busy(), label + ": direct forward left the session busy");
    check_logits(logits, session->VocabSize(), label);
    return logits;
}

LogitsTrace capture_plan(const std::shared_ptr<SessionState>& session,
                         const Plan& plan, const std::string& label) {
    LogitsTrace trace;
    for (size_t i = 0; i < plan.size(); ++i) {
        trace.push_back(forward(session, plan[i], label + " step " + std::to_string(i)));
    }
    return trace;
}

void test_shared_runtime_handle(const std::shared_ptr<DeviceRuntime>& runtime,
                                const std::shared_ptr<LoadedModel>& model,
                                const std::shared_ptr<SessionState>& session) {
    std::cout << "[RUN] Distinct runtime handles share the active device allocation domain" << std::endl;
    const auto before = runtime->Memory();
    expect(before.currentBytes > session->CacheBytes(), "Runtime sharing fixture has no loaded weights");
    auto second = DeviceRuntime::Create();
    expect(second.get() != runtime.get(), "Create returned the same public runtime handle");
    const auto observed = second->Memory();
    expect(observed.currentBytes == before.currentBytes && observed.peakBytes == before.peakBytes,
           "Second runtime handle cannot see the existing model/session allocations");
    expect(second->DeviceName() == runtime->DeviceName(), "Runtime handles report different devices");
    {
        auto temporary = model->CreateSession();
        const auto firstView = runtime->Memory();
        const auto secondView = second->Memory();
        expect(firstView.currentBytes == before.currentBytes + temporary->CacheBytes() &&
               secondView.currentBytes == firstView.currentBytes &&
               secondView.peakBytes == firstView.peakBytes,
               "Runtime handles did not observe the same live session allocation");
    }
    expect(second->Memory().currentBytes == before.currentBytes,
           "Second runtime handle did not observe session reclamation");
    second.reset();
    expect(runtime->Memory().currentBytes == before.currentBytes && model->ActiveSessions() == 1 &&
           session->ContextLength() == 0,
           "Dropping the second runtime handle disturbed the live model/session");
}

struct ReferenceTraces {
    LogitsTrace a;
    LogitsTrace b;
};

ReferenceTraces test_interleaved_logits(const std::shared_ptr<SessionState>& a,
                                       const std::shared_ptr<SessionState>& b,
                                       const Plan& planA, const Plan& planB) {
    std::cout << "[RUN] Interleaved logits, including short-to-long workspace growth" << std::endl;
    ReferenceTraces interleaved;
    // Run this before any large prefill: A starts with 8 tokens, B with 96.
    // Their first steps force the loaded execution slot from capacity 64 to 128.
    for (size_t i = 0; i < std::max(planA.size(), planB.size()); ++i) {
        if (i < planA.size()) {
            interleaved.a.push_back(forward(a, planA[i], "interleaved A " + std::to_string(i)));
        }
        if (i < planB.size()) {
            interleaved.b.push_back(forward(b, planB[i], "interleaved B " + std::to_string(i)));
        }
    }
    expect(a->ContextLength() == static_cast<int>(token_count(planA)), "A context was contaminated");
    expect(b->ContextLength() == static_cast<int>(token_count(planB)), "B context was contaminated");

    a->Reset();
    b->Reset();
    ReferenceTraces standalone{capture_plan(a, planA, "standalone A"),
                               capture_plan(b, planB, "standalone B")};
    for (size_t i = 0; i < planA.size(); ++i) {
        expect_logits(interleaved.a[i], standalone.a[i], "A interleaving " + std::to_string(i));
    }
    for (size_t i = 0; i < planB.size(); ++i) {
        expect_logits(interleaved.b[i], standalone.b[i], "B interleaving " + std::to_string(i));
    }
    return standalone;
}

RequestOptions request_options(bool sampling, bool penalties, uint64_t seed, int chunkSize) {
    RequestOptions options;
    options.prefillChunkSize = chunkSize;
    options.generation.max_new_tokens = kNewTokens;
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
    // No stop IDs: each request must exercise exactly kNewTokens and flush its
    // final emitted token into the cache, including if that token is an EOS ID.
    return options;
}

struct ObservedRequest {
    Tokens tokens;
    int terminalEvents = 0;
};

void take_step(const std::shared_ptr<RequestState>& request, ObservedRequest& observed,
               const std::string& label, const Tokens* expected = nullptr) {
    const auto step = request->Step();
    if (step.tokenId) {
        observed.tokens.push_back(*step.tokenId);
        if (expected) {
            const size_t index = observed.tokens.size() - 1;
            expect(index < expected->size() && *step.tokenId == (*expected)[index],
                   label + ": generated token differs at " + std::to_string(index));
        }
    }
    if (step.becameTerminal) ++observed.terminalEvents;
    const auto snapshot = request->Snapshot();
    expect(snapshot.status != RequestStatus::Failed, label + ": " + snapshot.error);
    expect(snapshot.status == step.status, label + ": step/snapshot statuses differ");
    expect(snapshot.generatedIds == observed.tokens, label + ": token events do not match the snapshot");
    expect(observed.terminalEvents <= 1, label + ": duplicate terminal event");
}

void check_terminal(const std::shared_ptr<RequestState>& request,
                    const ObservedRequest& observed, RequestStatus expected,
                    const std::string& label) {
    expect(request->Snapshot().status == expected, label + ": wrong terminal status");
    expect(observed.terminalEvents == 1, label + ": missing terminal event");
    const auto repeated = request->Step();
    expect(repeated.status == expected && !repeated.tokenId && !repeated.becameTerminal,
           label + ": stepping a terminal request changed its output");
}

Tokens run_request(const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                   const RequestOptions& options, const std::string& label) {
    session->Reset();
    auto request = RequestState::Create(session, prompt, options);
    ObservedRequest observed;
    for (int i = 0; i < kMaxSteps && !IsTerminal(request->Snapshot().status); ++i) {
        take_step(request, observed, label);
    }
    check_terminal(request, observed, RequestStatus::Completed, label);
    expect(observed.tokens.size() == static_cast<size_t>(options.generation.max_new_tokens),
           label + ": wrong generated token count");
    expect(session->ContextLength() == static_cast<int>(prompt.size() + observed.tokens.size()),
           label + ": final token was not committed to the cache");
    expect(!session->Busy(), label + ": completed request retained the session lease");
    return observed.tokens;
}

struct RequestReferences {
    RequestOptions optionsA;
    RequestOptions optionsB;
    Tokens a;
    Tokens b;
};

RequestReferences test_request_mode(const std::shared_ptr<SessionState>& a,
                                    const std::shared_ptr<SessionState>& b,
                                    const Tokens& promptA, const Tokens& promptB,
                                    bool sampling, bool penalties, const std::string& label) {
    std::cout << "[RUN] " << label << std::endl;
    RequestReferences reference{request_options(sampling, penalties, 113, 8),
                                request_options(sampling, penalties, 277, 48), {}, {}};
    reference.a = run_request(a, promptA, reference.optionsA, label + " standalone A");
    reference.b = run_request(b, promptB, reference.optionsB, label + " standalone B");

    a->Reset();
    b->Reset();
    auto requestA = RequestState::Create(a, promptA, reference.optionsA);
    auto requestB = RequestState::Create(b, promptB, reference.optionsB);
    ObservedRequest observedA, observedB;
    for (int i = 0; i < kMaxSteps; ++i) {
        const bool doneA = IsTerminal(requestA->Snapshot().status);
        const bool doneB = IsTerminal(requestB->Snapshot().status);
        if (doneA && doneB) break;
        if (!doneA) take_step(requestA, observedA, label + " interleaved A", &reference.a);
        if (!doneB) take_step(requestB, observedB, label + " interleaved B", &reference.b);
    }
    check_terminal(requestA, observedA, RequestStatus::Completed, label + " A");
    check_terminal(requestB, observedB, RequestStatus::Completed, label + " B");
    expect(observedA.tokens == reference.a && observedB.tokens == reference.b,
           label + ": request interleaving changed generated IDs");
    expect(a->ContextLength() == static_cast<int>(promptA.size() + reference.a.size()) &&
           b->ContextLength() == static_cast<int>(promptB.size() + reference.b.size()),
           label + ": final context length differs from standalone execution");
    expect(!a->Busy() && !b->Busy(), label + ": a completed session remains busy");
    return reference;
}

void test_request_log_reentry(LogReadProbe& probe,
                              const std::shared_ptr<SessionState>& session,
                              const Tokens& prompt,
                              std::shared_ptr<RequestState>& request) {
    std::cout << "[RUN] Log callbacks reenter request/session getters after a growing prefill step" << std::endl;
    session->Reset();
    probe.phase = LogProbePhase::Request;
    const auto options = request_options(false, false, 991, 192);
    request = RequestState::Create(session, repeat_tokens(prompt, 192), options);
    ObservedRequest observed;
    // The earlier trace grew the slot to 128. This step grows it again while
    // RequestState's step lock is in scope, so callback Snapshot must be deferred
    // until that lock and the nested session/runtime locks have been released.
    take_step(request, observed, "Logging request prefill");
    probe.Check("Request logging");
    const size_t index = static_cast<size_t>(LogProbePhase::Request);
    expect(probe.runtimeGrowthLogs[index] > 0 && probe.requestReads[index] > 0 &&
           probe.sessionReads[index] > 0 && observed.tokens.size() == 1,
           "Growing request prefill did not exercise reentrant log getters");
    request->Cancel();
    take_step(request, observed, "Logging request cancellation");
    check_terminal(request, observed, RequestStatus::Cancelled, "Logging request cancellation");
    request.reset();
    session->Reset();
}

void test_cancel_and_destroy(const std::shared_ptr<DeviceRuntime>& runtime,
                             const std::shared_ptr<LoadedModel>& model,
                             std::shared_ptr<SessionState>& a,
                             const std::shared_ptr<SessionState>& b,
                             const Tokens& promptA, const Tokens& promptB,
                             const Plan& planA, const ReferenceTraces& traces,
                             const RequestReferences& reference) {
    std::cout << "[RUN] Cancel/destroy A and reuse its session slot while B continues" << std::endl;
    a->Reset();
    b->Reset();
    auto requestA = RequestState::Create(a, promptA, reference.optionsA);
    auto requestB = RequestState::Create(b, promptB, reference.optionsB);
    ObservedRequest observedA, observedB;
    for (int i = 0; i < kMaxSteps && observedB.tokens.size() < 2; ++i) {
        take_step(requestA, observedA, "cancel A prefix", &reference.a);
        take_step(requestB, observedB, "surviving B prefix", &reference.b);
    }
    expect(observedA.tokens.size() >= 2 && observedB.tokens.size() == 2,
           "Cancellation fixture did not reach decode");
    const int cancelledLength = a->ContextLength();
    requestA->Cancel();
    take_step(requestA, observedA, "cancel A");
    check_terminal(requestA, observedA, RequestStatus::Cancelled, "cancel A");
    expect(!a->Busy() && a->ContextLength() == cancelledLength,
           "Cancellation submitted another token or retained the session lease");

    const size_t cacheBytes = a->CacheBytes();
    const size_t beforeRelease = runtime->Memory().currentBytes;
    requestA.reset();
    a.reset();
    expect(model->ActiveSessions() == 1, "Destroyed A still consumes a session slot");
    expect(runtime->Memory().currentBytes == beforeRelease - cacheBytes,
           "Destroying A did not release exactly its KV allocation");

    const size_t beforeReplacement = runtime->Memory().currentBytes;
    {
        auto replacement = model->CreateSession();
        expect(model->ActiveSessions() == 2 && replacement->ContextLength() == 0,
               "Replacement session did not start with an empty independent cache");
        expect(runtime->Memory().currentBytes == beforeReplacement + replacement->CacheBytes(),
               "Replacement session duplicated non-KV device resources");
        expect_logits(forward(replacement, planA.front(), "replacement session"),
                      traces.a.front(), "replacement session logits");
    }
    expect(model->ActiveSessions() == 1 && runtime->Memory().currentBytes == beforeReplacement,
           "Replacement session resources were not reclaimed");

    for (int i = 0; i < kMaxSteps && !IsTerminal(requestB->Snapshot().status); ++i) {
        take_step(requestB, observedB, "B after A destruction", &reference.b);
    }
    check_terminal(requestB, observedB, RequestStatus::Completed, "surviving B");
    expect(observedB.tokens == reference.b, "A cancellation/destruction changed B's output");
    expect(!b->Busy(), "Surviving B did not release its request lease");
}

void test_reset_and_truncate(const std::shared_ptr<SessionState>& a,
                             const std::shared_ptr<SessionState>& b,
                             const Plan& planA, const Plan& planB,
                             const ReferenceTraces& traces) {
    std::cout << "[RUN] Reset/truncate and continuation without cross-session contamination" << std::endl;
    a->Reset();
    b->Reset();
    for (size_t i = 0; i < 4; ++i) {
        expect_logits(forward(a, planA[i], "A before rollback"), traces.a[i], "A before rollback logits");
    }
    expect_logits(forward(b, planB[0], "B before A rollback"), traces.b[0], "B initial logits");
    const int prefixLength = static_cast<int>(planA[0].size() + planA[1].size());
    expect(a->Truncate(prefixLength) == prefixLength && a->ContextLength() == prefixLength,
           "Qwen3 did not restore the requested prefix length");
    expect(b->ContextLength() == static_cast<int>(planB[0].size()), "A rollback changed B's length");
    for (size_t i = 2; i < planA.size(); ++i) {
        expect_logits(forward(a, planA[i], "A after rollback"), traces.a[i], "A resumed logits");
    }
    expect_logits(forward(b, planB[1], "B after A rollback"), traces.b[1], "B after rollback logits");

    const int retainedLength = a->ContextLength();
    expect_error(ErrorCode::InvalidArgument, [&] { a->Truncate(-1); }, "Negative truncate");
    expect_error(ErrorCode::InvalidArgument, [&] { a->Truncate(retainedLength + 1); }, "Oversized truncate");
    expect(a->ContextLength() == retainedLength && !a->Busy(), "Invalid truncate changed session state");
    a->Reset();
    expect(a->ContextLength() == 0, "Reset did not empty A");
    const auto replay = capture_plan(a, planA, "A after reset");
    for (size_t i = 0; i < replay.size(); ++i) expect_logits(replay[i], traces.a[i], "Reset replay");
    for (size_t i = 2; i < planB.size(); ++i) {
        expect_logits(forward(b, planB[i], "B after A reset"), traces.b[i], "B after reset logits");
    }
}

void test_limits(const std::shared_ptr<DeviceRuntime>& runtime,
                 const std::shared_ptr<LoadedModel>& model,
                 const std::shared_ptr<SessionState>& a,
                 const std::shared_ptr<SessionState>& b,
                 const Tokens& promptA, const Plan& planA, const Plan& planB,
                 const ReferenceTraces& traces) {
    std::cout << "[RUN] Busy, queued cancellation, input validation and exact context capacity" << std::endl;
    a->Reset();
    b->Reset();
    const auto options = request_options(false, false, 19, 8);
    const size_t beforeQueued = runtime->Memory().currentBytes;
    auto queued = RequestState::Create(a, promptA, options);
    expect(a->Busy(), "Queued request did not acquire its session");
    expect_error(ErrorCode::Busy, [&] { RequestState::Create(a, promptA, options); }, "Same-session request");
    expect_error(ErrorCode::Busy, [&] { a->Reset(); }, "Reset while busy");
    expect_error(ErrorCode::Busy, [&] { a->Truncate(0); }, "Truncate while busy");
    expect_error(ErrorCode::Busy, [&] { a->ForwardLogits({promptA[0]}); }, "Forward while busy");
    expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Session-count bound");
    expect(model->ActiveSessions() == 2, "Rejected session changed the active count");
    queued->Cancel();
    ObservedRequest cancelled;
    take_step(queued, cancelled, "queued cancellation");
    check_terminal(queued, cancelled, RequestStatus::Cancelled, "queued cancellation");
    queued.reset();
    expect(!a->Busy() && a->ContextLength() == 0 && cancelled.tokens.empty(),
           "Queued cancellation changed the empty session");
    expect(runtime->Memory().currentBytes == beforeQueued, "Rejected/queued operations allocated device resources");

    expect_error(ErrorCode::InvalidArgument, [&] { a->ForwardLogits({}); }, "Empty input");
    expect_error(ErrorCode::InvalidArgument, [&] { a->ForwardLogits({-1}); }, "Negative token ID");
    expect_error(ErrorCode::InvalidArgument, [&] { a->ForwardLogits({a->VocabSize()}); }, "Out-of-vocabulary token ID");
    expect_error(ErrorCode::ContextOverflow,
                 [&] { a->ForwardLogits(repeat_tokens(promptA, kCapacity + 1)); }, "Oversized forward");
    auto oversized = options;
    oversized.generation.max_new_tokens = kCapacity;
    expect_error(ErrorCode::ContextOverflow,
                 [&] { RequestState::Create(a, {promptA[0]}, oversized); }, "Oversized request reservation");
    expect(!a->Busy() && a->ContextLength() == 0, "Rejected input changed session state");

    // Accept the boundary itself, then reject one more token without poisoning
    // the shared execution domain. This also grows the slot beyond both traces.
    forward(a, repeat_tokens(promptA, kCapacity), "Exact-capacity prefill");
    expect(a->ContextLength() == kCapacity, "Exact capacity was not accepted");
    expect_error(ErrorCode::ContextOverflow, [&] { a->ForwardLogits({promptA[0]}); }, "Full-cache forward");
    expect_error(ErrorCode::ContextOverflow,
                 [&] { RequestState::Create(a, {promptA[0]}, options); }, "Full-cache request");
    expect(!a->Busy() && a->ContextLength() == kCapacity, "Failed admission retained the session lease");
    expect_logits(forward(b, planB[0], "B after A capacity errors"), traces.b[0], "Unaffected B after capacity errors");
    a->Reset();
    expect_logits(forward(a, planA[0], "A after capacity reset"), traces.a[0], "A after capacity reset logits");
}

void test_release_and_recreate(const std::shared_ptr<DeviceRuntime>& runtime,
                               const std::shared_ptr<LoadedModel>& model,
                               std::shared_ptr<SessionState>& a,
                               std::shared_ptr<SessionState>& b) {
    std::cout << "[RUN] Session KV accounting and repeated resource reclamation" << std::endl;
    const size_t bytesA = a->CacheBytes();
    const size_t bytesB = b->CacheBytes();
    const size_t withSessions = runtime->Memory().currentBytes;
    expect(withSessions >= bytesA + bytesB, "Invalid tracked device memory total");
    // Shared primitives/workspaces may have grown during the tests. Their
    // retained allocations are model resources, not leaked per-session KV.
    const size_t modelResident = withSessions - bytesA - bytesB;
    a.reset();
    expect(model->ActiveSessions() == 1 && runtime->Memory().currentBytes == modelResident + bytesB,
           "Releasing the first session did not reclaim exactly its cache");
    b.reset();
    expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == modelResident,
           "Releasing both sessions did not restore model-resident memory");
    for (int i = 0; i < 3; ++i) {
        {
            auto first = model->CreateSession();
            auto second = model->CreateSession();
            expect(model->ActiveSessions() == 2, "Recreated sessions were not counted");
            expect(runtime->Memory().currentBytes == modelResident + first->CacheBytes() + second->CacheBytes(),
                   "Creating sessions duplicated weights or execution workspace");
        }
        expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == modelResident,
               "Repeated session create/destroy accumulated tracked allocations");
    }
    std::cout << "[INFO] Retained model/workspace bytes=" << modelResident
              << ", bytes per session=" << bytesA << std::endl;
}

void test_owner_lifetime(std::shared_ptr<DeviceRuntime>& runtime,
                         std::shared_ptr<LoadedModel>& model,
                         const Plan& planA, const ReferenceTraces& traces,
                         const Tokens& promptA, const RequestReferences& reference) {
    std::cout << "[RUN] Session survives destruction of external model/runtime handles" << std::endl;
    auto survivor = model->CreateSession();
    model.reset();
    runtime.reset();
    const auto trace = capture_plan(survivor, planA, "Surviving session");
    for (size_t i = 0; i < trace.size(); ++i) expect_logits(trace[i], traces.a[i], "Surviving session logits");
    expect(run_request(survivor, promptA, reference.optionsA, "Surviving session request") == reference.a,
           "Dropping model/runtime handles changed request output");
    survivor.reset();
}

void run_tests(const std::string& modelDirectory) {
    Tokenizer tokenizer;
    expect(tokenizer.load(modelDirectory), "Could not load the model tokenizer");
    // Tokenized text with fixed lengths makes prefill shapes reproducible.
    const Tokens promptA = repeat_tokens(tokenizer.encode("Explain how a small robot sorts red and blue blocks. "), 24);
    const Tokens promptB = repeat_tokens(tokenizer.encode("A patient gardener records rainfall, soil temperature, and the growth of young trees each morning. "), 144);
    const Plan planA = make_plan(promptA, 8, repeat_tokens(tokenizer.encode(" First choose a color."), 4));
    const Plan planB = make_plan(promptB, 96, repeat_tokens(tokenizer.encode(" Then compare the measurements."), 4));

    auto runtime = DeviceRuntime::Create();
    std::cout << "[INFO] Device: " << runtime->DeviceName() << std::endl;
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 2;
    test_reject_quantized_load(runtime, modelDirectory, options);
    std::shared_ptr<LoadedModel> model;
    std::shared_ptr<SessionState> a;
    std::shared_ptr<RequestState> loggingRequest;
    LogReadProbe logProbe{runtime, model, a, loggingRequest};
    ScopedReadLogCallback logRegistration(logProbe);
    std::cout << "[RUN] Log callbacks reenter runtime getters during model loading" << std::endl;
    model = LoadedModel::LoadQwen3Nf4(runtime, modelDirectory, options);
    logProbe.Check("Model loading");
    expect(logProbe.calls[static_cast<size_t>(LogProbePhase::Load)] > 0,
           "Model loading did not invoke the reentrant log callback");
    expect(model->MaxSequenceLength() == kCapacity && model->ActiveSessions() == 0,
           "Unexpected loaded-model limits or eager default session");
    const size_t loadedBytes = runtime->Memory().currentBytes;
    logProbe.phase = LogProbePhase::Session;
    a = model->CreateSession();
    expect(a->Capacity() == kCapacity && a->VocabSize() == model->VocabSize() && a->ContextLength() == 0,
           "Invalid initial Session A metadata");
    expect(a->CacheBytes() > 0 && a->CacheBytes() <= model->SessionReservationBytes(),
           "Session cache does not fit its advertised reservation");
    expect(model->ActiveSessions() == 1 && runtime->Memory().currentBytes == loadedBytes + a->CacheBytes(),
           "First session allocated more than independent KV");
    test_reject_quantized_session(runtime, model);
    test_shared_runtime_handle(runtime, model, a);
    auto b = model->CreateSession();
    expect(model->ActiveSessions() == 2 && runtime->Memory().currentBytes == loadedBytes + a->CacheBytes() + b->CacheBytes(),
           "Second session duplicated weights or workspace");
    const size_t beforeRejected = runtime->Memory().currentBytes;
    expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Initial session limit");
    expect(runtime->Memory().currentBytes == beforeRejected, "Rejected session allocated device memory");
    logProbe.Check("Session creation");
    const size_t sessionPhase = static_cast<size_t>(LogProbePhase::Session);
    expect(logProbe.modelReads[sessionPhase] > 0 && logProbe.sessionReads[sessionPhase] > 0,
           "Session creation did not invoke model/session getters from logging");

    logProbe.phase = LogProbePhase::Growth;
    const auto traces = test_interleaved_logits(a, b, planA, planB);
    logProbe.Check("Initial workspace growth");
    const size_t growthPhase = static_cast<size_t>(LogProbePhase::Growth);
    expect(logProbe.runtimeGrowthLogs[growthPhase] > 0 && logProbe.sessionReads[growthPhase] > 0,
           "First interleaved workspace growth did not invoke reentrant log getters");
    test_request_log_reentry(logProbe, a, promptA, loggingRequest);
    logProbe.Check("Completed log callback checks");
    logRegistration.Restore();
    expect(aila::get_log_level() == logRegistration.OriginalLevel(), "Log level was not restored");
    test_request_mode(a, b, promptA, promptB, false, false, "Greedy request interleaving");
    test_request_mode(a, b, promptA, promptB, true, false, "Fixed-seed sampling without penalties");
    const auto sampled = test_request_mode(a, b, promptA, promptB, true, true, "Fixed-seed sampling with penalties");
    test_cancel_and_destroy(runtime, model, a, b, promptA, promptB, planA, traces, sampled);
    a = model->CreateSession();
    test_reset_and_truncate(a, b, planA, planB, traces);
    test_limits(runtime, model, a, b, promptA, planA, planB, traces);
    test_release_and_recreate(runtime, model, a, b);
    test_owner_lifetime(runtime, model, planA, traces, promptA, sampled);
}

std::string model_directory_from_environment() {
#ifdef _WIN32
    char* value = nullptr;
    size_t size = 0;
    const auto error = _dupenv_s(&value, &size, "AILA_QWEN3_TEST_MODEL");
    const std::unique_ptr<char, decltype(&std::free)> owned(value, &std::free);
    expect(error == 0, "Could not read AILA_QWEN3_TEST_MODEL");
    return owned ? owned.get() : "";
#else
    const char* value = std::getenv("AILA_QWEN3_TEST_MODEL");
    return value ? value : "";
#endif
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string modelDirectory;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "Usage: Qwen3SessionTests --model <Qwen3-0.6B NF4 directory>\n"
                             "Fallback environment variable: AILA_QWEN3_TEST_MODEL\n";
                return 0;
            }
            if (argument != "--model" || i + 1 >= argc || !modelDirectory.empty()) {
                throw std::runtime_error("Expected --model <directory>");
            }
            modelDirectory = argv[++i];
        }
        if (modelDirectory.empty()) {
            modelDirectory = model_directory_from_environment();
        }
        if (modelDirectory.empty()) {
            std::cout << "[SKIP] Specify --model <directory> or AILA_QWEN3_TEST_MODEL for the real-model GPU test.\n";
            return 77;
        }
        expect(std::filesystem::is_directory(modelDirectory), "Model directory does not exist: " + modelDirectory);
        run_tests(modelDirectory);
        std::cout << "Qwen3 shared-session tests passed." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3 shared-session tests failed: " << error.what() << std::endl;
        return 1;
    }
}
