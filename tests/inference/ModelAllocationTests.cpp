#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "inference/Session.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace aila::inference;
using Tokens = std::vector<int>;
constexpr int kCapacity = 512;
constexpr int kGenerated = 6;
constexpr int kLargePrefill = 384;
constexpr size_t kMaximumBudgetSessions = 32;

struct Arguments {
    std::string model;
    std::string backend = "qwen3_nf4";
};

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
std::string ExpectError(ErrorCode code, Function&& function, const std::string& label) {
    try {
        function();
    } catch (const Error& error) {
        Expect(error.Code() == code, label + ": wrong error code: " + error.what());
        return error.what();
    }
    throw std::runtime_error(label + ": expected an inference error");
}

size_t CheckedAdd(size_t a, size_t b) {
    Expect(b <= std::numeric_limits<size_t>::max() - a, "Fixture memory sum overflow");
    return a + b;
}

size_t CheckedMultiply(size_t a, size_t b) {
    Expect(!a || b <= std::numeric_limits<size_t>::max() / a, "Fixture memory product overflow");
    return a * b;
}

bool SameMemory(const MemorySnapshot& a, const MemorySnapshot& b) {
    return std::tie(a.currentBytes, a.peakBytes, a.pinnedHostBytes, a.peakPinnedHostBytes,
                    a.deviceAllocations, a.pinnedHostAllocations, a.deviceLimitBytes,
                    a.pinnedHostLimitBytes) ==
           std::tie(b.currentBytes, b.peakBytes, b.pinnedHostBytes, b.peakPinnedHostBytes,
                    b.deviceAllocations, b.pinnedHostAllocations, b.deviceLimitBytes,
                    b.pinnedHostLimitBytes);
}

void ExpectCurrent(const MemorySnapshot& actual, const MemorySnapshot& expected,
                   const std::string& label) {
    Expect(actual.currentBytes == expected.currentBytes && actual.pinnedHostBytes == expected.pinnedHostBytes &&
           actual.deviceAllocations == expected.deviceAllocations &&
           actual.pinnedHostAllocations == expected.pinnedHostAllocations,
           label + ": tracked live USM changed; device bytes/count=" + std::to_string(actual.currentBytes) +
           "/" + std::to_string(actual.deviceAllocations) + " expected=" +
           std::to_string(expected.currentBytes) + "/" + std::to_string(expected.deviceAllocations));
}

void ExpectNoFault(const std::shared_ptr<DeviceRuntime>& runtime, const std::string& label) {
    const auto fault = runtime->FaultMessage();
    Expect(!fault, label + ": device domain faulted: " + fault.value_or(""));
}

void LogMemory(const std::shared_ptr<DeviceRuntime>& runtime, const std::string& stage) {
    const auto memory = runtime->Memory();
    std::cout << "[MEMORY] scope=context_tracked_usm stage=" << stage
              << " device_bytes=" << memory.currentBytes << " lifetime_peak_device_bytes=" << memory.peakBytes
              << " pinned_host_bytes=" << memory.pinnedHostBytes
              << " lifetime_peak_pinned_host_bytes=" << memory.peakPinnedHostBytes
              << " device_allocations=" << memory.deviceAllocations
              << " pinned_host_allocations=" << memory.pinnedHostAllocations
              << " device_limit=" << memory.deviceLimitBytes << " pinned_limit=" << memory.pinnedHostLimitBytes
              << std::endl;
}

void ExpectEmpty(const std::shared_ptr<DeviceRuntime>& runtime, const std::string& stage) {
    const auto memory = runtime->Memory();
    LogMemory(runtime, stage);
    Expect(memory.currentBytes == 0 && memory.pinnedHostBytes == 0 && memory.deviceAllocations == 0 &&
           memory.pinnedHostAllocations == 0, stage + ": model/session/workspace teardown retained tracked USM");
    ExpectNoFault(runtime, stage);
}

ModelOptions Options(size_t sessions = 4) {
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = sessions;
    options.maxSessionCacheBytes = 0;
    options.maxSessionHostBytes = 0;
    options.maxRecurrentSnapshots = 0;
    return options;
}

std::shared_ptr<LoadedModel> Load(const Arguments& arguments,
                                 const std::shared_ptr<DeviceRuntime>& runtime,
                                 ModelOptions options = Options()) {
    if (arguments.backend == "qwen3_nf4")
        return LoadedModel::LoadQwen3Nf4(runtime, arguments.model, options);
    if (arguments.backend == "qwen35_nf4")
        return LoadedModel::LoadQwen35Nf4(runtime, arguments.model, options);
    return LoadedModel::LoadQwen35Bf16(runtime, arguments.model, options);
}

RequestOptions Greedy(int chunk = 8, int generated = kGenerated) {
    RequestOptions options;
    options.prefillChunkSize = chunk;
    options.generation.do_sample = false;
    options.generation.temperature = 1.0f;
    options.generation.top_k = 1;
    options.generation.top_p = 1.0f;
    options.generation.max_new_tokens = generated;
    options.generation.thinking_budget_tokens = -1;
    return options;
}

Tokens Repeat(const Tokens& source, size_t count) {
    Expect(!source.empty(), "Tokenizer produced an empty fixture");
    Tokens result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(source[i % source.size()]);
    return result;
}

struct Fixtures {
    Tokens initial, continuation, sampling, large;
};

Fixtures MakeFixtures(const std::string& directory) {
    Tokenizer tokenizer;
    Expect(tokenizer.load(directory), "Could not load model tokenizer");
    return {Repeat(tokenizer.encode("A gardener records rainfall and compares the growth of young trees. "), 17),
            Repeat(tokenizer.encode(" Continue with a concrete example. "), 9),
            Repeat(tokenizer.encode("A robot sorts red and blue blocks and checks each result. "), 33),
            Repeat(tokenizer.encode("Describe the measurements in a laboratory notebook with dates and units. "), kLargePrefill)};
}

struct Trace {
    Tokens tokens;
    size_t terminalEvents = 0;
};

StepResult Step(const std::shared_ptr<RequestState>& request, Trace& trace) {
    const auto step = request->Step();
    if (step.tokenId) trace.tokens.push_back(*step.tokenId);
    if (step.becameTerminal) ++trace.terminalEvents;
    return step;
}

void Drain(const std::shared_ptr<RequestState>& request, Trace& trace, const std::string& label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    for (int iteration = 0; iteration < 2 * kCapacity; ++iteration) {
        if (IsTerminal(request->Progress().status)) break;
        Expect(std::chrono::steady_clock::now() < deadline, label + ": request exceeded its time bound");
        Step(request, trace);
    }
    const auto snapshot = request->Snapshot();
    Expect(IsTerminal(snapshot.status) && trace.terminalEvents == 1,
           label + ": request did not deliver exactly one terminal transition");
    Expect(snapshot.generatedIds == trace.tokens, label + ": snapshot/token-event mismatch");
    const auto repeated = request->Step();
    Expect(repeated.status == snapshot.status && !repeated.tokenId && !repeated.becameTerminal,
           label + ": terminal request submitted another step");
}

void ExpectCompleted(const std::shared_ptr<RequestState>& request,
                     const std::shared_ptr<SessionState>& session, const Trace& trace,
                     size_t promptLength, int startLength, int generated, const std::string& label) {
    const auto snapshot = request->Snapshot();
    Expect(snapshot.status == RequestStatus::Completed && !snapshot.errorCode && snapshot.error.empty(),
           label + ": request failed: " + snapshot.error);
    Expect(snapshot.promptTokensProcessed == promptLength && trace.tokens.size() == static_cast<size_t>(generated),
           label + ": wrong prompt/generation count");
    Expect(!session->Busy() && session->ContextLength() == startLength + static_cast<int>(promptLength) + generated,
           label + ": final token was not consumed into the session cache");
}

Tokens Run(const std::shared_ptr<SessionState>& session, const Tokens& prompt,
           RequestOptions options, const std::string& label) {
    const int start = session->ContextLength();
    auto request = RequestState::Create(session, prompt, options);
    Trace trace;
    Drain(request, trace, label);
    ExpectCompleted(request, session, trace, prompt.size(), start, options.generation.max_new_tokens, label);
    return trace.tokens;
}

void ExpectTokens(const Tokens& actual, const Tokens& expected, const std::string& label) {
    Expect(actual.size() == expected.size(), label + ": generated length differs from independent reference");
    for (size_t i = 0; i < expected.size(); ++i)
        Expect(actual[i] == expected[i], label + ": token " + std::to_string(i) + " actual=" +
               std::to_string(actual[i]) + " reference=" + std::to_string(expected[i]));
}

void AdvanceToFirstToken(const std::shared_ptr<RequestState>& request, Trace& trace,
                         const std::string& label) {
    for (int iteration = 0; iteration < kCapacity && trace.tokens.empty(); ++iteration) {
        const auto step = Step(request, trace);
        Expect(!IsTerminal(step.status), label + ": peer terminated before its first decode boundary");
    }
    Expect(trace.tokens.size() == 1 && request->Progress().status == RequestStatus::Decode,
           label + ": peer did not reach a live decode boundary");
}

void ExpectPeerUnchanged(const std::shared_ptr<RequestState>& request,
                         const std::shared_ptr<SessionState>& session,
                         const RequestSnapshot& before, int contextLength, const std::string& label) {
    const auto after = request->Snapshot();
    Expect(after.status == before.status && after.promptTokensProcessed == before.promptTokensProcessed &&
           after.generatedIds == before.generatedIds && after.errorCode == before.errorCode &&
           after.error == before.error && session->ContextLength() == contextLength && session->Busy(),
           label + ": local resource rejection changed the peer's request or context");
}

void CheckLiveModelLimitLock(const std::shared_ptr<DeviceRuntime>& runtime) {
    const auto before = runtime->Memory();
    ExpectError(ErrorCode::Busy, [&] { runtime->SetAllocationLimits(1, 1); }, "Changing limits with a live model");
    Expect(SameMemory(before, runtime->Memory()), "Busy limit update changed limits, allocations or peaks");
    ExpectNoFault(runtime, "Busy limit update");
}

struct Baseline {
    MemorySnapshot loaded, warmWithTwoSessions, largePrefill;
    size_t cacheBytes = 0;
    size_t cacheAllocations = 0;
    Tokens initial, continuation;
    bool prefillEligible = false;
};

Baseline BuildBaseline(const Arguments& arguments, const Fixtures& fixtures) {
    std::cout << "[RUN] one-byte-device-load-rejection-and-recovery" << std::endl;
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "fresh-baseline-runtime");
    runtime->SetAllocationLimits(1, 0);
    const auto beforeRejectedLoad = runtime->Memory();
    const auto loadError = ExpectError(ErrorCode::ResourceLimit, [&] { Load(arguments, runtime); }, "One-byte load budget");
    Expect(loadError.find("Device") != std::string::npos, "Load rejection did not identify the device allocation ceiling");
    const auto afterRejectedLoad = runtime->Memory();
    ExpectCurrent(afterRejectedLoad, beforeRejectedLoad, "Rejected one-byte load cleanup");
    Expect(afterRejectedLoad.deviceLimitBytes == 1 && afterRejectedLoad.pinnedHostLimitBytes == 0 &&
           afterRejectedLoad.peakBytes <= 1 && afterRejectedLoad.peakPinnedHostBytes == 0,
           "Rejected one-byte load changed its limits or exceeded the device ceiling");
    ExpectEmpty(runtime, "after-rejected-load");

    // Recovery uses this same execution domain. Later phases each obtain a
    // fresh domain, so their lifetime peaks do not include this baseline.
    runtime->SetAllocationLimits(0, 0);
    auto model = Load(arguments, runtime);
    Baseline baseline;
    baseline.loaded = runtime->Memory();
    baseline.cacheBytes = model->SessionReservationBytes();
    Expect(model->MaxSequenceLength() == kCapacity && baseline.cacheBytes > 0,
           "Fixture requires context 512 and a positive fixed session reservation");
    CheckLiveModelLimitLock(runtime);
    LogMemory(runtime, "baseline-loaded");
    std::cout << "[PASS] load-resource-limit-is-local-and-recoverable" << std::endl;
    std::cout << "[PASS] live-model-budget-update-is-busy-and-atomic" << std::endl;

    auto a = model->CreateSession();
    const auto firstSession = runtime->Memory();
    Expect(a->CacheBytes() == baseline.cacheBytes && firstSession.currentBytes ==
           CheckedAdd(baseline.loaded.currentBytes, baseline.cacheBytes), "Session reservation differs from actual device payload");
    Expect(firstSession.deviceAllocations > baseline.loaded.deviceAllocations &&
           firstSession.pinnedHostBytes == baseline.loaded.pinnedHostBytes &&
           firstSession.pinnedHostAllocations == baseline.loaded.pinnedHostAllocations,
           "Session allocation profile is not a fixed device-only USM delta");
    baseline.cacheAllocations = firstSession.deviceAllocations - baseline.loaded.deviceAllocations;
    auto b = model->CreateSession();
    Expect(runtime->Memory().currentBytes == CheckedAdd(firstSession.currentBytes, baseline.cacheBytes),
           "Second session duplicated shared model resources");
    baseline.initial = Run(b, fixtures.initial, Greedy(), "Independent initial reference");
    baseline.continuation = Run(b, fixtures.continuation, Greedy(4), "Independent continuation reference");
    baseline.warmWithTwoSessions = runtime->Memory();
    Expect(baseline.warmWithTwoSessions.pinnedHostBytes == 0 &&
           baseline.warmWithTwoSessions.peakPinnedHostBytes == 0 &&
           baseline.warmWithTwoSessions.pinnedHostAllocations == 0,
           "Greedy reference unexpectedly required pinned-host workspace");
    Expect(a->ContextLength() == 0, "Independent reference contaminated the idle session");
    LogMemory(runtime, "baseline-short-requests-two-sessions");

    // One bounded large prefill profiles the optional device-workspace case.
    // Only enable it if the resulting live payload alone exceeds the cap that
    // already admitted loading, two sessions and all peer request shapes.
    auto probe = RequestState::Create(a, fixtures.large, Greedy(kLargePrefill, 1));
    Trace probeTrace;
    const auto firstLarge = Step(probe, probeTrace);
    Expect(firstLarge.status == RequestStatus::Finalizing && firstLarge.tokenId && !firstLarge.becameTerminal,
           "Large-prefill profile failed: " + probe->Snapshot().error);
    baseline.largePrefill = runtime->Memory();
    baseline.prefillEligible = baseline.largePrefill.currentBytes > baseline.warmWithTwoSessions.peakBytes;
    LogMemory(runtime, "baseline-large-prefill");
    std::cout << "[PROFILE] session_cache_bytes=" << baseline.cacheBytes
              << " session_device_allocations=" << baseline.cacheAllocations
              << " session_pageable_host_reservation_bytes=" << model->SessionHostReservationBytes()
              << " prefill_candidate_cap=" << baseline.warmWithTwoSessions.peakBytes
              << " large_prefill_live_bytes=" << baseline.largePrefill.currentBytes
              << " prefill_eligible=" << (baseline.prefillEligible ? "true" : "false") << std::endl;
    Drain(probe, probeTrace, "Large-prefill profile");
    ExpectCompleted(probe, a, probeTrace, fixtures.large.size(), 0, 1, "Large-prefill profile");
    probe.reset();
    a.reset();
    b.reset();
    Expect(model->ActiveSessions() == 0, "Baseline retained a session owner");
    model.reset();
    ExpectEmpty(runtime, "baseline-model-destroyed");
    return baseline;
}

void TestDeviceSessionBudget(const Arguments& arguments, const Baseline& baseline) {
    std::cout << "[RUN] finite-device-session-ceiling-and-reclamation" << std::endl;
    const size_t twoSessions = CheckedAdd(baseline.loaded.currentBytes, CheckedMultiply(2, baseline.cacheBytes));
    const size_t cap = std::max(baseline.loaded.peakBytes, twoSessions);
    const size_t predicted = (cap - baseline.loaded.currentBytes) / baseline.cacheBytes;
    Expect(predicted >= 2 && predicted <= kMaximumBudgetSessions,
           "Measured load peak requires " + std::to_string(predicted) + " sessions; fixture bound is 2..32");
    const auto options = Options(predicted + 2);
    std::cout << "[BUDGET] algorithm=max(load_peak,resident+2*cache) load_peak=" << baseline.loaded.peakBytes
              << " resident=" << baseline.loaded.currentBytes << " cache=" << baseline.cacheBytes
              << " device_cap=" << cap << " expected_sessions=" << predicted
              << " max_sessions=" << options.maxSessions << " session_byte_gates=disabled" << std::endl;
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "fresh-device-budget-runtime");
    runtime->SetAllocationLimits(cap, 0);
    auto model = Load(arguments, runtime, options);
    const auto resident = runtime->Memory();
    ExpectCurrent(resident, baseline.loaded, "Bounded reload resident payload");
    Expect(resident.peakBytes <= cap && model->SessionReservationBytes() == baseline.cacheBytes,
           "Finite cap did not reproduce the measured loading/cache profile");
    CheckLiveModelLimitLock(runtime);

    std::vector<std::shared_ptr<SessionState>> sessions;
    sessions.reserve(predicted);
    for (size_t i = 0; i < predicted; ++i) {
        sessions.push_back(model->CreateSession());
        const auto current = runtime->Memory();
        Expect(current.currentBytes == CheckedAdd(resident.currentBytes, CheckedMultiply(i + 1, baseline.cacheBytes)) &&
               current.deviceAllocations == CheckedAdd(resident.deviceAllocations, CheckedMultiply(i + 1, baseline.cacheAllocations)) &&
               current.pinnedHostBytes == resident.pinnedHostBytes &&
               current.pinnedHostAllocations == resident.pinnedHostAllocations && current.peakBytes <= cap,
               "Successful bounded session changed shared payload or exceeded its device cap");
        Expect(sessions.back()->CacheBytes() == baseline.cacheBytes && sessions.back()->ContextLength() == 0,
               "New bounded session reported incorrect metadata");
    }
    Expect(model->ActiveSessions() == predicted && predicted < options.maxSessions,
           "Session-count gate would hide the device ceiling");
    const auto full = runtime->Memory();
    const auto error = ExpectError(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Exhausted device session budget");
    Expect(error.find("Device") != std::string::npos && error.find("budget") != std::string::npos,
           "Session admission did not reject at the Context device ceiling: " + error);
    Expect(SameMemory(full, runtime->Memory()) && model->ActiveSessions() == predicted,
           "Rejected session changed current/peak/count/limits or consumed a session slot");
    ExpectNoFault(runtime, "Device session rejection");
    LogMemory(runtime, "device-session-budget-full");

    sessions.back().reset();
    const auto released = runtime->Memory();
    Expect(released.currentBytes == full.currentBytes - baseline.cacheBytes &&
           released.deviceAllocations == full.deviceAllocations - baseline.cacheAllocations &&
           model->ActiveSessions() == predicted - 1, "Released session did not return exact device credit");
    sessions.back() = model->CreateSession();
    Expect(SameMemory(full, runtime->Memory()) && model->ActiveSessions() == predicted,
           "Replacement session did not reuse returned device credit");
    ExpectError(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Restored device ceiling");
    Expect(SameMemory(full, runtime->Memory()), "Second device rejection changed the allocation ledger");
    sessions.clear();
    Expect(model->ActiveSessions() == 0, "Bounded session vector retained credits");
    ExpectCurrent(runtime->Memory(), resident, "All bounded sessions released");
    model.reset();
    ExpectEmpty(runtime, "device-budget-model-destroyed");
    std::cout << "[PASS] finite-device-session-ceiling-and-reclamation" << std::endl;
}

void FinishPeer(const std::shared_ptr<RequestState>& request, const std::shared_ptr<SessionState>& peer,
                Trace& trace, const Fixtures& fixtures, const Baseline& baseline, const std::string& label) {
    Drain(request, trace, label + " initial");
    ExpectCompleted(request, peer, trace, fixtures.initial.size(), 0, kGenerated, label + " initial");
    ExpectTokens(trace.tokens, baseline.initial, label + " initial");
    ExpectTokens(Run(peer, fixtures.continuation, Greedy(4), label + " continuation"),
                 baseline.continuation, label + " continuation");
}

void TestPinnedSamplingBudget(const Arguments& arguments, const Fixtures& fixtures, const Baseline& baseline) {
    std::cout << "[RUN] pinned-sampling-resource-limit-with-live-peer" << std::endl;
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "fresh-pinned-budget-runtime");
    runtime->SetAllocationLimits(0, 1);
    auto model = Load(arguments, runtime);
    CheckLiveModelLimitLock(runtime);
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    auto peer = RequestState::Create(b, fixtures.initial, Greedy());
    Trace peerTrace;
    AdvanceToFirstToken(peer, peerTrace, "Pinned-budget peer");
    const auto beforePeer = peer->Snapshot();
    const int peerLength = b->ContextLength();
    const auto beforeSampling = runtime->Memory();

    auto sampling = Greedy();
    sampling.generation.do_sample = true;
    sampling.generation.top_k = 4;
    sampling.generation.temperature = 0.7f;
    sampling.generation.presence_penalty = 0.2f;
    sampling.generation.use_fixed_seed = true;
    sampling.generation.sampling_seed = 123;
    Expect(sampling.generation.has_penalties(), "Fixture must force the host sampling path");
    auto rejected = RequestState::Create(a, fixtures.sampling, sampling);
    Trace failedTrace;
    Drain(rejected, failedTrace, "Pinned-host sampling budget");
    const auto failure = rejected->Snapshot();
    Expect(failure.status == RequestStatus::Failed && failure.errorCode == ErrorCode::ResourceLimit &&
           failure.error.find("Pinned host") != std::string::npos && failedTrace.tokens.empty(),
           "Sampling did not fail locally at the pinned-host ceiling: " + failure.error);
    Expect(failure.promptTokensProcessed == 32 && a->ContextLength() == 0 && !a->Busy(),
           "Sampling rejection did not reset partially prefetched session A and release its request");
    const auto afterSampling = runtime->Memory();
    Expect(afterSampling.pinnedHostBytes == 0 && afterSampling.pinnedHostAllocations == 0 &&
           afterSampling.peakPinnedHostBytes == beforeSampling.peakPinnedHostBytes &&
           afterSampling.pinnedHostLimitBytes == 1, "Rejected pinned allocation changed its ledger or limit");
    ExpectNoFault(runtime, "Pinned sampling rejection");
    ExpectPeerUnchanged(peer, b, beforePeer, peerLength, "Pinned sampling rejection");
    FinishPeer(peer, b, peerTrace, fixtures, baseline, "Pinned-budget surviving B");
    ExpectTokens(Run(a, fixtures.initial, Greedy(), "Pinned-budget reset A recovery"),
                 baseline.initial, "Pinned-budget reset A recovery");
    LogMemory(runtime, "pinned-budget-peers-recovered");
    rejected.reset();
    peer.reset();
    a.reset();
    b.reset();
    Expect(model->ActiveSessions() == 0, "Pinned-budget terminal requests retained a session");
    model.reset();
    ExpectEmpty(runtime, "pinned-budget-model-destroyed");
    std::cout << "[PASS] pinned-sampling-resource-limit-with-live-peer" << std::endl;
}

void TestPrefillBudget(const Arguments& arguments, const Fixtures& fixtures, const Baseline& baseline) {
    if (!baseline.prefillEligible) {
        std::cout << "[NOT_COVERED] large-prefill-device-budget: measured live prefill payload does not exceed "
                     "the loading/short-request peak; no forced OOM or weaker assertion substituted." << std::endl;
        return;
    }
    const size_t cap = baseline.warmWithTwoSessions.peakBytes;
    std::cout << "[RUN] large-prefill-local-resource-limit-with-live-peer device_cap=" << cap
              << " baseline_large_live_bytes=" << baseline.largePrefill.currentBytes << std::endl;
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "fresh-prefill-budget-runtime");
    runtime->SetAllocationLimits(cap, 0);
    auto model = Load(arguments, runtime);
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    // Reproduce every reference shape with two sessions alive, then retain B
    // at a decode boundary while A asks the shared slot for the large shape.
    ExpectTokens(Run(a, fixtures.initial, Greedy(), "Prefill-budget warm initial"),
                 baseline.initial, "Prefill-budget warm initial");
    ExpectTokens(Run(a, fixtures.continuation, Greedy(4), "Prefill-budget warm continuation"),
                 baseline.continuation, "Prefill-budget warm continuation");
    a->Reset();
    auto peer = RequestState::Create(b, fixtures.initial, Greedy());
    Trace peerTrace;
    AdvanceToFirstToken(peer, peerTrace, "Prefill-budget peer");
    const auto beforePeer = peer->Snapshot();
    const int peerLength = b->ContextLength();
    ExpectCurrent(runtime->Memory(), baseline.warmWithTwoSessions, "Prefill-budget reproduced warm payload");
    auto rejected = RequestState::Create(a, fixtures.large, Greedy(kLargePrefill, 1));
    Trace failureTrace;
    const auto step = Step(rejected, failureTrace);
    const auto failure = rejected->Snapshot();
    Expect(step.status == RequestStatus::Failed && step.becameTerminal && !step.tokenId &&
           failure.errorCode == ErrorCode::ResourceLimit && failure.error.find("Device") != std::string::npos,
           "Measured large prefill did not produce local device ResourceLimit: " + failure.error);
    Drain(rejected, failureTrace, "Large-prefill resource limit");
    Expect(failure.promptTokensProcessed == 0 && a->ContextLength() == 0 && !a->Busy(),
           "Failed large prefill did not reset A before publishing its terminal result");
    const auto afterFailure = runtime->Memory();
    Expect(afterFailure.currentBytes <= cap && afterFailure.peakBytes <= cap,
           "Large prefill exceeded the configured explicit device budget");
    // Some larger execution buffers can survive a partially successful growth;
    // they remain accounted. Peer correctness and final teardown are mandatory.
    ExpectNoFault(runtime, "Large-prefill rejection");
    ExpectPeerUnchanged(peer, b, beforePeer, peerLength, "Large-prefill rejection");
    FinishPeer(peer, b, peerTrace, fixtures, baseline, "Prefill-budget surviving B");
    ExpectTokens(Run(a, fixtures.initial, Greedy(), "Prefill-budget reset A recovery"),
                 baseline.initial, "Prefill-budget reset A recovery");
    LogMemory(runtime, "prefill-budget-peers-recovered");
    rejected.reset();
    peer.reset();
    a.reset();
    b.reset();
    Expect(model->ActiveSessions() == 0, "Prefill-budget terminal requests retained a session");
    model.reset();
    ExpectEmpty(runtime, "prefill-budget-model-destroyed");
    std::cout << "[PASS] large-prefill-local-resource-limit-with-live-peer" << std::endl;
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    Expect(_putenv_s(name, value) == 0, "Could not configure fixture environment");
#else
    Expect(setenv(name, value, 1) == 0, "Could not configure fixture environment");
#endif
}

} // namespace

int main(int argc, char** argv) {
    try {
        Arguments arguments;
        bool modelSpecified = false, backendSpecified = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "Usage: ModelAllocationTests --model <directory>\n"
                             "                            [--backend qwen3_nf4|qwen35_nf4|qwen35_bf16]\n"
                             "Default backend: qwen3_nf4. Context 512, six greedy tokens per reference leg.\n"
                             "Serial GPU fixture; limits cover Context-owned USM, not total RSS/VRAM.\n"
                             "The optional large-prefill case reports NOT_COVERED when its measured\n"
                             "working set cannot be separated from the mandatory loading peak.\n";
                return 0;
            }
            if (argument == "--model") {
                Expect(!modelSpecified && i + 1 < argc, "Expected one --model <directory>");
                modelSpecified = true;
                arguments.model = argv[++i];
                Expect(!arguments.model.empty(), "Model directory must not be empty");
            } else if (argument == "--backend") {
                Expect(!backendSpecified && i + 1 < argc, "Expected one --backend <name>");
                backendSpecified = true;
                arguments.backend = argv[++i];
                Expect(arguments.backend == "qwen3_nf4" || arguments.backend == "qwen35_nf4" ||
                       arguments.backend == "qwen35_bf16", "Unsupported backend: " + arguments.backend);
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }
        if (!modelSpecified) {
            std::cout << "[SKIP] Specify --model <directory> for the real-model allocation fixture." << std::endl;
            return 77;
        }
        Expect(std::filesystem::is_directory(arguments.model), "Model directory does not exist: " + arguments.model);
        // Process-local settings establish the supported shared-session KV and
        // recurrent path before any cached kernel setting is first evaluated.
        SetEnvironment("AILA_KV_QUANT", "0");
        SetEnvironment("AILA_VLM_KV_QUANT", "0");
        SetEnvironment("AILA_Q35_LINEAR_DELTA", "1");
        aila::env::g_kv_quant_override = false;
        aila::set_log_level(aila::LogLevel::Warning);
        std::cout << "[CONFIG] backend=" << arguments.backend << " model=" << arguments.model
                  << " context=" << kCapacity << " recurrent_snapshots=0 kv=bf16 linear_delta=1 gpu_execution=serial\n"
                     "[SCOPE] Context-owned device/pinned USM only; excludes driver/library internals, "
                     "pageable host mirrors, allocator overhead and total process RSS/VRAM." << std::endl;
        const auto fixtures = MakeFixtures(arguments.model);
        const auto baseline = BuildBaseline(arguments, fixtures);
        TestDeviceSessionBudget(arguments, baseline);
        TestPinnedSamplingBudget(arguments, fixtures, baseline);
        TestPrefillBudget(arguments, fixtures, baseline);
        std::cout << "Model allocation mandatory cases passed; large_prefill_budget="
                  << (baseline.prefillEligible ? "passed" : "not_covered") << '.' << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Model allocation test failed: " << error.what() << std::endl;
        return 1;
    }
}
