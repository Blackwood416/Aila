#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <TlHelp32.h>

#include "runtime/InferenceClient.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cwctype>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_BUILD_ID
#error "Inference pipe tests require AILA_BUILD_ID"
#endif
#ifndef AILA_FAKE_INFERENCE_WORKER_PATH
#error "Inference pipe tests require the absolute fake inference Worker executable path"
#endif

namespace {

namespace fs = std::filesystem;
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr int32_t CrashPromptToken = 1023;
constexpr uint32_t SessionCapacity = 2048;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Predicate>
void WaitUntil(Predicate predicate, const std::string& message, std::chrono::milliseconds timeout = 3s) {
    const auto deadline = Clock::now() + timeout;
    do {
        if (predicate()) return;
        std::this_thread::sleep_for(2ms);
    } while (Clock::now() < deadline);
    throw std::runtime_error(message);
}

template <typename Function>
void ExpectRemoteError(Function function, wire::ErrorCode code) {
    try { function(); }
    catch (const host::InferenceClientError& error) {
        Expect(error.RemoteCode() == code, "Unexpected remote error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected a remote inference error");
}

template <typename Function>
void ExpectUnavailable(Function function) {
    try { function(); }
    catch (const host::InferenceClientError& error) {
        Expect(error.Code() == host::ClientErrorCode::Closed || error.Code() == host::ClientErrorCode::InvalidHandle,
               "Old/released handle failed for an unexpected reason: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Old or released inference handle was still usable");
}

std::optional<std::wstring> EnvironmentValue(const wchar_t* name) {
    SetLastError(ERROR_SUCCESS);
    const DWORD size = GetEnvironmentVariableW(name, nullptr, 0);
    if (size == 0) {
        if (GetLastError() == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
        Expect(GetLastError() == ERROR_SUCCESS, "Could not inspect host environment");
        return std::wstring{};
    }
    std::vector<wchar_t> buffer(size);
    const DWORD copied = GetEnvironmentVariableW(name, buffer.data(), size);
    Expect(copied < size, "Host environment changed during inspection");
    return std::wstring(buffer.data(), copied);
}

DWORD ProcessHandleCount() {
    DWORD count = 0;
    Expect(GetProcessHandleCount(GetCurrentProcess(), &count) != FALSE, "GetProcessHandleCount failed");
    return count;
}

class SnapshotHandle final {
public:
    explicit SnapshotHandle(HANDLE value) : value(value) {}
    ~SnapshotHandle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    SnapshotHandle(const SnapshotHandle&) = delete;
    SnapshotHandle& operator=(const SnapshotHandle&) = delete;
    HANDLE Get() const noexcept { return value; }
private:
    HANDLE value;
};

void ExpectNoGpuRuntime() {
    SnapshotHandle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId()));
    Expect(snapshot.Get() != INVALID_HANDLE_VALUE, "Could not enumerate host modules");
    MODULEENTRY32W module{};
    module.dwSize = sizeof(module);
    Expect(Module32FirstW(snapshot.Get(), &module) != FALSE, "Could not read host module list");
    do {
        std::wstring name(module.szModule);
        std::transform(name.begin(), name.end(), name.begin(), [](wchar_t value) {
            return static_cast<wchar_t>(std::towlower(value));
        });
        Expect(name.find(L"sycl") == std::wstring::npos && name.find(L"dnnl") == std::wstring::npos &&
                   name.find(L"onednn") == std::wstring::npos,
               "Ordinary C++ inference host loaded a SYCL/oneDNN module");
    } while (Module32NextW(snapshot.Get(), &module));
    Expect(GetLastError() == ERROR_NO_MORE_FILES, "Host module enumeration failed");
}

struct HostIsolation {
    fs::path directory = fs::current_path();
    std::optional<std::wstring> path = EnvironmentValue(L"PATH");
    void Check() const {
        Expect(fs::current_path() == directory, "Starting/stopping an inference Worker changed host cwd");
        Expect(EnvironmentValue(L"PATH") == path, "Starting/stopping an inference Worker changed host PATH");
        ExpectNoGpuRuntime();
    }
};

host::InferenceClientOptions ClientOptions() {
    host::InferenceClientOptions options;
    auto& limits = options.limits;
    limits.maxActiveRequests = 2;
    limits.maxOutstandingRequests = 4;
    limits.maxPromptTokens = 64;
    limits.maxGeneratedTokens = 256;
    limits.maxOutstandingTokens = 2048;
    limits.maxPrefillChunkSize = 32;
    limits.maxSessions = 2;
    limits.maxSessionCacheBytes = 2ULL * SessionCapacity * 16;
    limits.maxRetainedRequestHandles = 8;
    limits.tokenWindow = 8;
    limits.maxHeaderBytes = 16 * 1024;
    limits.maxAttachmentBytes = 64 * sizeof(int32_t);
    limits.maxPendingCommands = 16;
    limits.reservedControlCommands = 4;
    limits.maxIngressBytes = 2 * 1024 * 1024;
    limits.maxTransportBytes = 1024 * 1024;
    limits.maxLogQueueBytes = 8192;
    limits.maxShutdownGraceMs = 5000;
    limits.connectionWriteTimeoutMs = 2000;
    options.handshakeTimeout = 3s;
    options.commandTimeout = 3s;
    options.shutdownTimeout = 5s;
    return options;
}

host::InferenceRequestOptions RequestOptions(uint32_t count) {
    host::InferenceRequestOptions options;
    options.generation.maxNewTokens = count;
    options.generation.doSample = false;
    options.generation.useFixedSeed = true;
    options.prefillChunkSize = 16;
    return options;
}

class Run final {
public:
    explicit Run(host::InferenceClientOptions options = ClientOptions(), bool hybridModel = false)
        : hybrid(hybridModel), limits(options.limits) {
        const fs::path executable = fs::absolute(fs::path(AILA_FAKE_INFERENCE_WORKER_PATH)).lexically_normal();
        Expect(fs::is_regular_file(executable), "Fake inference Worker executable is missing");
        client = host::InferenceClient::Start(executable.parent_path(), executable, AILA_BUILD_ID, std::move(options));
        model = Load();
        Expect(model->Info().vocabSize == 2048 && model->Info().maxSequenceLength == SessionCapacity,
               "Fake model metadata changed in transit");
        Expect(model->Epoch() == client->Snapshot().epoch && wire::valid_epoch(model->Epoch()),
               "Model was not bound to the negotiated epoch");
        const auto& capabilities = model->Info().capabilities;
        const auto expected = hybrid ? "qwen35_nf4_bf16_kv" : "qwen3_nf4_bf16_kv";
        const auto other = hybrid ? "qwen3_nf4_bf16_kv" : "qwen35_nf4_bf16_kv";
        Expect(std::find(capabilities.begin(), capabilities.end(), expected) != capabilities.end() &&
                   std::find(capabilities.begin(), capabilities.end(), other) == capabilities.end(),
               "Loaded model capabilities do not match the selected backend");
    }
    ~Run() {
        if (client) {
            try { client->Shutdown(5s); } catch (...) {}
        }
    }
    std::shared_ptr<host::InferenceModel> Load() {
        host::InferenceModelOptions options;
        options.maxSequenceLength = SessionCapacity;
        options.maxSessions = limits.maxSessions;
        options.maxSessionCacheBytes = limits.maxSessionCacheBytes;
        return hybrid ? client->LoadQwen35("fake-pipe-hybrid-model", options, 3s)
                      : client->LoadQwen3("fake-pipe-model", options, 3s);
    }
    void ExpectBounds() const {
        const auto local = client->Snapshot();
        Expect(local.outstandingRequests <= limits.maxOutstandingRequests &&
                   local.retainedRequestHandles <= limits.maxRetainedRequestHandles &&
                   local.pendingCommands <= limits.maxPendingCommands &&
                   local.reservedTokens <= limits.maxOutstandingTokens &&
                   local.peakIngressBytes <= limits.maxIngressBytes && local.peakTransportBytes <= limits.maxTransportBytes,
               "Host exceeded negotiated queue/reservation bounds");
    }
    void WaitForNoRequests() {
        WaitUntil([&] {
            const auto remote = client->Query(1s);
            const auto local = client->Snapshot();
            ExpectBounds();
            Expect(remote.ingressBytes <= limits.maxIngressBytes && remote.transportBytes <= limits.maxTransportBytes,
                   "Worker exceeded negotiated byte reservations");
            return remote.outstandingRequests == 0 && remote.ipcOutstandingRequests == 0 &&
                remote.retainedRequestHandles == 0 && remote.reservedTokens == 0 &&
                local.outstandingRequests == 0 && local.retainedRequestHandles == 0 &&
                local.reservedTokens == 0 && local.transportBytes == 0;
        }, "Released requests did not return host/Worker reservations");
    }
    const bool hybrid;
    wire::Limits limits;
    std::shared_ptr<host::InferenceClient> client;
    std::shared_ptr<host::InferenceModel> model;
};

host::InferenceReadEvent ReadEvent(const std::shared_ptr<host::InferenceRequest>& request) {
    const auto deadline = Clock::now() + 3s;
    do {
        auto result = request->Read(100ms);
        if (result.status == host::InferenceReadStatus::Timeout) continue;
        Expect(result.status == host::InferenceReadStatus::Event && result.event.has_value(),
               "Inference stream closed before its terminal");
        return std::move(*result.event);
    } while (Clock::now() < deadline);
    throw std::runtime_error("Inference event timed out");
}

struct Captured {
    std::vector<int32_t> tokens;
    host::InferenceReadEvent terminal;
};

Captured Drain(const std::shared_ptr<host::InferenceRequest>& request, std::vector<int32_t> prefix = {}) {
    Captured captured;
    captured.tokens = std::move(prefix);
    const auto deadline = Clock::now() + 5s;
    do {
        auto result = request->Read(100ms);
        if (result.status == host::InferenceReadStatus::Timeout) continue;
        Expect(result.status == host::InferenceReadStatus::Event && result.event.has_value(), "Stream ended without terminal");
        auto event = std::move(*result.event);
        Expect(!event.localFailure && !event.deliveryAbandoned && event.discardedPublishedTokens == 0,
               "Healthy user-consumed stream was abandoned or failed locally");
        Expect(event.sequence == captured.tokens.size() + 1, "User-visible stream sequence has a gap");
        if (event.terminal) {
            Expect(event.summary.generatedTokens == captured.tokens.size() &&
                       event.summary.eventCount == event.sequence && event.deliveredTokens == captured.tokens.size(),
                   "Terminal did not summarize the delivered token prefix");
            captured.terminal = std::move(event);
            Expect(request->Read(0ms).status == host::InferenceReadStatus::Closed, "Terminal was delivered more than once");
            return captured;
        }
        Expect(event.tokenId >= 0 && event.tokenId < 1000 && event.summary.generatedTokens == event.sequence,
               "CPU token payload/progress was corrupted");
        captured.tokens.push_back(event.tokenId);
    } while (Clock::now() < deadline);
    throw std::runtime_error("Inference drain timed out");
}

std::vector<int32_t> ExpectedTokens(const std::vector<int32_t>& prompt, uint32_t count, uint32_t initialLength = 0) {
    uint32_t length = initialLength + static_cast<uint32_t>(prompt.size());
    int32_t token = prompt.back();
    std::vector<int32_t> result;
    for (uint32_t index = 0; index < count; ++index) {
        token = (token + static_cast<int32_t>(length) + 1) % 1000;
        result.push_back(token);
        ++length;
    }
    return result;
}

void ExpectCompleted(const Captured& captured, const std::vector<int32_t>& expected) {
    Expect(captured.tokens == expected, "Session output changed under interleaving or cache continuation");
    Expect(captured.terminal.summary.status == wire::RequestStatus::Completed &&
               captured.terminal.summary.endReason == wire::EndReason::Length && !captured.terminal.summary.errorCode,
           "Healthy CPU request did not finish at its requested length");
}

void ConcurrentSessions(bool hybrid) {
    Run run(ClientOptions(), hybrid);
    const auto a = run.model->CreateSession();
    const auto b = run.model->CreateSession();
    const std::vector<int32_t> promptA{11, 12, 13}, promptB{71, 72, 73};
    const auto requestA = a->Submit(promptA, RequestOptions(96));
    const auto requestB = b->Submit(promptB, RequestOptions(80));
    Expect(requestA->Id() != requestB->Id() && requestA->Epoch() == requestB->Epoch(), "Request identities are not independent");
    ExpectRemoteError([&] { (void)a->Submit({1}, RequestOptions(2)); }, wire::ErrorCode::Busy);
    auto readerA = std::async(std::launch::async, [requestA] { return Drain(requestA); });
    auto readerB = std::async(std::launch::async, [requestB] { return Drain(requestB); });
    WaitUntil([&] { return run.client->Query(1s).activeRequests == 2; }, "Two sessions never became active concurrently");
    ExpectCompleted(readerA.get(), ExpectedTokens(promptA, 96));
    ExpectCompleted(readerB.get(), ExpectedTokens(promptB, 80));
    Expect(a->Query().contextLength == std::optional<uint32_t>{99} &&
               b->Query().contextLength == std::optional<uint32_t>{83}, "Independent session cache lengths were mixed");
    requestA->Release();
    requestB->Release();
    run.WaitForNoRequests();
    const std::vector<int32_t> appended{4, 5};
    const auto next = a->Submit(appended, RequestOptions(4));
    ExpectCompleted(Drain(next), ExpectedTokens(appended, 4, 99));
    Expect(a->Query().contextLength == std::optional<uint32_t>{105}, "Final generated token was not committed before continuation");
    next->Release();
    run.WaitForNoRequests();
    run.client->Shutdown();
}

void TestConcurrentSessions() { ConcurrentSessions(false); }
void TestHybridConcurrentSessions() { ConcurrentSessions(true); }

void TestCancelReleaseAndParentLifetime() {
    Run run;
    const auto a = run.model->CreateSession();
    const auto b = run.model->CreateSession();
    const auto requestA = a->Submit({21, 22, 23}, RequestOptions(256));
    const std::vector<int32_t> promptB{81, 82, 83};
    const auto requestB = b->Submit(promptB, RequestOptions(12));
    const auto first = ReadEvent(requestA);
    Expect(!first.terminal && first.sequence == 1, "Cancellation test did not enter generation");
    Expect(requestA->Cancel(1s), "Active cancellation was not accepted");
    run.model->Release();
    a->Release();
    b->Release();
    ExpectUnavailable([&] { (void)run.model->CreateSession(); });
    ExpectUnavailable([&] { (void)a->Query(); });
    const auto cancelled = Drain(requestA, {first.tokenId});
    Expect(cancelled.terminal.summary.status == wire::RequestStatus::Cancelled &&
               cancelled.terminal.summary.endReason == wire::EndReason::Cancelled,
           "Explicit cancellation changed into failure or successful completion");
    ExpectCompleted(Drain(requestB), ExpectedTokens(promptB, 12));
    requestA->Release();
    requestA->Release();
    requestB->Release();
    requestB->Release();
    ExpectUnavailable([&] { (void)requestA->Read(0ms); });
    run.WaitForNoRequests();
    Expect(run.client->Query().accepting, "Request cancellation/release poisoned the Worker");
    run.client->Shutdown();
}

void TestStoppedConsumerDoesNotBlockPeer() {
    auto options = ClientOptions();
    options.limits.tokenWindow = 4;
    options.limits.maxBufferedTokensPerRequest = 8;
    options.limits.maxOutstandingRequests = 2;
    Run run(options);
    const auto a = run.model->CreateSession();
    const auto b = run.model->CreateSession();
    const auto requestA = a->Submit({31, 32, 33}, RequestOptions(200));
    const std::vector<int32_t> promptB{91, 92, 93};
    const auto requestB = b->Submit(promptB, RequestOptions(6));
    // Deliberately never call A.Read: only its bounded private pipe reader runs.
    ExpectCompleted(Drain(requestB), ExpectedTokens(promptB, 6));
    wire::RequestInfo blocked;
    WaitUntil([&] { blocked = requestA->Query(1s); return blocked.completionReady; },
              "Unread long request did not reach local output backpressure");
    Expect(blocked.summary.endReason == wire::EndReason::OutputBackpressure &&
               blocked.summary.status == wire::RequestStatus::Cancelled &&
               blocked.lastSentSequence <= run.limits.tokenWindow &&
               blocked.bufferedTokens <= run.limits.maxBufferedTokensPerRequest,
           "Unread request escaped its bounded token window/ring");
    Expect(requestA->Snapshot().deliveredTokens == 0 && !requestA->Snapshot().terminalConsumed,
           "Private event reader consumed user output before abandonment");
    requestB->Release();
    WaitUntil([&] { return run.client->Query(1s).ipcOutstandingRequests == 1; },
              "Completed unread A lost its transport reservation or B failed to drain");
    requestA->Release();
    run.WaitForNoRequests();
    const auto abandoned = requestA->Snapshot();
    Expect(abandoned.released && abandoned.deliveryAbandoned && !abandoned.localFailure &&
               abandoned.finalReceived && abandoned.terminalReceived && abandoned.completionReady &&
               abandoned.summary.endReason == wire::EndReason::OutputBackpressure &&
               abandoned.deliveredTokens == 0 && abandoned.discardedPublishedTokens == abandoned.summary.generatedTokens &&
               abandoned.acknowledgedSequence == abandoned.summary.eventCount && abandoned.bufferedTokens == 0,
           "Release did not privately drain and preserve A's original terminal summary");
    ExpectUnavailable([&] { (void)requestA->Read(0ms); });
    run.client->Shutdown();
}

void TestUnreadCompletionAndConcurrentShutdown() {
    Run run;
    const auto a = run.model->CreateSession();
    const auto b = run.model->CreateSession();
    const auto completed = a->Submit({41, 42, 43}, RequestOptions(4));
    WaitUntil([&] { return completed->Snapshot().completionReady; }, "Short unread request did not complete");
    const auto original = completed->Snapshot();
    Expect(original.summary.status == wire::RequestStatus::Completed && original.bufferedTokens == 4 &&
               !original.terminalConsumed && original.deliveredTokens == 0 && original.acknowledgedSequence == 0,
           "Unread completed stream was already consumed or acknowledged");
    const auto active = b->Submit({51, 52, 53}, RequestOptions(256));
    const auto first = ReadEvent(active);
    Expect(!first.terminal && first.sequence == 1 && !active->Snapshot().completionReady,
           "Shutdown fixture did not retain an active peer stream");
    std::promise<void> start;
    const auto gate = start.get_future().share();
    auto stopA = std::async(std::launch::async, [client = run.client, gate] { gate.wait(); client->Shutdown(5s); });
    auto stopB = std::async(std::launch::async, [client = run.client, gate] { gate.wait(); client->Shutdown(5s); });
    start.set_value();
    stopA.get();
    stopB.get();
    const auto closed = run.client->Snapshot();
    Expect(closed.status == wire::DomainStatus::Closed && closed.error.empty() &&
               closed.outstandingRequests == 0 && closed.retainedRequestHandles == 0 &&
               closed.pendingCommands == 0 && closed.ingressBytes == 0 && closed.transportBytes == 0,
           "Concurrent shutdown did not finish the shared close barrier and release reservations");
    const auto terminalA = ReadEvent(completed);
    Expect(terminalA.terminal && !terminalA.localFailure && terminalA.deliveryAbandoned &&
               terminalA.summary.status == wire::RequestStatus::Completed && terminalA.summary.endReason == wire::EndReason::Length &&
               terminalA.summary.generatedTokens == original.summary.generatedTokens && terminalA.summary.eventCount == original.summary.eventCount &&
               terminalA.deliveredTokens == 0 && terminalA.discardedPublishedTokens == 4,
           "Shutdown rewrote or silently delivered an already completed unread stream");
    const auto terminalB = ReadEvent(active);
    Expect(terminalB.terminal && !terminalB.localFailure && terminalB.deliveryAbandoned &&
               terminalB.summary.status == wire::RequestStatus::Cancelled &&
               (terminalB.summary.endReason == wire::EndReason::SchedulerStopped ||
                terminalB.summary.endReason == wire::EndReason::OutputBackpressure) &&
               terminalB.deliveredTokens == 1 &&
               terminalB.deliveredTokens + terminalB.discardedPublishedTokens == terminalB.summary.generatedTokens,
           "Shutdown lost the active stream's delivered/discarded token accounting");
    Expect(completed->Read(0ms).status == host::InferenceReadStatus::Closed &&
               active->Read(0ms).status == host::InferenceReadStatus::Closed, "Shutdown terminal was delivered twice");
    Expect(completed->Snapshot().acknowledgedSequence == terminalA.sequence &&
               active->Snapshot().acknowledgedSequence == terminalB.sequence,
           "Shutdown completed before private terminal ACKs");
}

void TestOldEpochDoesNotRebindHandles() {
    Run old;
    const auto oldSession = old.model->CreateSession();
    const auto oldRequest = oldSession->Submit({61, 62, 63}, RequestOptions(3));
    ExpectCompleted(Drain(oldRequest), ExpectedTokens({61, 62, 63}, 3));
    const std::string oldEpoch = oldSession->Epoch();
    old.client->Shutdown();
    Run fresh;
    const auto newSession = fresh.model->CreateSession();
    Expect(oldSession->Info().sessionId == newSession->Info().sessionId && old.model->Info().modelId == fresh.model->Info().modelId,
           "Restart fixture did not reuse numeric object IDs");
    Expect(oldEpoch != newSession->Epoch() && oldSession->Epoch() == oldEpoch && oldRequest->Epoch() == oldEpoch,
           "Worker restart reused an epoch or rebound an old handle");
    ExpectUnavailable([&] { (void)oldSession->Submit({1}, RequestOptions(2)); });
    ExpectUnavailable([&] { (void)old.model->CreateSession(); });
    ExpectUnavailable([&] { (void)oldRequest->Query(); });
    const auto request = newSession->Submit({71, 72, 73}, RequestOptions(5));
    ExpectCompleted(Drain(request), ExpectedTokens({71, 72, 73}, 5));
    Expect(fresh.client->Query().accepting, "Old handle use damaged the new epoch");
    request->Release();
    fresh.WaitForNoRequests();
    fresh.client->Shutdown();
}

void ExpectSingleLocalFailure(const std::shared_ptr<host::InferenceRequest>& request, uint64_t alreadyDelivered) {
    uint64_t delivered = alreadyDelivered;
    const auto deadline = Clock::now() + 4s;
    do {
        const auto event = ReadEvent(request);
        if (!event.terminal) { ++delivered; continue; }
        Expect(event.localFailure && event.deliveryAbandoned && event.sequence == 0 && event.summary.eventCount == 0 &&
                   event.summary.status == wire::RequestStatus::Failed && event.summary.errorCode == wire::ErrorCode::TransportError &&
                   event.deliveredTokens == delivered,
               "Worker crash was disguised as a wire completion or lost delivered-token accounting");
        for (unsigned repeat = 0; repeat < 4; ++repeat)
            Expect(request->Read(0ms).status == host::InferenceReadStatus::Closed, "Worker crash delivered duplicate local terminals");
        return;
    } while (Clock::now() < deadline);
    throw std::runtime_error("Worker crash did not fail its outstanding request");
}

void TestWorkerCrashFailsEachStreamOnce() {
    Run run;
    const auto a = run.model->CreateSession();
    const auto b = run.model->CreateSession();
    const auto peer = a->Submit({81, 82, 83}, RequestOptions(256));
    const auto first = ReadEvent(peer);
    Expect(!first.terminal, "Crash fixture needs an already accepted live peer");
    const auto epoch = peer->Epoch();
    std::shared_ptr<host::InferenceRequest> trigger;
    try { trigger = b->Submit({CrashPromptToken, 2, 3}, RequestOptions(32)); }
    catch (const host::InferenceClientError& error) {
        // Process termination can beat the accepted response's reader. The
        // existing peer still proves cleanup for a published request handle.
        Expect(error.Code() == host::ClientErrorCode::Closed, "Crash submit failed for an unrelated reason");
    }
    ExpectSingleLocalFailure(peer, 1);
    if (trigger) ExpectSingleLocalFailure(trigger, 0);
    const auto failed = run.client->Snapshot();
    Expect(failed.status == wire::DomainStatus::Faulted && failed.epoch == epoch && !failed.error.empty() &&
               failed.outstandingRequests == 0 && failed.retainedRequestHandles == 0 &&
               failed.pendingCommands == 0 && failed.ingressBytes == 0 && failed.transportBytes == 0,
           "Worker crash leaked host reservations or silently restarted the connection");
    ExpectUnavailable([&] { (void)run.client->Query(); });
    ExpectUnavailable([&] { (void)a->Submit({1}, RequestOptions(2)); });
    ExpectUnavailable([&] { run.client->Shutdown(5s); });
}

std::shared_ptr<host::InferenceSession> ReuseSessionSlot(const std::shared_ptr<host::InferenceModel>& model) {
    std::shared_ptr<host::InferenceSession> result;
    WaitUntil([&] {
        try { result = model->CreateSession(1s); return true; }
        catch (const host::InferenceClientError& error) {
            if (error.RemoteCode() != wire::ErrorCode::ResourceLimit) throw;
            return false;
        }
    }, "Released session slots were not reclaimed by the lifecycle executor");
    return result;
}

void TestRepeatedLifecycleAndHandleStability() {
    // Earlier cases warm ordinary C++/Win32 lazy initialization before this
    // baseline. A small fixed allowance covers thread teardown observations;
    // it cannot hide a per-connection pipe/process handle leak over five runs.
    const DWORD baseline = ProcessHandleCount();
    for (unsigned cycle = 0; cycle < 5; ++cycle) {
        {
            Run run(ClientOptions(), cycle % 2 != 0);
            for (unsigned iteration = 0; iteration < 3; ++iteration) {
                const auto a = ReuseSessionSlot(run.model);
                const auto b = ReuseSessionSlot(run.model);
                const auto request = a->Submit({91, 92, 93}, RequestOptions(4));
                ExpectCompleted(Drain(request), ExpectedTokens({91, 92, 93}, 4));
                request->Release();
                a->Release();
                b->Release();
                run.WaitForNoRequests();
            }
            const auto previousId = run.model->Info().modelId;
            run.model->Release();
            std::shared_ptr<host::InferenceModel> replacement;
            WaitUntil([&] {
                try { replacement = run.Load(); return true; }
                catch (const host::InferenceClientError& error) {
                    if (error.RemoteCode() != wire::ErrorCode::ResourceLimit) throw;
                    return false;
                }
            }, "Released model and session resources prevented model slot reuse");
            Expect(replacement->Info().modelId > previousId, "Same-epoch model handle IDs were reused");
            const auto session = replacement->CreateSession();
            const auto request = session->Submit({101, 102, 103}, RequestOptions(2));
            ExpectCompleted(Drain(request), ExpectedTokens({101, 102, 103}, 2));
            request->Release();
            session->Release();
            replacement->Release();
            run.WaitForNoRequests();
            run.client->Shutdown();
        }
        WaitUntil([&] { return ProcessHandleCount() <= baseline + 2; },
                  "Repeated inference process lifetimes leaked host pipe/process handles", 2s);
    }
}

} // namespace

int main() {
    const auto started = Clock::now();
    const std::pair<const char*, void (*)()> tests[]{
        {"concurrent real-pipe sessions and cache continuation", TestConcurrentSessions},
        {"Hybrid real-pipe sessions and cache continuation", TestHybridConcurrentSessions},
        {"cancel, release and parent lifetime", TestCancelReleaseAndParentLifetime},
        {"stopped consumer and private release drain", TestStoppedConsumerDoesNotBlockPeer},
        {"unread completion and concurrent shutdown", TestUnreadCompletionAndConcurrentShutdown},
        {"old epoch handles cannot rebind", TestOldEpochDoesNotRebindHandles},
        {"Worker crash produces one local terminal", TestWorkerCrashFailsEachStreamOnce},
        {"repeated lifecycle and OS handle stability", TestRepeatedLifecycleAndHandleStability},
    };
    try {
        HostIsolation isolation;
        isolation.Check();
        for (const auto& test : tests) {
            test.second();
            isolation.Check();
            std::cout << "PASS " << test.first << '\n';
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started);
        Expect(elapsed < 60s, "CPU pipe suite exceeded its 60-second time budget");
        std::cout << "PASS all real inference pipe tests (" << elapsed.count() << " ms)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL inference pipe test: " << error.what() << '\n';
        return 1;
    }
}
