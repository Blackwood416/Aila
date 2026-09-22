#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "runtime/InferenceClient.hpp"
#include "profile/Profiling.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_IPC_WORKER_PATH
#error "Shared inference IPC tests require the actual Worker target path"
#endif
#ifndef AILA_BUILD_ID
#error "Shared inference IPC tests require the Worker build identity"
#endif

namespace {

namespace core = aila::inference;
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int32_t>;
constexpr int DefaultCapacity = 512;
enum class Backend { Qwen3, Qwen35, Qwen35Bf16 };
constexpr std::array<const char*, 5> DefaultCaseNames{
    "concurrency", "cancel-backpressure", "session-recreate", "parent-shutdown", "epoch"};
constexpr std::array<const char*, 4> ExtendedCaseNames{"long-context", "soak", "thinking-budget", "session-context"};

struct FixtureOptions {
    uint32_t context = DefaultCapacity;
    size_t soakRounds = 20;
};

const char* BackendName(Backend backend) {
    if (backend == Backend::Qwen35Bf16) return "qwen35_bf16";
    return backend == Backend::Qwen35 ? "qwen35_nf4" : "qwen3_nf4";
}

const char* ModelEnvironment(Backend backend) {
    if (backend == Backend::Qwen35Bf16) return "AILA_QWEN35_BF16_TEST_MODEL";
    return backend == Backend::Qwen35 ? "AILA_QWEN35_TEST_MODEL" : "AILA_QWEN3_TEST_MODEL";
}

bool IsHybrid(Backend backend) { return backend != Backend::Qwen3; }

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

size_t CountOption(const std::string& value, size_t minimum, size_t maximum, const std::string& option) {
    size_t count = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    Expect(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
               count >= minimum && count <= maximum,
           option + " must be an integer in " + std::to_string(minimum) + ".." + std::to_string(maximum));
    return count;
}

struct Case {
    Tokens prompt;
    Tokens continuation;
    host::InferenceRequestOptions options;
    host::InferenceRequestOptions continuationOptions;
    Tokens expected;
    Tokens expectedContinuation;
};

struct ContextCase {
    uint32_t target = 0, restored = 0, finalLength = 0;
    Tokens replay, expected;
    host::InferenceRequestOptions options;
};

struct ReferenceSet {
    std::vector<Case> cases;
    std::vector<Case> longCases;
    std::vector<Case> soakCases;
    std::vector<Case> thinkingCases;
    std::vector<ContextCase> contextCases;
    Tokens longCancellationExpected;
    uint32_t capacity = 0;
    uint32_t vocabSize = 0;
    uint64_t sessionCacheBytes = 0;
    uint64_t sessionHostBytes = 0;
};

core::RequestOptions DirectOptions(const host::InferenceRequestOptions& input) {
    core::RequestOptions result;
    result.prefillChunkSize = static_cast<int>(input.prefillChunkSize);
    result.stopTokenIds.assign(input.stopTokenIds.begin(), input.stopTokenIds.end());
    result.thinkingBudget = input.thinkingBudget;
    auto& to = result.generation;
    const auto& from = input.generation;
    to.max_new_tokens = static_cast<int>(from.maxNewTokens);
    to.do_sample = from.doSample;
    to.temperature = static_cast<float>(from.temperature);
    to.top_k = static_cast<int>(from.topK);
    to.top_p = static_cast<float>(from.topP);
    to.use_fixed_seed = from.useFixedSeed;
    to.sampling_seed = from.samplingSeed;
    to.repetition_penalty = static_cast<float>(from.repetitionPenalty);
    to.presence_penalty = static_cast<float>(from.presencePenalty);
    to.frequency_penalty = static_cast<float>(from.frequencyPenalty);
    to.thinking_budget_tokens = -1;
    return result;
}

Tokens Direct(const std::shared_ptr<core::SessionState>& session, const Tokens& prompt,
              const host::InferenceRequestOptions& options) {
    auto request = core::RequestState::Create(session, std::vector<int>(prompt.begin(), prompt.end()), DirectOptions(options));
    while (!core::IsTerminal(request->Step().status)) {}
    const auto result = request->Snapshot();
    Expect(result.status == core::RequestStatus::Completed, "direct reference failed: " + result.error);
    return Tokens(result.generatedIds.begin(), result.generatedIds.end());
}

void Same(const Tokens& actual, const Tokens& expected, const std::string& label);

core::MemorySnapshot SampleMemory(const std::shared_ptr<core::DeviceRuntime>& runtime,
                                  const char* phase, const std::string& stage,
                                  size_t round = 0, size_t activeSessions = 0) {
    const auto memory = runtime->Memory();
    std::cout << "[MEMORY] scope=direct_context_tracked_usm phase=" << phase << " stage=" << stage
              << " round=" << round << " active_sessions=" << activeSessions
              << " device_bytes=" << memory.currentBytes << " lifetime_peak_device_bytes=" << memory.peakBytes
              << " pinned_host_bytes=" << memory.pinnedHostBytes << " lifetime_peak_pinned_host_bytes=" << memory.peakPinnedHostBytes
              << " device_allocations=" << memory.deviceAllocations << " pinned_host_allocations=" << memory.pinnedHostAllocations << '\n';
    return memory;
}

Tokens FixedLengthTokens(const Tokenizer& tokenizer, const std::string& text, size_t count) {
    const auto source = tokenizer.encode(text);
    Expect(!source.empty(), "Exact-length token fixture could not encode its source text");
    Tokens tokens;
    tokens.reserve(count);
    for (size_t index = 0; index < count; ++index) tokens.push_back(source[index % source.size()]);
    return tokens;
}

std::vector<Case> ExtendedCases(const Tokenizer& tokenizer, uint32_t capacity, bool longContext) {
    const std::array<std::string, 4> texts{
        "Aster records the temperature each morning and compares consecutive measurements. ",
        "Beryl keeps a journal of train arrivals, distances, and elapsed minutes. ",
        "Coral describes a C++ vector algorithm and several distinct boundary conditions. ",
        "Dahlia lists observations about books, gardens, colours, and simple arithmetic. "};
    const std::array<size_t, 4> promptLengths = longContext
        ? std::array<size_t, 4>{capacity / 4 + 13, capacity / 2 + 29, capacity * 3 / 4 + 7, capacity - 192}
        : std::array<size_t, 4>{33, 65, 97, 161};
    const std::array<uint32_t, 4> chunks = longContext
        ? std::array<uint32_t, 4>{64, 128, 256, 512} : std::array<uint32_t, 4>{8, 16, 32, 64};
    const std::array<uint32_t, 4> continuationChunks{1, 7, 16, 32};
    std::vector<Case> cases;
    for (size_t index = 0; index < texts.size(); ++index) {
        Case test;
        // These exact-length pretokenized inputs exercise cache boundaries;
        // they are not a chat answer-quality benchmark.
        test.prompt = FixedLengthTokens(tokenizer, texts[index], promptLengths[index]);
        test.options.prefillChunkSize = chunks[index];
        test.options.generation.maxNewTokens = static_cast<uint32_t>((longContext ? 20 : 4) + index * (longContext ? 3 : 1));
        test.options.generation.doSample = index >= 2;
        test.options.generation.useFixedSeed = true;
        test.options.generation.samplingSeed = 123456780ull + index;
        test.options.generation.repetitionPenalty = index >= 2 ? 1.08 : 1.0;
        test.options.generation.presencePenalty = index >= 2 ? 0.1 : 0.0;
        test.options.generation.frequencyPenalty = index >= 2 ? 0.05 : 0.0;
        test.continuationOptions = test.options;
        test.continuationOptions.prefillChunkSize = continuationChunks[index];
        test.continuationOptions.generation.maxNewTokens = static_cast<uint32_t>((longContext ? 9 : 2) + index);
        const size_t continuationLength = longContext
            ? (index == 3 ? capacity - test.prompt.size() - test.options.generation.maxNewTokens -
                              test.continuationOptions.generation.maxNewTokens : 33 + index * 32)
            : 9 + index * 7;
        test.continuation = FixedLengthTokens(tokenizer, " Continue with a different observation. " + texts[index], continuationLength);
        Expect(test.prompt.size() > test.options.prefillChunkSize &&
                   test.prompt.size() + test.options.generation.maxNewTokens + test.continuation.size() +
                       test.continuationOptions.generation.maxNewTokens <= capacity,
               "Extended token fixture does not fit or does not exercise incremental prefill");
        cases.push_back(std::move(test));
    }
    return cases;
}

void CaptureReferences(const std::shared_ptr<core::DeviceRuntime>& runtime,
                       const std::shared_ptr<core::LoadedModel>& model,
                       std::vector<Case>& cases, const std::string& group) {
    for (size_t index = 0; index < cases.size(); ++index) {
        auto& test = cases[index];
        auto session = model->CreateSession();
        test.expected = Direct(session, test.prompt, test.options);
        test.expectedContinuation = Direct(session, test.continuation, test.continuationOptions);
        Expect(test.expected.size() == test.options.generation.maxNewTokens &&
                   test.expectedContinuation.size() == test.continuationOptions.generation.maxNewTokens &&
                   session->ContextLength() == test.prompt.size() + test.expected.size() +
                       test.continuation.size() + test.expectedContinuation.size(),
               "Direct reference did not consume its full independent history");
        std::cout << "[REFERENCE] group=" << group << " request=" << index << " prompt=" << test.prompt.size()
                  << " prefill_chunk=" << test.options.prefillChunkSize
                  << " continuation_prompt=" << test.continuation.size()
                  << " continuation_chunk=" << test.continuationOptions.prefillChunkSize
                  << " generated=" << test.expected.size() << " continuation=" << test.expectedContinuation.size() << '\n';
        const auto live = SampleMemory(runtime, "warmup", group + "_session_live", index, 1);
        const auto cacheBytes = session->CacheBytes();
        session.reset();
        const auto released = SampleMemory(runtime, "warmup", group + "_session_released", index);
        Expect(model->ActiveSessions() == 0 && live.currentBytes >= cacheBytes &&
                   released.currentBytes == live.currentBytes - cacheBytes,
               "Direct reference retained a released Session or its Context-tracked cache");
    }
}

void DirectSoak(const std::shared_ptr<core::DeviceRuntime>& runtime,
                const std::shared_ptr<core::LoadedModel>& model, const std::vector<Case>& cases, size_t rounds) {
    const auto wave = [&](const char* phase, size_t round) {
        Expect(model->ActiveSessions() == 0, "Direct soak inherited a live Session");
        const auto before = runtime->Memory();
        std::vector<std::shared_ptr<core::SessionState>> sessions;
        for (size_t index = 0; index < 4; ++index) sessions.push_back(model->CreateSession());
        const auto allocated = SampleMemory(runtime, phase, "soak_four_sessions", round, sessions.size());
        Expect(model->ActiveSessions() == 4 &&
                   allocated.currentBytes == before.currentBytes + 4 * model->SessionReservationBytes(),
               "Direct soak duplicated model resources or lost a Session reservation");
        bool refused = false;
        try { model->CreateSession(); }
        catch (const core::Error& error) {
            if (error.Code() != core::ErrorCode::ResourceLimit) throw;
            refused = true;
        }
        const auto rejected = runtime->Memory();
        Expect(refused && rejected.currentBytes == allocated.currentBytes && rejected.peakBytes == allocated.peakBytes,
               "Fifth direct Session was accepted or allocated tracked payload before rejection");
        for (size_t index = 0; index < sessions.size(); ++index) {
            const auto& test = cases[(index + round) % cases.size()];
            Same(Direct(sessions[index], test.prompt, test.options), test.expected, "direct soak prompt");
            Same(Direct(sessions[index], test.continuation, test.continuationOptions), test.expectedContinuation,
                 "direct soak continuation");
        }
        const auto live = SampleMemory(runtime, phase, "soak_completed_live", round, sessions.size());
        sessions.clear();
        const auto released = SampleMemory(runtime, phase, "soak_released", round);
        Expect(model->ActiveSessions() == 0 && live.currentBytes >= 4 * model->SessionReservationBytes() &&
                   released.currentBytes == live.currentBytes - 4 * model->SessionReservationBytes(),
               "Direct soak did not reclaim all four Session payloads");
        return released;
    };
    // Warm all request shapes with four live Sessions before fixing the
    // steady-state current-byte baseline. The cumulative peak is descriptive.
    const auto warm = wave("warmup", 0);
    for (size_t round = 0; round < rounds; ++round) {
        const auto released = wave("steady", round + 1);
        Expect(released.currentBytes == warm.currentBytes && released.pinnedHostBytes == warm.pinnedHostBytes &&
                   released.deviceAllocations == warm.deviceAllocations && released.pinnedHostAllocations == warm.pinnedHostAllocations,
               "Direct soak grew Context-tracked device/pinned allocations after warmup");
    }
    std::cout << "[PASS] direct soak rounds=" << rounds << " four_session_reclamation=true tracked_usm_stable=true\n";
}

ReferenceSet References(const std::string& directory, Backend backend, const FixtureOptions& fixture,
                        bool longContext, bool soak, bool thinking, bool contextOperations) {
    Tokenizer tokenizer;
    Expect(tokenizer.load(directory), "Could not load tokenizer");
    const std::array<std::string, 4> prompts{
        "List the first few prime numbers and explain why they are prime. /no_think",
        "用中文简短说明共享模型权重与独立会话缓存的关系。 /no_think",
        "A train leaves at noon and travels 60 kilometres in an hour. Explain its speed, then give a short example. /no_think",
        "Write a small C++ function that sums a vector of integers. Explain the time complexity and one possible overflow issue. /no_think"};
    ReferenceSet reference;
    auto runtime = core::DeviceRuntime::Create();
    std::cout << "[MEMORY_SCOPE] Context-tracked device/pinned USM only, including explicit sampling/shared scratch; "
                 "excludes library/driver internals, pageable host mirrors, allocator overhead and total RSS/VRAM. "
                 "Peaks are cumulative for this direct runtime.\n";
    const auto emptyRuntime = SampleMemory(runtime, "baseline", "runtime_created");
    core::ModelOptions options;
    options.maxSequenceLength = static_cast<int>(fixture.context);
    options.maxSessions = 4;
    // Match the remote zero-snapshot Hybrid policy. A nontrivial truncate may
    // restore zero; the caller must use its returned length to replay history.
    options.maxRecurrentSnapshots = 0;
    const auto loader = backend == Backend::Qwen3 ? core::LoadedModel::LoadQwen3Nf4
        : backend == Backend::Qwen35 ? core::LoadedModel::LoadQwen35Nf4 : core::LoadedModel::LoadQwen35Bf16;
    auto model = loader(runtime, directory, options);
    reference.capacity = static_cast<uint32_t>(model->MaxSequenceLength());
    Expect(reference.capacity == fixture.context, "Model does not support the requested fixture context");
    SampleMemory(runtime, "baseline", "model_loaded");
    reference.vocabSize = static_cast<uint32_t>(model->VocabSize());
    reference.sessionCacheBytes = model->SessionReservationBytes();
    reference.sessionHostBytes = model->SessionHostReservationBytes();
    Expect(tokenizer.vocab_size() > 0 && static_cast<uint32_t>(tokenizer.vocab_size()) <= reference.vocabSize,
           "Tokenizer vocabulary exceeds the model's padded vocabulary");
    std::cout << "[MODEL] backend=" << BackendName(backend) << " context=" << reference.capacity << " vocab=" << reference.vocabSize
              << " device_session_payload_bytes=" << reference.sessionCacheBytes
              << " host_session_payload_bytes=" << reference.sessionHostBytes << '\n';
    constexpr std::array<uint32_t, 4> hybridChunks{1, 8, 16, 32};
    for (size_t index = 0; index < prompts.size(); ++index) {
        Case test;
        const auto prompt = tokenizer.apply_chat_template("You are a helpful assistant.", prompts[index]);
        test.prompt.assign(prompt.begin(), prompt.end());
        const auto continuation = tokenizer.encode(" Continue with one more useful example.");
        test.continuation.assign(continuation.begin(), continuation.end());
        test.options.prefillChunkSize = IsHybrid(backend) ? hybridChunks[index] : index % 2 ? 32 : 16;
        test.options.generation.maxNewTokens = static_cast<uint32_t>(20 + index * 3);
        test.options.generation.doSample = index >= 2;
        test.options.generation.useFixedSeed = true;
        test.options.generation.samplingSeed = 987654321ull + index;
        test.options.generation.repetitionPenalty = index >= 2 ? 1.08 : 1.0;
        test.options.generation.presencePenalty = index >= 2 ? 0.1 : 0.0;
        test.options.generation.frequencyPenalty = index >= 2 ? 0.05 : 0.0;
        test.continuationOptions = test.options;
        test.continuationOptions.generation.maxNewTokens = 5;
        if (IsHybrid(backend)) test.continuationOptions.prefillChunkSize = index % 2 ? 3 : 1;
        Expect(test.prompt.size() + 256 <= reference.capacity &&
                   test.prompt.size() + test.options.generation.maxNewTokens + test.continuation.size() +
                       test.continuationOptions.generation.maxNewTokens <= reference.capacity,
               "Fixture prompt exceeds the fixed Session capacity");
        reference.cases.push_back(std::move(test));
    }
    CaptureReferences(runtime, model, reference.cases, "default");
    std::cout << "[PASS] direct greedy, seeded/penalized and continuation references\n";
    if (contextOperations) {
        for (size_t index = 0; index < reference.cases.size(); ++index) {
            const auto& test = reference.cases[index];
            auto session = model->CreateSession();
            Same(Direct(session, test.prompt, test.options), test.expected, "direct context initial");
            ContextCase context;
            context.target = static_cast<uint32_t>(index == 0 ? 0 : index == 1 ? test.prompt.size() / 2 :
                index == 2 ? test.prompt.size() - 1 : session->ContextLength());
            context.restored = static_cast<uint32_t>(session->Truncate(static_cast<int>(context.target)));
            Expect(context.restored <= context.target && session->ContextLength() == context.restored,
                   "Direct context reference restored an invalid checkpoint");
            if (context.restored < test.prompt.size()) {
                context.replay.assign(test.prompt.begin() + context.restored, test.prompt.end());
                context.options = test.options;
            } else {
                context.replay = test.continuation;
                context.options = test.continuationOptions;
            }
            context.expected = Direct(session, context.replay, context.options);
            context.finalLength = static_cast<uint32_t>(session->ContextLength());
            session->Reset();
            Same(Direct(session, test.prompt, test.options), test.expected, "direct reset initial parity");
            Same(Direct(session, test.continuation, test.continuationOptions), test.expectedContinuation,
                 "direct reset continuation parity");
            std::cout << "[REFERENCE] group=session-context request=" << index << " target=" << context.target
                      << " restored=" << context.restored << " replay=" << context.replay.size()
                      << " generated=" << context.expected.size() << " final_length=" << context.finalLength << '\n';
            reference.contextCases.push_back(std::move(context));
        }
        Expect(model->ActiveSessions() == 0, "Direct context references retained a Session");
        std::cout << "[PASS] direct reset/truncate/replay and seeded continuation references\n";
    }
    if (longContext) {
        reference.longCases = ExtendedCases(tokenizer, reference.capacity, true);
        CaptureReferences(runtime, model, reference.longCases, "long-context");
        auto session = model->CreateSession();
        auto cancellationOptions = reference.longCases.front().options;
        cancellationOptions.generation.maxNewTokens = 256;
        reference.longCancellationExpected = Direct(session, reference.longCases.front().prompt, cancellationOptions);
        Expect(reference.longCancellationExpected.size() == 256, "Long cancellation reference was incomplete");
        session.reset();
        SampleMemory(runtime, "warmup", "long_context_cancellation_reference_released");
    }
    if (soak) {
        reference.soakCases = ExtendedCases(tokenizer, reference.capacity, false);
        CaptureReferences(runtime, model, reference.soakCases, "soak");
        DirectSoak(runtime, model, reference.soakCases, fixture.soakRounds);
    }
    if (thinking) {
        wire::ThinkingBudgetOptions budget;
        budget.openTokenId = tokenizer.special_token_id("<think>");
        budget.closeTokenId = tokenizer.special_token_id("</think>");
        budget.promptOpensThink = true;
        budget.closeTokenIds = {budget.closeTokenId};
        const auto suffix = tokenizer.encode("\n\n");
        budget.closeTokenIds.insert(budget.closeTokenIds.end(), suffix.begin(), suffix.end());
        Expect(!aila::generation::ValidateThinkingBudget(budget, reference.vocabSize, {}),
               "Thinking fixture tokenizer has no valid closing sequence");
        Expect(budget.closeTokenIds.size() > 1, "Thinking boundary fixture requires a closing suffix");
        const auto prompt = tokenizer.encode(
            "<|im_start|>user\nWhat is 7 times 9? Explain the arithmetic.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n");
        const std::array<uint32_t, 4> budgets{0, 2, 5, 0};
        for (size_t index = 0; index < budgets.size(); ++index) {
            Case test;
            test.prompt.assign(prompt.begin(), prompt.end());
            test.options.prefillChunkSize = 7;
            test.options.generation.maxNewTokens = index == 3
                ? static_cast<uint32_t>(budget.closeTokenIds.size() - 1) : 16;
            test.options.generation.doSample = index == 1 || index == 2;
            test.options.generation.useFixedSeed = true;
            test.options.generation.samplingSeed = 2026090900 + index;
            test.options.generation.repetitionPenalty = 1.08;
            test.options.thinkingBudget = budget;
            test.options.thinkingBudget->maxTokens = budgets[index];
            const auto continuation = tokenizer.encode(" Continue with another example.");
            test.continuation.assign(continuation.begin(), continuation.end());
            test.continuationOptions = test.options;
            test.continuationOptions.thinkingBudget.reset();
            test.continuationOptions.generation.maxNewTokens = 5;
            auto session = model->CreateSession();
            auto request = core::RequestState::Create(session, std::vector<int>(test.prompt.begin(), test.prompt.end()),
                                                      DirectOptions(test.options));
            while (!core::IsTerminal(request->Step().status)) {}
            const auto result = request->Snapshot();
            Expect(result.status == core::RequestStatus::Completed && result.endReason == core::RequestEndReason::Length,
                   "Direct thinking reference did not complete with Length");
            test.expected.assign(result.generatedIds.begin(), result.generatedIds.end());
            if (index == 0) Expect(test.expected.size() >= budget.closeTokenIds.size() &&
                std::equal(budget.closeTokenIds.begin(), budget.closeTokenIds.end(), test.expected.begin()),
                "Zero budget did not inject the complete tokenizer closing sequence");
            if (index == 3) Expect(test.expected.empty(), "Short output emitted an incomplete thinking close");
            Expect(session->ContextLength() == test.prompt.size() + test.expected.size(),
                   "Thinking reference did not retain exactly the emitted tokens in KV");
            test.expectedContinuation = Direct(session, test.continuation, test.continuationOptions);
            std::cout << "[REFERENCE] group=thinking-budget request=" << index << " budget=" << budgets[index]
                      << " generated=" << test.expected.size() << " continuation=" << test.expectedContinuation.size() << '\n';
            reference.thinkingCases.push_back(std::move(test));
        }
    }
    Expect(model->ActiveSessions() == 0, "Direct fixture retained a Session before model release");
    SampleMemory(runtime, "steady", "all_direct_sessions_released");
    model.reset();
    const auto released = SampleMemory(runtime, "teardown", "model_released");
    Expect(released.currentBytes == emptyRuntime.currentBytes && released.pinnedHostBytes == emptyRuntime.pinnedHostBytes &&
               released.deviceAllocations == emptyRuntime.deviceAllocations &&
               released.pinnedHostAllocations == emptyRuntime.pinnedHostAllocations,
           "Direct model teardown retained Context-tracked device/pinned payload or allocation records");
    return reference; // All direct device owners are destroyed before Worker starts.
}

struct Output {
    Tokens tokens;
    host::InferenceReadEvent terminal;
    double firstTokenMs = 0;
};

std::vector<Output> Drain(const std::vector<std::shared_ptr<host::InferenceRequest>>& requests,
                          Clock::time_point start = Clock::now(), std::chrono::milliseconds timeout = 120s) {
    std::vector<Output> output(requests.size());
    std::vector<bool> finished(requests.size());
    size_t remaining = requests.size();
    const auto deadline = Clock::now() + timeout;
    while (remaining && Clock::now() < deadline) {
        for (size_t index = 0; index < requests.size(); ++index) {
            if (finished[index]) continue;
            const auto read = requests[index]->Read(10ms);
            if (read.status == host::InferenceReadStatus::Timeout) continue;
            Expect(read.status == host::InferenceReadStatus::Event && read.event.has_value(), "request closed without a terminal");
            const auto& event = *read.event;
            if (event.terminal) {
                Expect(!event.localFailure, "transport failed: " + event.summary.message);
                output[index].terminal = event;
                finished[index] = true;
                --remaining;
                Expect(requests[index]->Read(0ms).status == host::InferenceReadStatus::Closed, "second terminal was exposed");
            } else {
                if (output[index].tokens.empty()) output[index].firstTokenMs =
                    std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                output[index].tokens.push_back(event.tokenId);
            }
        }
    }
    Expect(remaining == 0, "real IPC inference timed out");
    return output;
}

void Same(const Tokens& actual, const Tokens& expected, const std::string& label) {
    if (actual == expected) return;
    size_t offset = 0;
    while (offset < actual.size() && offset < expected.size() && actual[offset] == expected[offset]) ++offset;
    throw std::runtime_error(label + " diverged at token " + std::to_string(offset) +
        " actual=" + (offset < actual.size() ? std::to_string(actual[offset]) : "end") +
        " expected=" + (offset < expected.size() ? std::to_string(expected[offset]) : "end"));
}

void WaitReclaimed(const std::shared_ptr<host::InferenceClient>& client) {
    const auto deadline = Clock::now() + 10s;
    do {
        const auto local = client->Snapshot();
        const auto remote = client->Query();
        if (local.outstandingRequests == 0 && remote.ipcOutstandingRequests == 0) return;
        std::this_thread::sleep_for(2ms);
    } while (Clock::now() < deadline);
    throw std::runtime_error("IPC output reservations were not reclaimed");
}

std::shared_ptr<host::InferenceSession> CreateEventually(const std::shared_ptr<host::InferenceModel>& model) {
    const auto deadline = Clock::now() + 10s;
    for (;;) {
        try {
            auto session = model->CreateSession();
            const auto state = session->Query();
            Expect(!state.busy && state.contextLength == 0, "New Session retained previous context");
            Expect(state.modelId == model->Info().modelId && state.capacity == model->Info().maxSequenceLength &&
                       state.cacheBytes == model->Info().sessionReservationBytes,
                   "New Session metadata differs from its model reservation");
            return session;
        }
        catch (const host::InferenceClientError& error) {
            if (error.RemoteCode() != wire::ErrorCode::ResourceLimit || Clock::now() >= deadline) throw;
            std::this_thread::sleep_for(2ms);
        }
    }
}

void TestConcurrency(const std::shared_ptr<host::InferenceClient>& client,
                     const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases) {
    bool observedOverlap = false;
    for (const size_t count : {size_t{1}, size_t{2}, size_t{4}}) {
        std::vector<std::shared_ptr<host::InferenceSession>> sessions;
        std::vector<std::shared_ptr<host::InferenceRequest>> requests;
        for (size_t index = 0; index < count; ++index) sessions.push_back(CreateEventually(model));
        const auto start = Clock::now();
        for (size_t index = 0; index < count; ++index) requests.push_back(sessions[index]->Submit(cases[index].prompt, cases[index].options));
        const auto state = client->Query();
        observedOverlap |= state.activeRequests + state.queuedRequests >= 2;
        const auto output = Drain(requests, start);
        for (size_t index = 0; index < count; ++index) {
            Same(output[index].tokens, cases[index].expected, "IPC " + std::to_string(count) + " sessions / " + std::to_string(index));
            Expect(output[index].terminal.summary.endReason == wire::EndReason::Length, "unexpected end reason");
            const auto query = sessions[index]->Query();
            Expect(!query.busy && query.contextLength == cases[index].prompt.size() + output[index].tokens.size(),
                   "completed Session did not retain final token");
            std::cout << "[MEASURE] sessions=" << count << " request=" << index << " prompt=" << cases[index].prompt.size()
                      << " prefill_chunk=" << cases[index].options.prefillChunkSize
                      << " generated=" << output[index].tokens.size() << " caller_first_token_ms=" << output[index].firstTokenMs << '\n';
        }
        requests.clear();
        for (size_t index = 0; index < count; ++index)
            requests.push_back(sessions[index]->Submit(cases[index].continuation, cases[index].continuationOptions));
        const auto continued = Drain(requests);
        for (size_t index = 0; index < count; ++index) {
            Same(continued[index].tokens, cases[index].expectedContinuation, "IPC continuation");
            const auto state = sessions[index]->Query();
            Expect(!state.busy && state.contextLength == cases[index].prompt.size() + cases[index].expected.size() +
                       cases[index].continuation.size() + continued[index].tokens.size(),
                   "Continuation context did not retain the complete independent history");
        }
        requests.clear(); sessions.clear();
        WaitReclaimed(client);
        std::cout << "[PASS] " << count << " IPC sessions and continuation\n";
    }
    Expect(observedOverlap, "real Worker did not show overlapping accepted execution");
}

void TestThinkingBudget(const std::shared_ptr<host::InferenceClient>& client,
                        const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases) {
    Expect(cases.size() == 4, "Missing independent thinking references");
    std::vector<std::shared_ptr<host::InferenceSession>> sessions;
    std::vector<std::shared_ptr<host::InferenceRequest>> requests;
    for (const auto& test : cases) {
        sessions.push_back(CreateEventually(model));
        requests.push_back(sessions.back()->Submit(test.prompt, test.options));
    }
    const auto output = Drain(requests);
    for (size_t index = 0; index < cases.size(); ++index) {
        const auto& test = cases[index];
        Same(output[index].tokens, test.expected, "concurrent IPC thinking budget " + std::to_string(index));
        const auto& summary = output[index].terminal.summary;
        Expect(summary.status == wire::RequestStatus::Completed && summary.endReason == wire::EndReason::Length &&
                   summary.generatedTokens == test.expected.size() && summary.promptTokensProcessed == test.prompt.size(),
               "Thinking output counts or early Length reason changed over IPC");
        const auto query = sessions[index]->Query();
        Expect(!query.busy && query.contextLength == test.prompt.size() + test.expected.size(),
               "Thinking request retained the wrong cache length");
    }
    requests.clear();
    for (size_t index = 0; index < cases.size(); ++index)
        requests.push_back(sessions[index]->Submit(cases[index].continuation, cases[index].continuationOptions));
    const auto continued = Drain(requests);
    for (size_t index = 0; index < cases.size(); ++index) {
        Same(continued[index].tokens, cases[index].expectedContinuation, "IPC thinking continuation");
        const auto query = sessions[index]->Query();
        Expect(query.contextLength == cases[index].prompt.size() + cases[index].expected.size() +
                   cases[index].continuation.size() + cases[index].expectedContinuation.size(),
               "Budgeted continuation lost history or injected unreported tokens");
    }
    requests.clear();
    sessions.clear();
    WaitReclaimed(client);
    std::cout << "[PASS] four independent thinking budgets, forced close, early Length, seeded parity and continuation\n";
}

void TestCancelAndBackpressure(const std::shared_ptr<host::InferenceClient>& client,
                               const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases) {
    {
        auto a = CreateEventually(model), b = CreateEventually(model);
        auto longOptions = cases[0].options;
        longOptions.generation.maxNewTokens = 256;
        auto requestA = a->Submit(cases[0].prompt, longOptions);
        auto requestB = b->Submit(cases[1].prompt, cases[1].options);
        const auto first = requestA->Read(30s);
        Expect(first.status == host::InferenceReadStatus::Event && !first.event->terminal, "cancel target produced no first token");
        Expect(requestA->Cancel(), "long request completed before cancellation");
        const auto result = Drain({requestA, requestB});
        Expect(result[0].terminal.summary.status == wire::RequestStatus::Cancelled, "explicit cancel did not reach terminal");
        Same(result[1].tokens, cases[1].expected, "cancel peer");
    }
    WaitReclaimed(client);
    {
        auto a = CreateEventually(model), b = CreateEventually(model);
        auto longOptions = cases[0].options;
        longOptions.generation.maxNewTokens = 256;
        auto requestA = a->Submit(cases[0].prompt, longOptions);
        auto requestB = b->Submit(cases[1].prompt, cases[1].options);
        const auto result = Drain({requestB}); // Keep A completely unread; both pipe readers keep running.
        Same(result[0].tokens, cases[1].expected, "slow-consumer peer");
        wire::RequestInfo state;
        const auto deadline = Clock::now() + 40s;
        do {
            state = requestA->Query();
            if (state.completionReady) break;
            std::this_thread::sleep_for(5ms);
        } while (Clock::now() < deadline);
        Expect(state.completionReady && state.summary.endReason == wire::EndReason::OutputBackpressure,
               "unread long request did not hit local output backpressure");
        Expect(state.lastSentSequence <= 32 && state.bufferedTokens <= 128 && state.summary.generatedTokens <= 160,
               "IPC/core backpressure exceeded reserved token windows");
        Expect(client->Query().ipcOutstandingRequests != 0, "unread terminal released remote reservation");
        requestA->Release();
        requestA.reset();
        WaitReclaimed(client);
    }
    WaitReclaimed(client);
    std::cout << "[PASS] explicit cancel, unread 256-token backpressure and private release drain\n";
}

void TestSessionRecreate(const std::shared_ptr<host::InferenceClient>& client,
                         const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases) {
    std::set<uint64_t> identities;
    for (size_t round = 0; round < 2; ++round) {
        std::vector<std::shared_ptr<host::InferenceSession>> sessions;
        std::vector<std::shared_ptr<host::InferenceRequest>> requests;
        // Refill all four slots after releasing the preceding wave. Creating
        // just one replacement could hide leaked Session reservations.
        for (size_t index = 0; index < 4; ++index) {
            auto session = CreateEventually(model);
            Expect(identities.insert(session->Info().sessionId).second, "Released Session ID was reused in one epoch");
            sessions.push_back(std::move(session));
        }
        for (size_t index = 0; index < sessions.size(); ++index) {
            const auto& test = cases[(index + round + 1) % cases.size()];
            requests.push_back(sessions[index]->Submit(test.prompt, test.options));
        }
        const auto output = Drain(requests);
        for (size_t index = 0; index < sessions.size(); ++index) {
            const auto& test = cases[(index + round + 1) % cases.size()];
            Same(output[index].tokens, test.expected, "recreated Session / round " + std::to_string(round));
            Expect(output[index].terminal.summary.status == wire::RequestStatus::Completed,
                   "Recreated Session failed to complete");
        }
        requests.clear();
        const auto& appended = cases[(round + 1) % cases.size()];
        auto continuation = sessions[0]->Submit(appended.continuation, appended.continuationOptions);
        Same(Drain({continuation})[0].tokens, appended.expectedContinuation, "recreated Session continuation");
        continuation.reset();
        sessions.clear();
        WaitReclaimed(client);
    }
    std::cout << "[PASS] all four Session slots reused with fresh KV/recurrent state and independent continuations\n";
}

struct IdentityHistory {
    std::set<uint64_t> sessions;
    std::set<uint64_t> requests;
    void Remember(const std::shared_ptr<host::InferenceSession>& session) {
        Expect(sessions.insert(session->Info().sessionId).second, "Extended fixture reused a Session ID within one epoch");
    }
    void Remember(const std::shared_ptr<host::InferenceRequest>& request) {
        Expect(requests.insert(request->Id()).second, "Extended fixture reused a request ID within one epoch");
    }
};

void WaitFullyReclaimed(const std::shared_ptr<host::InferenceClient>& client, const std::string& stage) {
    const auto deadline = Clock::now() + 20s;
    do {
        const auto remote = client->Query();
        const auto local = client->Snapshot();
        Expect(remote.accepting && !remote.faulted && local.status == wire::DomainStatus::Ready,
               stage + ": healthy cleanup unexpectedly stopped the Worker");
        // remote.pendingCommands includes this worker.query itself. Do not
        // assert zero for that self-observation. Full Session-slot recovery is
        // checked independently by allocating all four slots in the next wave.
        if (local.outstandingRequests == 0 && local.retainedRequestHandles == 0 &&
            local.pendingCommands == 0 && local.queuedCommands == 0 && local.reservedTokens == 0 &&
            remote.ipcOutstandingRequests == 0 && remote.retainedRequestHandles == 0 &&
            remote.outstandingRequests == 0 && remote.activeRequests == 0 && remote.queuedRequests == 0 &&
            remote.reservedTokens == 0 && remote.reservedSessionCacheBytes == 0) {
            std::cout << "[RECLAIM] stage=" << stage
                      << " local_request_handles=0 remote_request_handles=0 ipc_outstanding=0 scheduler_cache_reservation=0\n";
            return;
        }
        std::this_thread::sleep_for(2ms);
    } while (Clock::now() < deadline);
    throw std::runtime_error(stage + ": request handles, output credits or Scheduler reservations did not return to zero");
}

void ExpectFifthSessionRefused(const std::shared_ptr<host::InferenceModel>& model) {
    bool refused = false;
    try { model->CreateSession(); }
    catch (const host::InferenceClientError& error) {
        if (error.RemoteCode() != wire::ErrorCode::ResourceLimit) throw;
        refused = true;
    }
    Expect(refused, "Worker admitted a fifth Session beyond the four-slot fixture limit");
}

void VerifyCompleted(const Output& output, const Case& test, bool continuation, const std::string& label) {
    const auto& expected = continuation ? test.expectedContinuation : test.expected;
    const auto& prompt = continuation ? test.continuation : test.prompt;
    Same(output.tokens, expected, label);
    Expect(output.terminal.summary.status == wire::RequestStatus::Completed &&
               output.terminal.summary.endReason == wire::EndReason::Length &&
               output.terminal.summary.promptTokensProcessed == prompt.size() &&
               output.terminal.summary.generatedTokens == expected.size(),
           label + ": complete token output had an inconsistent terminal summary");
}

template <typename Function>
void ContextError(Function function, wire::ErrorCode expected) {
    try { function(); }
    catch (const host::InferenceClientError& error) {
        Expect(error.RemoteCode() == expected, "Unexpected context error: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("Expected context operation rejection was absent");
}

void TestSessionContext(const std::shared_ptr<host::InferenceClient>& client,
                        const std::shared_ptr<host::InferenceModel>& model, const ReferenceSet& reference) {
    const auto& cases = reference.cases;
    Expect(reference.contextCases.size() == 4 && cases.size() == 4, "Missing direct context references");
    std::vector<std::shared_ptr<host::InferenceSession>> sessions;
    std::vector<std::shared_ptr<host::InferenceRequest>> requests;
    std::set<uint64_t> identities;
    for (const auto& test : cases) {
        sessions.push_back(CreateEventually(model));
        identities.insert(sessions.back()->Info().sessionId);
        requests.push_back(sessions.back()->Submit(test.prompt, test.options));
    }
    auto output = Drain(requests);
    for (size_t index = 0; index < 4; ++index)
        VerifyCompleted(output[index], cases[index], false, "context initial " + std::to_string(index));
    requests.clear();
    for (size_t index = 0; index < 4; ++index) {
        const auto& context = reference.contextCases[index];
        const auto restored = sessions[index]->Truncate(context.target);
        Expect(!restored.busy && restored.contextLength == context.restored &&
               sessions[index]->Query().contextLength == context.restored,
               "Remote truncate did not match the direct checkpoint");
        ContextError([&] { sessions[index]->Truncate(context.restored + 1); }, wire::ErrorCode::InvalidArgument);
        Expect(sessions[index]->Query().contextLength == context.restored, "Rejected truncate changed context");
        requests.push_back(sessions[index]->Submit(context.replay, context.options));
    }
    output = Drain(requests);
    for (size_t index = 0; index < 4; ++index) {
        const auto& context = reference.contextCases[index];
        Same(output[index].tokens, context.expected, "context replay " + std::to_string(index));
        const auto& summary = output[index].terminal.summary;
        Expect(summary.status == wire::RequestStatus::Completed && summary.endReason == wire::EndReason::Length &&
               summary.promptTokensProcessed == context.replay.size() && summary.generatedTokens == context.expected.size() &&
               sessions[index]->Query().contextLength == context.finalLength,
               "Replayed request changed usage, completion or final cache length");
        std::cout << "[CONTEXT] request=" << index << " target=" << context.target << " restored=" << context.restored
                  << " replay=" << context.replay.size() << " generated=" << context.expected.size()
                  << " final_length=" << context.finalLength << " exact_tokens=true\n";
    }
    std::cout << "[PASS] four remote truncations, actual checkpoint replay and exact direct-reference tokens/usage\n";
    requests.clear();
    for (size_t index = 0; index < 4; ++index) {
        Expect(sessions[index]->Reset().contextLength == 0, "Remote reset retained context");
        requests.push_back(sessions[index]->Submit(cases[index].prompt, cases[index].options));
    }
    output = Drain(requests);
    for (size_t index = 0; index < 4; ++index)
        VerifyCompleted(output[index], cases[index], false, "after reset " + std::to_string(index));
    requests.clear();
    for (size_t index = 0; index < 4; ++index)
        requests.push_back(sessions[index]->Submit(cases[index].continuation, cases[index].continuationOptions));
    output = Drain(requests);
    for (size_t index = 0; index < 4; ++index) {
        VerifyCompleted(output[index], cases[index], true, "reset continuation " + std::to_string(index));
        Expect(sessions[index]->Query().contextLength == cases[index].prompt.size() + cases[index].expected.size() +
               cases[index].continuation.size() + cases[index].expectedContinuation.size(), "Reset continuation lost history");
    }
    requests.clear();
    std::cout << "[PASS] four independent resets, seeded/penalized replay and cache continuation\n";
    sessions[0]->Reset(); sessions[1]->Reset();
    auto longOptions = cases[0].options;
    longOptions.generation.maxNewTokens = 128;
    auto active = sessions[0]->Submit(cases[0].prompt, longOptions);
    ContextError([&] { sessions[0]->Reset(); }, wire::ErrorCode::Busy);
    ContextError([&] { sessions[0]->Truncate(0); }, wire::ErrorCode::Busy);
    auto peer = sessions[1]->Submit(cases[1].prompt, cases[1].options);
    active->Cancel();
    output = Drain({active, peer});
    Expect(output[0].terminal.summary.status == wire::RequestStatus::Cancelled, "Busy context probe did not cancel cleanly");
    VerifyCompleted(output[1], cases[1], false, "context cancelled peer");
    active.reset(); peer.reset();
    Expect(sessions[0]->Reset().contextLength == 0, "Cancelled context could not reset");
    auto recovered = sessions[0]->Submit(cases[0].prompt, cases[0].options);
    VerifyCompleted(Drain({recovered})[0], cases[0], false, "reset after cancellation");
    recovered.reset();
    sessions.clear();
    WaitFullyReclaimed(client, "session-context released");
    for (size_t index = 0; index < 4; ++index) {
        auto session = CreateEventually(model);
        Expect(identities.insert(session->Info().sessionId).second, "Context lifecycle reused a stale Session ID");
        Expect(session->Truncate(0).contextLength == 0 && session->Reset().contextLength == 0,
               "Reused physical Session slot retained state");
        sessions.push_back(std::move(session));
    }
    sessions.clear();
    WaitFullyReclaimed(client, "session-context replacements released");
    std::cout << "[PASS] Busy mutations, cancellation recovery, peer isolation and four-slot lifecycle reuse\n";
}

void ExtendedWave(const std::shared_ptr<host::InferenceClient>& client,
                  const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases,
                  size_t width, size_t offset, const std::string& label, IdentityHistory& identities,
                  bool requireOverlap = false) {
    std::vector<std::shared_ptr<host::InferenceSession>> sessions;
    std::vector<std::shared_ptr<host::InferenceRequest>> requests;
    for (size_t index = 0; index < width; ++index) {
        auto session = CreateEventually(model);
        identities.Remember(session);
        sessions.push_back(std::move(session));
    }
    if (width == 4) ExpectFifthSessionRefused(model);
    const auto start = Clock::now();
    for (size_t index = 0; index < width; ++index) {
        const auto& test = cases[(index + offset) % cases.size()];
        auto request = sessions[index]->Submit(test.prompt, test.options);
        identities.Remember(request);
        requests.push_back(std::move(request));
    }
    if (requireOverlap && width > 1) {
        const auto state = client->Query();
        Expect(state.activeRequests + state.queuedRequests >= 2, label + ": no two accepted requests overlapped");
    }
    const auto output = Drain(requests, start, 300s);
    for (size_t index = 0; index < width; ++index) {
        const auto& test = cases[(index + offset) % cases.size()];
        VerifyCompleted(output[index], test, false, label + " prompt " + std::to_string(index));
        const auto state = sessions[index]->Query();
        Expect(!state.busy && state.contextLength == test.prompt.size() + test.expected.size(),
               label + ": initial request did not retain its complete private context");
        std::cout << "[MEASURE] case=" << label << " sessions=" << width << " request=" << index
                  << " prompt=" << test.prompt.size() << " prefill_chunk=" << test.options.prefillChunkSize
                  << " generated=" << output[index].tokens.size() << " caller_first_token_ms=" << output[index].firstTokenMs << '\n';
    }
    requests.clear();
    for (size_t index = 0; index < width; ++index) {
        const auto& test = cases[(index + offset) % cases.size()];
        auto request = sessions[index]->Submit(test.continuation, test.continuationOptions);
        identities.Remember(request);
        requests.push_back(std::move(request));
    }
    const auto continued = Drain(requests, Clock::now(), 300s);
    for (size_t index = 0; index < width; ++index) {
        const auto& test = cases[(index + offset) % cases.size()];
        VerifyCompleted(continued[index], test, true, label + " continuation " + std::to_string(index));
        const auto state = sessions[index]->Query();
        Expect(!state.busy && state.contextLength == test.prompt.size() + test.expected.size() +
                   test.continuation.size() + test.expectedContinuation.size(),
               label + ": continuation lost its private KV/recurrent history");
    }
    requests.clear();
    sessions.clear();
    WaitFullyReclaimed(client, label);
}

void ProbeAllSessionSlots(const std::shared_ptr<host::InferenceClient>& client,
                         const std::shared_ptr<host::InferenceModel>& model,
                         IdentityHistory& identities, const std::string& stage) {
    std::vector<std::shared_ptr<host::InferenceSession>> sessions;
    for (size_t index = 0; index < 4; ++index) {
        auto session = CreateEventually(model);
        identities.Remember(session);
        sessions.push_back(std::move(session));
    }
    ExpectFifthSessionRefused(model);
    sessions.clear();
    WaitFullyReclaimed(client, stage);
    std::cout << "[PASS] " << stage << " all_four_session_slots_restored=true\n";
}

void TestLongContext(const std::shared_ptr<host::InferenceClient>& client,
                     const std::shared_ptr<host::InferenceModel>& model, const ReferenceSet& reference) {
    const auto& cases = reference.longCases;
    Expect(reference.capacity >= 2048 && cases.size() == 4, "Long-context fixture requires its independent references");
    IdentityHistory identities;
    for (const size_t width : {size_t{1}, size_t{2}, size_t{4}}) {
        ExtendedWave(client, model, cases, width, 0, "long-context_" + std::to_string(width), identities, true);
    }
    Expect(cases.back().prompt.size() + cases.back().expected.size() + cases.back().continuation.size() +
               cases.back().expectedContinuation.size() == reference.capacity,
           "Longest continuation did not exercise the exact Session capacity boundary");

    for (const bool duringPrefill : {true, false}) {
        const std::string label = duringPrefill ? "long-context_cancel_prefill" : "long-context_cancel_decode";
        const auto& targetCase = duringPrefill ? cases.back() : cases.front();
        const auto& peerCase = duringPrefill ? cases.front() : cases.back();
        auto a = CreateEventually(model), b = CreateEventually(model);
        identities.Remember(a); identities.Remember(b);
        auto options = targetCase.options;
        if (duringPrefill) options.prefillChunkSize = 8;
        else options.generation.maxNewTokens = 256;
        auto target = a->Submit(targetCase.prompt, options);
        auto peer = b->Submit(peerCase.prompt, peerCase.options);
        identities.Remember(target); identities.Remember(peer);
        Tokens prefix;
        if (duringPrefill) {
            const auto deadline = Clock::now() + 300s;
            bool observed = false;
            while (Clock::now() < deadline) {
                const auto state = target->Query();
                Expect(!state.completionReady && state.summary.promptTokensProcessed < targetCase.prompt.size(),
                       "Long prefill completed before the partial-prefill cancellation observation");
                if (state.summary.promptTokensProcessed > 0) { observed = true; break; }
                std::this_thread::sleep_for(2ms);
            }
            Expect(observed, "No incremental long prefill progress was observed");
        } else {
            const auto first = target->Read(300s);
            Expect(first.status == host::InferenceReadStatus::Event && first.event && !first.event->terminal,
                   "Long decode cancellation target produced no initial token");
            prefix.push_back(first.event->tokenId);
        }
        Expect(target->Cancel(), "Long cancellation target completed before cancellation");
        const auto result = Drain({target, peer}, Clock::now(), 300s);
        Expect(result[0].terminal.summary.status == wire::RequestStatus::Cancelled &&
                   result[0].terminal.summary.endReason == wire::EndReason::Cancelled,
               label + ": cancellation did not produce its distinct terminal");
        if (duringPrefill) {
            Expect(result[0].tokens.empty() && result[0].terminal.summary.generatedTokens == 0 &&
                       result[0].terminal.summary.promptTokensProcessed > 0 &&
                       result[0].terminal.summary.promptTokensProcessed < targetCase.prompt.size(),
                   "Partial-prefill cancellation reached decode or lost its completed prefill progress");
        } else {
            prefix.insert(prefix.end(), result[0].tokens.begin(), result[0].tokens.end());
            Expect(prefix.size() <= reference.longCancellationExpected.size() &&
                       result[0].terminal.summary.generatedTokens == prefix.size(),
                   "Cancelled decode token accounting exceeded the independent reference");
            Same(prefix, Tokens(reference.longCancellationExpected.begin(), reference.longCancellationExpected.begin() + prefix.size()),
                 "long cancelled token prefix");
        }
        VerifyCompleted(result[1], peerCase, false, label + " peer");
        target.reset(); peer.reset(); a.reset();
        auto continuation = b->Submit(peerCase.continuation, peerCase.continuationOptions);
        identities.Remember(continuation);
        VerifyCompleted(Drain({continuation}, Clock::now(), 300s)[0], peerCase, true, label + " peer continuation");
        const auto state = b->Query();
        Expect(!state.busy && state.contextLength == peerCase.prompt.size() + peerCase.expected.size() +
                   peerCase.continuation.size() + peerCase.expectedContinuation.size(),
               label + ": cancellation changed the surviving Session context");
        continuation.reset(); b.reset();
        WaitFullyReclaimed(client, label);
        std::cout << "[PASS] " << label << " cancelled_prompt_tokens=" << result[0].terminal.summary.promptTokensProcessed
                  << " cancelled_generated_tokens=" << result[0].terminal.summary.generatedTokens << '\n';
    }
    // Querying contextLength == 0 alone would miss stale DeltaNet payload.
    // Execute fresh Sessions against the independent short references too.
    ExtendedWave(client, model, reference.cases, 4, 0, "long-context_after_cancel", identities);
    ProbeAllSessionSlots(client, model, identities, "long-context_final_reuse");
    std::cout << "[PASS] long-context context=" << reference.capacity
              << " lengths=4 widths=1,2,4 incremental_prefill=true exact_capacity_continuation=true cancel_prefill_decode=true\n";
}

void TestSoak(const std::shared_ptr<host::InferenceClient>& client,
              const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases, size_t rounds) {
    Expect(rounds >= 20 && cases.size() == 4, "Soak requires at least twenty complete four-Session rounds");
    IdentityHistory identities;
    for (size_t round = 0; round < rounds; ++round) {
        ExtendedWave(client, model, cases, 4, round, "soak_round_" + std::to_string(round + 1), identities);
        std::cout << "[PASS] soak round=" << round + 1 << " sessions=4 prompt_and_continuation_token_parity=true\n";
    }
    ProbeAllSessionSlots(client, model, identities, "soak_final_reuse");
    std::cout << "[PASS] soak rounds=" << rounds << " unique_session_ids=" << identities.sessions.size()
              << " unique_request_ids=" << identities.requests.size() << " request_handles_reclaimed=true\n";
}

void TestParentReleaseAndShutdown(const std::shared_ptr<host::InferenceClient>& client,
                                  const std::shared_ptr<host::InferenceModel>& model, const std::vector<Case>& cases) {
    auto session = CreateEventually(model);
    auto request = session->Submit(cases[0].prompt, cases[0].options);
    session->Release();
    model->Release();
    Same(Drain({request})[0].tokens, cases[0].expected, "released-parent request");
    WaitReclaimed(client);
    request.reset(); session.reset();
    client->Shutdown(20s);
    Expect(client->Snapshot().status == wire::DomainStatus::Closed, "Worker shutdown did not close domain");
    std::cout << "[PASS] parent release retained in-flight request; graceful shutdown barrier\n";
}

std::shared_ptr<host::InferenceClient> StartWorker(Backend backend) {
    const std::filesystem::path executable(AILA_IPC_WORKER_PATH);
    host::InferenceClientOptions options;
    options.handshakeTimeout = 10s;
    options.commandTimeout = 120s;
    options.shutdownTimeout = 20s;
    if (IsHybrid(backend)) {
        auto& capabilities = options.requiredCapabilities;
        capabilities.erase(std::remove(capabilities.begin(), capabilities.end(), "qwen3_nf4_bf16_kv"), capabilities.end());
        capabilities.emplace_back(wire::model_capability(BackendName(backend)));
    }
    return host::InferenceClient::Start(executable.parent_path(), executable, AILA_BUILD_ID, options);
}

std::shared_ptr<host::InferenceModel> LoadModel(const std::shared_ptr<host::InferenceClient>& client,
                                               const std::string& directory, Backend backend,
                                               const ReferenceSet& reference) {
    host::InferenceModelOptions options;
    options.maxSequenceLength = reference.capacity;
    options.maxSessions = 4;
    std::shared_ptr<host::InferenceModel> model;
    if (backend == Backend::Qwen3) model = client->LoadQwen3(directory, options);
    else if (backend == Backend::Qwen35) model = client->LoadQwen35(directory, options);
    else model = client->LoadQwen35Bf16(directory, options);
    const auto& info = model->Info();
    for (const auto candidate : {Backend::Qwen3, Backend::Qwen35, Backend::Qwen35Bf16}) {
        const bool advertised = std::find(info.capabilities.begin(), info.capabilities.end(),
                                          wire::model_capability(BackendName(candidate))) != info.capabilities.end();
        Expect(advertised == (backend == candidate), "Loaded model advertised the wrong backend capability");
    }
    Expect(info.vocabSize == reference.vocabSize && info.maxSequenceLength == reference.capacity &&
               info.sessionReservationBytes == reference.sessionCacheBytes,
           "IPC model metadata differs from the direct model");
    return model;
}

void Run(const std::string& directory, Backend backend, const std::set<std::string>& selected,
         const FixtureOptions& fixture) {
    aila::set_log_level(aila::LogLevel::Warning);
    const bool longContext = selected.count("long-context") != 0;
    const bool soak = selected.count("soak") != 0;
    const bool thinking = selected.count("thinking-budget") != 0;
    const bool contextOperations = selected.count("session-context") != 0;
    std::cout << "[RUN] backend=" << BackendName(backend) << " model=" << directory
              << " context=" << fixture.context << " suite=" << (selected.empty() ? "default" : "selected") << " cases=";
    if (selected.empty()) {
        for (const auto* name : DefaultCaseNames) std::cout << name << ' ';
    } else {
        for (const auto& name : selected) std::cout << name << ' ';
    }
    std::cout << "soak_rounds=" << (soak ? fixture.soakRounds : 0) << '\n';
    const auto reference = References(directory, backend, fixture, longContext, soak, thinking, contextOperations);
    const auto& cases = reference.cases;
    std::cout << "[PASS] direct device owners released before isolated Worker startup\n";
    auto client = StartWorker(backend);
    const auto firstEpoch = client->Snapshot().epoch;
    auto model = LoadModel(client, directory, backend, reference);
    const auto run = [&](const char* name) { return selected.empty() || selected.count(name) != 0; };
    if (run("concurrency")) TestConcurrency(client, model, cases);
    if (run("cancel-backpressure")) TestCancelAndBackpressure(client, model, cases);
    if (run("session-recreate")) TestSessionRecreate(client, model, cases);
    if (longContext) TestLongContext(client, model, reference);
    if (soak) TestSoak(client, model, reference.soakCases, fixture.soakRounds);
    if (thinking) TestThinkingBudget(client, model, reference.thinkingCases);
    if (contextOperations) TestSessionContext(client, model, reference);
    if (run("parent-shutdown")) TestParentReleaseAndShutdown(client, model, cases);
    else {
        model->Release();
        WaitReclaimed(client);
        client->Shutdown(20s);
        Expect(client->Snapshot().status == wire::DomainStatus::Closed, "Worker shutdown did not close domain");
    }
    model.reset();
    client.reset();
    if (run("epoch")) {
        auto restarted = StartWorker(backend);
        Expect(restarted->Snapshot().epoch != firstEpoch, "restarted Worker reused epoch");
        auto newModel = LoadModel(restarted, directory, backend, reference);
        auto session = CreateEventually(newModel);
        auto request = session->Submit(cases[0].prompt, cases[0].options);
        Same(Drain({request})[0].tokens, cases[0].expected, "new epoch inference");
        request.reset();
        request = session->Submit(cases[0].continuation, cases[0].continuationOptions);
        Same(Drain({request})[0].tokens, cases[0].expectedContinuation, "new epoch continuation");
        request.reset(); session.reset(); newModel.reset();
        WaitReclaimed(restarted);
        restarted->Shutdown(20s);
        Expect(restarted->Snapshot().status == wire::DomainStatus::Closed, "Restarted Worker shutdown did not close domain");
        std::cout << "[PASS] fresh Worker epoch, reload and token parity\n";
    }
    std::cout << "[PASS] suite=" << (selected.empty() ? "default" : "selected")
              << " completed_cases=" << (selected.empty() ? DefaultCaseNames.size() : selected.size())
              << " context=" << fixture.context << " all_selected_cases_completed=true\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string directory;
        Backend backend = Backend::Qwen3;
        bool backendSpecified = false;
        bool contextSpecified = false, soakRoundsSpecified = false;
        FixtureOptions fixture;
        std::set<std::string> selected;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" || argument == "--list-cases") {
                std::cout << "Usage: " << argv[0] << " [--backend qwen3_nf4|qwen35_nf4|qwen35_bf16] [--model <directory>]\n"
                             "       [--context 512|2048|4096] [--case <name> ...] [--soak-rounds 20..10000]\n"
                             "Defaults: Qwen3, context 512, the original five lightweight cases.\n"
                             "Environment: AILA_QWEN3_TEST_MODEL / AILA_QWEN35_TEST_MODEL / AILA_QWEN35_BF16_TEST_MODEL by backend.\n"
                             "Default cases:";
                for (const auto* name : DefaultCaseNames) std::cout << ' ' << name;
                std::cout << "\nOpt-in cases:";
                for (const auto* name : ExtendedCaseNames) std::cout << ' ' << name;
                std::cout << "\nlong-context requires --context 2048 or 4096; soak defaults to 20 four-Session rounds.\n"
                             "Repeat --case to combine cases; excluded cases are not claimed as run.\n"
                             "Direct references/soak, Worker and optional epoch restart run sequentially.\n"
                             "Memory samples are Context-tracked USM payload, not total RSS or VRAM.\n";
                return 0;
            }
            Expect(index + 1 < argc, "Missing value for " + argument);
            const std::string value = argv[++index];
            if (argument == "--backend") {
                Expect(!backendSpecified && (value == "qwen3_nf4" || value == "qwen35_nf4" || value == "qwen35_bf16"),
                       "Expected one supported --backend");
                backend = value == "qwen3_nf4" ? Backend::Qwen3 : value == "qwen35_nf4" ? Backend::Qwen35 : Backend::Qwen35Bf16;
                backendSpecified = true;
            } else if (argument == "--model") {
                Expect(directory.empty() && !value.empty(), "Expected one nonempty --model directory");
                directory = value;
            } else if (argument == "--case") {
                Expect(std::find(DefaultCaseNames.begin(), DefaultCaseNames.end(), value) != DefaultCaseNames.end() ||
                           std::find(ExtendedCaseNames.begin(), ExtendedCaseNames.end(), value) != ExtendedCaseNames.end(),
                       "Unknown fixture case: " + value);
                Expect(selected.insert(value).second, "Duplicate fixture case: " + value);
            } else if (argument == "--context") {
                Expect(!contextSpecified, "Duplicate --context");
                fixture.context = static_cast<uint32_t>(CountOption(value, 512, 4096, argument));
                Expect(fixture.context == 512 || fixture.context == 2048 || fixture.context == 4096,
                       "--context must be 512, 2048 or 4096");
                contextSpecified = true;
            } else if (argument == "--soak-rounds") {
                Expect(!soakRoundsSpecified, "Duplicate --soak-rounds");
                fixture.soakRounds = CountOption(value, 20, 10000, argument);
                soakRoundsSpecified = true;
            } else throw std::invalid_argument("Unknown argument: " + argument);
        }
        Expect(selected.count("long-context") == 0 || fixture.context >= 2048,
               "--case long-context requires --context 2048 or 4096");
        Expect(!soakRoundsSpecified || selected.count("soak") != 0, "--soak-rounds requires --case soak");
        if (directory.empty()) {
            char* value = nullptr;
            size_t length = 0;
            if (_dupenv_s(&value, &length, ModelEnvironment(backend)) == 0 && value) directory = value;
            std::free(value);
        }
        if (directory.empty()) {
            std::cout << "[SKIP] Set " << ModelEnvironment(backend) << " or pass --model.\n";
            return 77;
        }
        directory = std::filesystem::absolute(directory).lexically_normal().string();
        Expect(std::filesystem::is_directory(directory), "Model directory does not exist");
        Run(directory, backend, selected, fixture);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
