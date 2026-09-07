#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "inference/Scheduler.hpp"
#include "inference/Session.hpp"
#include "models/Qwen35HybridBnb4Backend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"
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
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int>;
using Plan = std::vector<Tokens>;
using LogitsTrace = std::vector<std::vector<float>>;

constexpr int kCapacity = 512;
constexpr size_t kSnapshots = 4;
constexpr int kSnapshotStep = 64;
constexpr int kGenerated = 6;
constexpr int kContinuationGenerated = 3;
constexpr int kMaxSteps = 2 * kCapacity;
constexpr auto kRequestTimeout = 90s;

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void expect_error(ErrorCode code, Function&& operation, const std::string& label) {
    try {
        operation();
    } catch (const Error& error) {
        expect(error.Code() == code, label + ": wrong inference error: " + error.what());
        return;
    }
    throw std::runtime_error(label + ": expected an inference error");
}

std::optional<std::string> environment_value(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) != 0 || !value) {
        if (value) std::free(value);
        return std::nullopt;
    }
    const std::unique_ptr<char, decltype(&std::free)> owner(value, &std::free);
    return std::string(value);
#else
    const char* value = std::getenv(name);
    return value ? std::optional<std::string>(value) : std::nullopt;
#endif
}

// Only change environment settings while there is no Scheduler worker. Scoped
// KV admission reads are intentionally dynamic; no kernel option is toggled.
class ScopedEnvironment final {
public:
    ScopedEnvironment(const char* variable, std::optional<std::string> value)
        : name(variable), original(environment_value(variable)) {
        expect(Set(value) == 0, "Could not set fixture environment: " + name);
    }
    ~ScopedEnvironment() { Set(original); }
    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

private:
    int Set(const std::optional<std::string>& value) const noexcept {
#ifdef _WIN32
        return _putenv_s(name.c_str(), value ? value->c_str() : "");
#else
        return value ? setenv(name.c_str(), value->c_str(), 1) : unsetenv(name.c_str());
#endif
    }

    std::string name;
    std::optional<std::string> original;
};

class ScopedKvOverride final {
public:
    explicit ScopedKvOverride(bool enabled) : original(aila::env::g_kv_quant_override) {
        aila::env::g_kv_quant_override = enabled;
    }
    ~ScopedKvOverride() { aila::env::g_kv_quant_override = original; }
    ScopedKvOverride(const ScopedKvOverride&) = delete;
    ScopedKvOverride& operator=(const ScopedKvOverride&) = delete;

private:
    bool original;
};

class ScopedSnapshotStep final {
public:
    ScopedSnapshotStep() : original(aila::env::g_q35_prefill_step_override) {
        aila::env::g_q35_prefill_step_override = kSnapshotStep;
    }
    ~ScopedSnapshotStep() { aila::env::g_q35_prefill_step_override = original; }

private:
    int original;
};

class ScopedLogLevel final {
public:
    explicit ScopedLogLevel(aila::LogLevel level) : previous(aila::get_log_level()) {
        aila::set_log_level(level);
    }
    ~ScopedLogLevel() { aila::set_log_level(previous); }

private:
    aila::LogLevel previous;
};

// A workspace growth log is flushed after the request releases its execution
// locks. Hold this boundary so all peers can be admitted deterministically.
// This callback owns no model/session/request and submits no device work.
class PrefillGate final {
public:
    PrefillGate() : previousLevel(aila::get_log_level()) {
        aila::set_log_level(aila::LogLevel::Info);
        aila::set_log_callback(&Callback, this);
    }
    ~PrefillGate() {
        Release();
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
        if (!message || !std::strstr(message, "[Qwen3.5] Runtime buffers resized")) return;
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

// Declare after Scheduler so exceptional unwinding releases a blocked worker
// before Scheduler's destructor joins it.
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

Plan make_plan(const Tokens& source, std::initializer_list<size_t> shapes) {
    size_t total = 0;
    for (size_t shape : shapes) {
        expect(shape > 0 && total + shape <= kCapacity, "Invalid plan shape");
        total += shape;
    }
    const auto tokens = repeat_tokens(source, total);
    Plan plan;
    size_t offset = 0;
    for (size_t shape : shapes) {
        plan.emplace_back(tokens.begin() + offset, tokens.begin() + offset + shape);
        offset += shape;
    }
    return plan;
}

void expect_logits(const std::vector<float>& actual, const std::vector<float>& expected,
                   const std::string& label) {
    expect(actual.size() == expected.size() && !actual.empty(), label + ": invalid logits dimensions");
    for (size_t i = 0; i < actual.size(); ++i) {
        // Each reference uses exactly the same initial/incremental/decode
        // shapes. BF16-to-float conversion is lossless; require exact equality.
        if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
            size_t differences = 0;
            size_t nonfinite = 0;
            size_t maxIndex = 0;
            size_t actualArgmax = 0;
            size_t expectedArgmax = 0;
            float maxAbsolute = 0.0f;
            for (size_t j = 0; j < actual.size(); ++j) {
                if (!std::isfinite(actual[j]) || !std::isfinite(expected[j])) ++nonfinite;
                if (!std::isfinite(actual[j]) || actual[j] != expected[j]) ++differences;
                const float difference = std::abs(actual[j] - expected[j]);
                if (difference > maxAbsolute) {
                    maxAbsolute = difference;
                    maxIndex = j;
                }
                if (actual[j] > actual[actualArgmax]) actualArgmax = j;
                if (expected[j] > expected[expectedArgmax]) expectedArgmax = j;
            }
            std::ostringstream message;
            message << label << ": logit " << i << " differs (" << actual[i]
                    << " versus " << expected[i] << "), different=" << differences
                    << '/' << actual.size() << ", nonfinite=" << nonfinite
                    << ", max_abs=" << maxAbsolute << " at " << maxIndex;
            if (nonfinite == 0) {
                message << ", actual_argmax=" << actualArgmax << ':' << actual[actualArgmax]
                        << ", expected_argmax=" << expectedArgmax << ':' << expected[expectedArgmax];
            }
            throw std::runtime_error(message.str());
        }
    }
}

std::vector<float> forward(const std::shared_ptr<SessionState>& session,
                           const Tokens& tokens, const std::string& label) {
    const int before = session->ContextLength();
    auto logits = session->ForwardLogits(tokens);
    expect(session->ContextLength() == before + static_cast<int>(tokens.size()) && !session->Busy(),
           label + ": invalid context advance or retained session lease");
    expect(logits.size() == static_cast<size_t>(session->VocabSize()), label + ": invalid logits size");
    for (const float value : logits) {
        if (!std::isfinite(value)) throw std::runtime_error(label + ": non-finite logit");
    }
    return logits;
}

LogitsTrace capture_plan(const std::shared_ptr<SessionState>& session, const Plan& plan,
                         const std::string& label) {
    LogitsTrace result;
    result.reserve(plan.size());
    for (size_t i = 0; i < plan.size(); ++i) {
        result.push_back(forward(session, plan[i], label + " step " + std::to_string(i)));
    }
    return result;
}

class DrainContextOnExit final {
public:
    explicit DrainContextOnExit(Context& value) : context(value) {}
    ~DrainContextOnExit() {
        try { context.synchronize(); } catch (...) {}
    }

private:
    Context& context;
};

// Read the legacy backend's actual last valid row directly. Its small
// incremental path writes every row, while initial/large prefill and decode
// write only row zero; Tensor::shape may retain an older, larger capacity.
// This reference does not call LoadedModel/SessionState and therefore catches
// both a first-row wrapper bug and a stale-capacity last-row wrapper bug.
LogitsTrace capture_legacy_last_rows(const std::string& modelDirectory, const Plan& plan) {
    std::cout << "[RUN] Independent legacy logits reference for incremental 3/7/16, larger prefill and decode" << std::endl;
    ModelSpec spec;
    std::string error;
    const bool parsed = aila::modelspec::load_from_dir(modelDirectory, spec, &error);
    expect(parsed, "Could not load legacy reference specification: " + error);
    LogitsTrace reference;
    Context context;
    {
        ModelWeights weights;
        Qwen35HybridBnb4Backend backend;
        Tensor tokenIds;
        // Destruct before tokenIds/backend/weights on every exceptional exit.
        DrainContextOnExit drain(context);
        weights = LoadModelWeightsFromDir(modelDirectory, context);
        const bool loaded = backend.load(context, weights, spec, kCapacity, &error);
        expect(loaded, "Could not load legacy Hybrid reference: " + error);
        tokenIds = Tensor::allocate(context, {kCapacity}, dnnl::memory::data_type::s32);
        using bf16 = sycl::ext::oneapi::bfloat16;
        const size_t vocabSize = static_cast<size_t>(backend.vocab_size());
        expect(vocabSize > 0, "Legacy Hybrid reference has no vocabulary");
        std::vector<bf16> hostLogits(vocabSize);
        reference.reserve(plan.size());
        for (const auto& step : plan) {
            const int before = backend.get_current_context_len();
            context.memcpy_h2d(tokenIds.data(), step.data(), step.size() * sizeof(int));
            Tensor& logits = backend.forward(context, tokenIds.data_as<int>(), static_cast<int>(step.size()));
            const size_t validLastRow = before > 0 && step.size() > 1 && step.size() <= 16
                ? step.size() - 1 : 0;
            expect(logits.dtype() == dnnl::memory::data_type::bf16 &&
                       logits.numel() >= static_cast<int64_t>((validLastRow + 1) * vocabSize),
                   "Legacy reference does not contain the expected valid output row");
            context.memcpy_d2h(hostLogits.data(), logits.data_as<bf16>() + validLastRow * vocabSize,
                               vocabSize * sizeof(bf16));
            context.synchronize();
            expect(backend.get_current_context_len() == before + static_cast<int>(step.size()),
                   "Legacy reference did not consume the complete step");
            auto& output = reference.emplace_back(vocabSize);
            for (size_t i = 0; i < vocabSize; ++i) {
                output[i] = static_cast<float>(hostLogits[i]);
                if (!std::isfinite(output[i])) throw std::runtime_error("Non-finite legacy reference logit");
            }
        }
    }
    context.synchronize();
    expect(context.current_allocated_bytes() == 0, "Legacy reference retained tracked device allocations");
    // This Context is destroyed before any DeviceRuntime is created. Q35's
    // attention paths never enter the process-wide masked-attention cache.
    return reference;
}

void test_legacy_last_rows(const std::shared_ptr<LoadedModel>& model,
                           const Plan& plan, const LogitsTrace& reference) {
    std::cout << "[RUN] Shared Hybrid logits select the final valid row independently of buffer capacity" << std::endl;
    expect(plan.size() == reference.size(), "Legacy logits reference has the wrong number of steps");
    auto session = model->CreateSession();
    for (size_t i = 0; i < plan.size(); ++i) {
        expect_logits(forward(session, plan[i], "Shared last-row check"), reference[i],
                      "Shared versus legacy final valid row step " + std::to_string(i));
    }
}

// Recompute from the actual returned checkpoint, including when it is zero.
// Fixture checkpoints coincide with original step boundaries, so this never
// makes an invalid exact comparison between different GEMM reduction shapes.
void replay_from(const std::shared_ptr<SessionState>& session, int restored,
                  const Plan& plan, const LogitsTrace& reference, const std::string& label) {
    expect(session->ContextLength() == restored && plan.size() == reference.size(),
           label + ": invalid replay checkpoint");
    size_t offset = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const size_t end = offset + plan[i].size();
        if (end > static_cast<size_t>(restored)) {
            expect(offset == static_cast<size_t>(session->ContextLength()),
                   label + ": checkpoint changed the original step shape");
            expect_logits(forward(session, plan[i], label), reference[i],
                          label + " step " + std::to_string(i));
        }
        offset = end;
    }
    expect(session->ContextLength() == static_cast<int>(offset), label + ": replay omitted tokens");
}

std::string describe_tokens(const Tokens& tokens) {
    std::ostringstream text;
    text << '[';
    const size_t count = std::min(tokens.size(), size_t{16});
    for (size_t i = 0; i < count; ++i) text << (i ? "," : "") << tokens[i];
    if (count < tokens.size()) text << ",...";
    text << ']';
    return text.str();
}

void expect_tokens(const Tokens& actual, const Tokens& expected, const std::string& label) {
    if (actual == expected) return;
    size_t index = 0;
    while (index < actual.size() && index < expected.size() && actual[index] == expected[index]) ++index;
    throw std::runtime_error(label + ": token mismatch at " + std::to_string(index) +
                             " (actual=" + (index < actual.size() ? std::to_string(actual[index]) : "<end>") +
                             ", expected=" + (index < expected.size() ? std::to_string(expected[index]) : "<end>") +
                             "; actual size " + std::to_string(actual.size()) +
                             ", expected size " + std::to_string(expected.size()) +
                             "), actual trace=" + describe_tokens(actual) +
                             ", expected trace=" + describe_tokens(expected));
}

std::string describe_request(size_t promptTokens, const RequestOptions& options) {
    std::ostringstream text;
    text << "prompt=" << promptTokens << " prefill_shapes=[";
    size_t remaining = promptTokens;
    bool first = true;
    while (remaining != 0) {
        const size_t count = std::min(remaining, static_cast<size_t>(options.prefillChunkSize));
        text << (first ? "" : ",") << count;
        remaining -= count;
        first = false;
    }
    const auto& generation = options.generation;
    text << "] generate=" << generation.max_new_tokens
         << " sample=" << generation.do_sample << " fixed_seed=" << generation.use_fixed_seed
         << " seed=" << generation.sampling_seed
         << " repetition_penalty=" << generation.repetition_penalty
         << " presence_penalty=" << generation.presence_penalty
         << " frequency_penalty=" << generation.frequency_penalty
         << " stop_ids=" << describe_tokens(options.stopTokenIds);
    return text.str();
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
    // No stop IDs: require the complete trace and final-token cache commit.
    return options;
}

Tokens run_independent(const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                       const RequestOptions& options, const std::string& label) {
    const int originalLength = session->ContextLength();
    auto request = RequestState::Create(session, prompt, options);
    Tokens result;
    int terminalEvents = 0;
    for (int i = 0; i < kMaxSteps; ++i) {
        const auto step = request->Step();
        if (step.tokenId) result.push_back(*step.tokenId);
        if (step.becameTerminal) ++terminalEvents;
        if (IsTerminal(step.status)) break;
    }
    const auto snapshot = request->Snapshot();
    expect(snapshot.status == RequestStatus::Completed && terminalEvents == 1 &&
               snapshot.error.empty() && !snapshot.errorCode,
           label + ": independent request failed: " + snapshot.error);
    expect(result.size() == static_cast<size_t>(options.generation.max_new_tokens) &&
               snapshot.generatedIds == result && snapshot.promptTokensProcessed == prompt.size(),
           label + ": incomplete independent request trace");
    expect(!session->Busy() && session->ContextLength() ==
               originalLength + static_cast<int>(prompt.size() + result.size()),
           label + ": final token was not committed or session lease remains held");
    const auto repeated = request->Step();
    expect(!repeated.becameTerminal && !repeated.tokenId && repeated.status == RequestStatus::Completed,
           label + ": terminal request emitted another event");
    return result;
}

SchedulerOptions scheduler_options(size_t active, size_t ringSize, int chunkSize) {
    SchedulerOptions options;
    options.maxActiveRequests = active;
    options.maxOutstandingRequests = active;
    options.maxPromptTokens = kCapacity;
    options.maxGeneratedTokens = 128;
    options.maxOutstandingTokens = active * kCapacity;
    options.maxPrefillChunkSize = chunkSize;
    options.maxBufferedTokensPerRequest = ringSize;
    return options;
}

struct StreamObservation {
    Tokens tokens;
    RequestEvent terminal;
    bool ended = false;
};

bool read_one(const std::shared_ptr<ScheduledRequest>& handle, StreamObservation& observed,
              std::chrono::milliseconds timeout, const std::string& label) {
    auto read = handle->Read(timeout);
    if (read.status == ReadStatus::Timeout) return false;
    expect(read.status == ReadStatus::Event && read.event.has_value(),
           label + ": stream closed before its terminal event");
    const auto& event = *read.event;
    expect(!observed.ended && event.requestId == handle->Id() &&
               event.sequence == observed.tokens.size() + 1,
           label + ": invalid routing, sequence or repeated terminal");
    if (event.kind == RequestEventKind::Token) {
        expect(event.tokenId >= 0 && !IsTerminal(event.status) &&
                   event.endReason == RequestEndReason::None && !event.errorCode && event.error.empty(),
               label + ": malformed token event");
        observed.tokens.push_back(event.tokenId);
        expect(event.generatedTokens == observed.tokens.size(), label + ": wrong emitted token count");
    } else {
        expect(IsTerminal(event.status) && event.tokenId == -1 &&
                   event.generatedTokens == observed.tokens.size(), label + ": malformed terminal");
        observed.terminal = std::move(*read.event);
        observed.ended = true;
    }
    return true;
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
    for (const auto& handle : handles) {
        expect(handle->Read(0ms).status == ReadStatus::Closed,
               label + ": terminal event was not followed by Closed");
        const auto snapshot = handle->Snapshot();
        expect(snapshot.terminalConsumed && snapshot.bufferedTokens == 0,
               label + ": consumed stream retained buffered tokens");
    }
    return observations;
}

void expect_completed(const StreamObservation& observed,
                      const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                      const RequestOptions& options, int originalLength, const std::string& label) {
    expect(observed.ended && observed.terminal.status == RequestStatus::Completed &&
               observed.terminal.endReason == RequestEndReason::Length &&
               !observed.terminal.errorCode && observed.terminal.error.empty(),
           label + ": request did not complete: " + observed.terminal.error);
    expect(observed.tokens.size() == static_cast<size_t>(options.generation.max_new_tokens) &&
               observed.terminal.promptTokensProcessed == prompt.size(), label + ": incomplete trace");
    expect(!session->Busy() && session->ContextLength() ==
               originalLength + static_cast<int>(prompt.size() + observed.tokens.size()),
           label + ": final token was not committed or session lease remains held");
}

void expect_idle(const Scheduler& scheduler, const std::string& label) {
    const auto snapshot = scheduler.Snapshot();
    expect(!snapshot.faulted && snapshot.outstandingRequests == 0 && snapshot.activeRequests == 0 &&
               snapshot.queuedRequests == 0 && snapshot.reservedTokens == 0 &&
               snapshot.reservedSessionCacheBytes == 0,
           label + ": scheduler retained requests or reservations");
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
        "Describe an experiment comparing two ways to store solar energy. ",
        "A librarian maps historical travel diaries to the cities they describe. "
    }};
    const std::array<size_t, 4> lengths{{113, 79, 37, 129}};
    const std::array<int, 4> chunks{{96, 23, 7, 64}};
    const std::array<size_t, 4> continuationLengths{{7, 11, 5, 9}};
    const std::array<int, 4> continuationChunks{{3, 5, 2, 4}};
    const std::array<uint64_t, 4> seeds{{19, 113, 277, 691}};
    const auto continuation = tokenizer.encode(" Continue with a concrete example and explain the result. ");
    const auto probe = repeat_tokens(tokenizer.encode(" Therefore"), 1);
    std::array<TestCase, 4> result;
    for (size_t i = 0; i < result.size(); ++i) {
        result[i].prompt = repeat_tokens(tokenizer.encode(texts[i]), lengths[i]);
        result[i].continuation = repeat_tokens(continuation, continuationLengths[i]);
        result[i].probe = probe;
        result[i].initialOptions = request_options(i != 0, i >= 2, seeds[i], chunks[i]);
        result[i].continuationOptions = request_options(
            i != 0, i >= 2, seeds[i] + 1000, continuationChunks[i], kContinuationGenerated);
    }
    return result;
}

template <typename Function>
void expect_quantized_rejection(const std::shared_ptr<DeviceRuntime>& runtime,
                                Function&& operation, const std::string& label) {
    const auto before = runtime->Memory();
    try {
        operation();
    } catch (const Error& error) {
        expect(error.Code() == ErrorCode::InvalidArgument &&
                   std::string(error.what()).find("BF16") != std::string::npos,
               label + ": expected a BF16-only InvalidArgument diagnostic");
        const auto after = runtime->Memory();
        expect(after.currentBytes == before.currentBytes && after.peakBytes == before.peakBytes,
               label + ": rejected quantized KV allocated device resources");
        return;
    }
    throw std::runtime_error(label + ": effective quantized KV was admitted");
}

template <typename Function>
void quantized_matrix(const std::shared_ptr<DeviceRuntime>& runtime,
                       Function&& operation, const std::string& label) {
    {
        ScopedEnvironment global("AILA_KV_QUANT", "1");
        ScopedEnvironment scoped("AILA_VLM_KV_QUANT", std::nullopt);
        expect_quantized_rejection(runtime, operation, label + " global fallback");
    }
    {
        ScopedEnvironment global("AILA_KV_QUANT", "0");
        ScopedEnvironment scoped("AILA_VLM_KV_QUANT", "1");
        expect_quantized_rejection(runtime, operation, label + " scoped override");
    }
    {
        ScopedEnvironment global("AILA_KV_QUANT", "0");
        ScopedEnvironment scoped("AILA_VLM_KV_QUANT", std::nullopt);
        ScopedKvOverride override(true);
        expect_quantized_rejection(runtime, operation, label + " engine override fallback");
    }
}

void test_quantized_session_admission(const std::shared_ptr<DeviceRuntime>& runtime,
                                      const std::shared_ptr<LoadedModel>& model) {
    std::cout << "[RUN] Effective scoped/global KV configuration is checked before session allocation" << std::endl;
    const auto resident = runtime->Memory().currentBytes;
    auto anchor = model->CreateSession();
    quantized_matrix(runtime, [&] { model->CreateSession(); }, "Hybrid session admission");
    expect(model->ActiveSessions() == 1 && anchor->ContextLength() == 0 && !anchor->Busy(),
           "Quantized session rejection consumed a slot or changed the existing session");
    {
        ScopedEnvironment global("AILA_KV_QUANT", "1");
        ScopedEnvironment scoped("AILA_VLM_KV_QUANT", "0");
        ScopedKvOverride override(true);
        auto permitted = model->CreateSession();
        expect(model->ActiveSessions() == 2 && permitted->CacheBytes() == anchor->CacheBytes() &&
                   runtime->Memory().currentBytes == resident + 2 * anchor->CacheBytes(),
               "Scoped BF16 override failed to supersede global quantized KV");
    }
    anchor.reset();
    expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == resident,
           "KV configuration fixture retained session resources");
}

struct References {
    std::array<Tokens, 4> initial;
    size_t cacheBytes = 0;
    size_t hostBytes = 0;
};

References test_four_sessions(const std::shared_ptr<DeviceRuntime>& runtime,
                             const std::shared_ptr<LoadedModel>& model,
                             const std::array<TestCase, 4>& cases) {
    std::cout << "[RUN] Four Hybrid streams share weights and match independent greedy/seed/penalty requests" << std::endl;
    const size_t resident = runtime->Memory().currentBytes;
    References reference;
    reference.cacheBytes = model->SessionReservationBytes();
    reference.hostBytes = model->SessionHostReservationBytes();
    expect(reference.cacheBytes > 0 && reference.hostBytes > 0,
           "Hybrid fixture requires Full KV and recurrent host/device reservations");
    std::array<std::shared_ptr<SessionState>, 4> sessions;
    for (auto& session : sessions) {
        session = model->CreateSession();
        expect(session->CacheBytes() == reference.cacheBytes && session->ContextLength() == 0,
               "Hybrid session reservation does not match its allocation");
    }
    expect(model->ActiveSessions() == sessions.size() &&
               runtime->Memory().currentBytes == resident + sessions.size() * reference.cacheBytes,
           "Session creation duplicated weights/workspace or omitted recurrent device allocations");

    PrefillGate gate;
    Scheduler scheduler(scheduler_options(4, 32, 128));
    ReleaseGateOnExit releaseGate(gate);
    std::vector<std::shared_ptr<ScheduledRequest>> handles;
    handles.push_back(scheduler.Submit(sessions[0], cases[0].prompt, cases[0].initialOptions));
    expect(gate.AwaitEntry(), "First Hybrid prefill did not enter its workspace-growth gate");
    for (size_t i = 1; i < sessions.size(); ++i) {
        handles.push_back(scheduler.Submit(sessions[i], cases[i].prompt, cases[i].initialOptions));
    }
    const auto queued = scheduler.Snapshot();
    expect(queued.outstandingRequests == 4 && queued.activeRequests == 1 && queued.queuedRequests == 3,
           "Fixture did not enqueue all four Hybrid sessions before resuming the first step");
    gate.Release();
    const auto initial = drain_all(handles, "Four initial Hybrid streams");
    for (size_t i = 0; i < sessions.size(); ++i) {
        expect_completed(initial[i], sessions[i], cases[i].prompt, cases[i].initialOptions, 0,
                         "Initial Hybrid session " + std::to_string(i));
    }
    handles.clear();
    for (size_t i = 0; i < sessions.size(); ++i) {
        handles.push_back(scheduler.Submit(sessions[i], cases[i].continuation, cases[i].continuationOptions));
    }
    const auto appended = drain_all(handles, "Four appended Hybrid streams");
    for (size_t i = 0; i < sessions.size(); ++i) {
        expect_completed(appended[i], sessions[i], cases[i].continuation, cases[i].continuationOptions,
                         static_cast<int>(cases[i].prompt.size()) + kGenerated,
                         "Appended Hybrid session " + std::to_string(i));
    }
    scheduler.Shutdown();
    gate.Check();
    expect_idle(scheduler, "Four-session Hybrid scheduler");

    std::array<std::vector<float>, 4> probes;
    for (size_t i = 0; i < sessions.size(); ++i) {
        probes[i] = forward(sessions[i], cases[i].probe, "Scheduled Hybrid state probe");
    }
    for (size_t i = 0; i < sessions.size(); ++i) {
        const auto label = "Independent Hybrid session " + std::to_string(i);
        sessions[i]->Reset();
        reference.initial[i] = run_independent(sessions[i], cases[i].prompt, cases[i].initialOptions, label);
        if (initial[i].tokens != reference.initial[i]) {
            std::cerr << "[DIAG] " << label << " initial "
                      << describe_request(cases[i].prompt.size(), cases[i].initialOptions)
                      << " independent_context=" << sessions[i]->ContextLength() << std::endl;
        }
        expect_tokens(initial[i].tokens, reference.initial[i], label + " initial output");
        const int beforeAppend = sessions[i]->ContextLength();
        const auto continuation = run_independent(
            sessions[i], cases[i].continuation, cases[i].continuationOptions, label);
        if (appended[i].tokens != continuation) {
            std::cerr << "[DIAG] " << label << " initial "
                      << describe_request(cases[i].prompt.size(), cases[i].initialOptions)
                      << " tokens=" << describe_tokens(reference.initial[i]) << '\n'
                      << "[DIAG] " << label << " appended "
                      << describe_request(cases[i].continuation.size(), cases[i].continuationOptions)
                      << " original_context=" << beforeAppend
                      << " independent_context=" << sessions[i]->ContextLength()
                      << " scheduled_prompt_processed=" << appended[i].terminal.promptTokensProcessed
                      << " scheduled_generated=" << appended[i].terminal.generatedTokens << std::endl;
        }
        expect_tokens(appended[i].tokens, continuation, label + " appended output");
        expect_logits(probes[i], forward(sessions[i], cases[i].probe, label), label + " final state");
    }
    const auto modelResident = runtime->Memory().currentBytes - sessions.size() * reference.cacheBytes;
    for (auto& session : sessions) session.reset();
    expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == modelResident,
           "Completed Hybrid streams retained Full KV or recurrent device state");
    std::cout << "[INFO] Hybrid session device payload=" << reference.cacheBytes
              << ", host mirror/snapshot reservation=" << reference.hostBytes
              << ", max recurrent snapshots=" << kSnapshots << std::endl;
    return reference;
}

void test_interleaved_logits(const std::shared_ptr<LoadedModel>& model,
                             const Tokens& sourceA, const Tokens& sourceB) {
    std::cout << "[RUN] Full KV, recurrent state and convolution heads remain isolated across mixed shapes" << std::endl;
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const auto planA = make_plan(sourceA, {8, 19, 1, 1, 37, 1, 1, 5, 1, 1});
    const auto planB = make_plan(sourceB, {96, 3, 1, 17, 1, 1, 5, 1});
    LogitsTrace interleavedA, interleavedB;
    for (size_t i = 0; i < std::max(planA.size(), planB.size()); ++i) {
        if (i < planA.size()) interleavedA.push_back(forward(a, planA[i], "Interleaved A"));
        if (i < planB.size()) interleavedB.push_back(forward(b, planB[i], "Interleaved B"));
    }
    a->Reset();
    b->Reset();
    const auto referenceA = capture_plan(a, planA, "Independent mixed-shape A");
    const auto referenceB = capture_plan(b, planB, "Independent mixed-shape B");
    for (size_t i = 0; i < planA.size(); ++i) {
        expect_logits(interleavedA[i], referenceA[i], "Mixed-shape A step " + std::to_string(i));
    }
    for (size_t i = 0; i < planB.size(); ++i) {
        expect_logits(interleavedB[i], referenceB[i], "Mixed-shape B step " + std::to_string(i));
    }
}

void test_checkpoint_rollback(const std::shared_ptr<LoadedModel>& model,
                              const Tokens& sourceA, const Tokens& sourceB) {
    std::cout << "[RUN] Actual checkpoint lengths, decode-boundary snapshots, eviction and zero fallback" << std::endl;
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const auto peerPlan = make_plan(sourceB, {9, 5, 1, 7, 1, 1});
    const auto peerTrace = capture_plan(b, peerPlan, "Independent rollback peer");
    b->Reset();
    expect_logits(forward(b, peerPlan[0], "Peer before rollback"), peerTrace[0], "Initial peer state");

    const auto plan = make_plan(sourceA, {11, 6, 1, 1, 1, 1});
    const auto reference = capture_plan(a, plan, "Checkpoint 17 reference");
    const int restored = a->Truncate(20);
    expect(restored == 17 && a->ContextLength() == 17 && b->ContextLength() == 9,
           "Truncate must report checkpoint 17, not the requested unsaved length 20");
    replay_from(a, restored, plan, reference, "Replay from actual checkpoint 17");
    expect_logits(forward(b, peerPlan[1], "Peer after checkpoint 17"), peerTrace[1], "Peer after rollback");

    a->Reset();
    const auto boundaryPlan = make_plan(sourceA, {63, 1, 1, 1, 1});
    const auto boundaryTrace = capture_plan(a, boundaryPlan, "Decode checkpoint 64 reference");
    const int boundary = a->Truncate(65);
    expect(boundary == kSnapshotStep && a->ContextLength() == kSnapshotStep,
           "A single decode at the snapshot stride did not retain checkpoint 64");
    replay_from(a, boundary, boundaryPlan, boundaryTrace, "Replay from decode checkpoint 64");
    expect_logits(forward(b, peerPlan[2], "Peer after stride rollback"), peerTrace[2], "Stride rollback peer");

    a->Reset();
    const auto missingPlan = make_plan(sourceA, {6, 3, 1});
    const auto missingTrace = capture_plan(a, missingPlan, "Missing checkpoint reference");
    const int missing = a->Truncate(4);
    expect(missing == 0 && a->ContextLength() == 0,
           "A prefix preceding every snapshot must return the actual reset length zero");
    replay_from(a, missing, missingPlan, missingTrace, "Replay after missing-checkpoint reset");
    expect_logits(forward(b, peerPlan[3], "Peer after missing checkpoint"), peerTrace[3], "Missing checkpoint peer");

    a->Reset();
    // Six prefill boundaries leave only 21, 32, 45 and 60 at a bound of four.
    // The final single-token decode at 61 must not allocate another snapshot.
    const auto evictionPlan = make_plan(sourceA, {5, 7, 9, 11, 13, 15, 1});
    const auto evictionTrace = capture_plan(a, evictionPlan, "Eviction reference");
    const int retained = a->Truncate(22);
    expect(retained == 21 && a->ContextLength() == 21,
           "Snapshot eviction failed to retain the oldest of the four permitted checkpoints");
    replay_from(a, retained, evictionPlan, evictionTrace, "Replay from oldest retained checkpoint");
    const int evicted = a->Truncate(12);
    expect(evicted == 0 && a->ContextLength() == 0,
           "An evicted checkpoint remained available beyond the configured snapshot count");
    replay_from(a, evicted, evictionPlan, evictionTrace, "Replay after evicted-checkpoint reset");
    expect_logits(forward(b, peerPlan[4], "Peer after eviction"), peerTrace[4], "Snapshot eviction peer");

    const int length = a->ContextLength();
    expect(a->Truncate(length) == length, "Truncate to the current length must be a no-op");
    expect_error(ErrorCode::InvalidArgument, [&] { a->Truncate(-1); }, "Negative Hybrid truncate");
    expect_error(ErrorCode::InvalidArgument, [&] { a->Truncate(length + 1); }, "Oversized Hybrid truncate");
    expect(a->ContextLength() == length && !a->Busy(), "Invalid Hybrid truncate changed its session");
    a->Reset();
    expect(a->ContextLength() == 0, "Hybrid reset did not empty the context");
    expect_logits(forward(a, evictionPlan[0], "Reset Hybrid A"), evictionTrace[0], "Reset clears recurrent mirrors");
    expect_logits(forward(b, peerPlan[5], "Peer after final reset"), peerTrace[5], "Reset did not contaminate B");
}

void test_cancel_or_drop(const std::shared_ptr<DeviceRuntime>& runtime,
                         const std::shared_ptr<LoadedModel>& model,
                         const std::array<TestCase, 4>& cases, const References& reference,
                         bool drop, int firstChunk) {
    const std::string label = drop ? "Drop Hybrid A while B continues" : "Cancel Hybrid A while B continues";
    std::cout << "[RUN] " << label << std::endl;
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const std::weak_ptr<SessionState> weakA = a;
    const auto promptA = repeat_tokens(cases[0].prompt, static_cast<size_t>(firstChunk + 32));
    const auto optionsA = request_options(true, true, 441, firstChunk);
    PrefillGate gate;
    Scheduler scheduler(scheduler_options(2, 16, firstChunk));
    ReleaseGateOnExit releaseGate(gate);
    auto first = scheduler.Submit(a, promptA, optionsA);
    expect(gate.AwaitEntry(), label + ": first prefill did not reach its gate");
    auto second = scheduler.Submit(b, cases[1].prompt, cases[1].initialOptions);
    const int completedPrefix = a->ContextLength();
    expect(completedPrefix == firstChunk && a->Busy(), label + ": not at the expected safe step boundary");
    expect_error(ErrorCode::Busy, [&] { a->Reset(); }, label + " reset while busy");
    expect_error(ErrorCode::Busy, [&] { a->Truncate(0); }, label + " truncate while busy");
    const auto atGate = runtime->Memory();
    if (drop) first.reset();
    else first->Cancel();
    a.reset();
    expect(!weakA.expired() && model->ActiveSessions() == 2,
           label + ": a running step released its recurrent state too early");
    gate.Release();
    if (drop) {
        const auto observed = drain_all({second}, label);
        expect_completed(observed[0], b, cases[1].prompt, cases[1].initialOptions, 0, label + " B");
        expect_tokens(observed[0].tokens, reference.initial[1], label + " B output");
    } else {
        const auto observed = drain_all({first, second}, label);
        expect(observed[0].terminal.status == RequestStatus::Cancelled &&
                   observed[0].terminal.endReason == RequestEndReason::Cancelled &&
                   observed[0].terminal.promptTokensProcessed == static_cast<size_t>(completedPrefix) &&
                   observed[0].tokens.empty(), label + ": cancellation submitted more work or lost progress");
        expect_completed(observed[1], b, cases[1].prompt, cases[1].initialOptions, 0, label + " B");
        expect_tokens(observed[1].tokens, reference.initial[1], label + " B output");
    }
    scheduler.Shutdown();
    gate.Check();
    expect_idle(scheduler, label);
    expect(weakA.expired() && model->ActiveSessions() == 1 &&
               runtime->Memory().currentBytes == atGate.currentBytes - reference.cacheBytes,
           label + ": Full KV and recurrent device payload were not reclaimed exactly");
    const auto survivingProbe = forward(b, cases[1].probe, label + " surviving state");
    b->Reset();
    expect_tokens(run_independent(b, cases[1].prompt, cases[1].initialOptions, label + " reference"),
                  reference.initial[1], label + " independent output");
    expect_logits(survivingProbe, forward(b, cases[1].probe, label + " reference probe"),
                  label + " final recurrent/KV state");
}

void test_repeated_reclamation(const std::shared_ptr<DeviceRuntime>& runtime,
                               const std::shared_ptr<LoadedModel>& model,
                               const Tokens& source, size_t cacheBytes) {
    std::cout << "[RUN] Repeated Hybrid streams reclaim session caches and snapshot owners" << std::endl;
    expect(model->ActiveSessions() == 0, "Earlier Hybrid fixture retained a session");
    const size_t resident = runtime->Memory().currentBytes;
    for (int iteration = 0; iteration < 2; ++iteration) {
        std::array<std::shared_ptr<SessionState>, 4> sessions;
        std::array<std::weak_ptr<SessionState>, 4> weakSessions;
        for (size_t i = 0; i < sessions.size(); ++i) {
            sessions[i] = model->CreateSession();
            weakSessions[i] = sessions[i];
        }
        expect(model->ActiveSessions() == 4 && runtime->Memory().currentBytes == resident + 4 * cacheBytes,
               "Recreating Hybrid sessions duplicated shared resources");
        const auto before = runtime->Memory();
        expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, "Hybrid session-count limit");
        const auto after = runtime->Memory();
        expect(before.currentBytes == after.currentBytes && before.peakBytes == after.peakBytes,
               "Rejected fifth Hybrid session allocated device resources");
        Scheduler scheduler(scheduler_options(4, 4, 8));
        std::vector<std::shared_ptr<ScheduledRequest>> handles;
        for (size_t i = 0; i < sessions.size(); ++i) {
            handles.push_back(scheduler.Submit(sessions[i], repeat_tokens(source, 16),
                                              request_options(false, false, i, 8, 2)));
            sessions[i].reset();
        }
        const auto observed = drain_all(handles, "Repeated Hybrid reclamation");
        for (const auto& trace : observed) {
            expect(trace.terminal.status == RequestStatus::Completed && trace.tokens.size() == 2,
                   "Reclamation request failed: " + trace.terminal.error);
        }
        scheduler.Shutdown();
        expect_idle(scheduler, "Hybrid reclamation scheduler");
        for (const auto& weak : weakSessions) expect(weak.expired(), "Terminal handle retained Hybrid session state");
        expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == resident,
               "Repeated Hybrid teardown leaked tracked device allocations or active sessions");
    }
}

void test_owner_lifetime(std::shared_ptr<DeviceRuntime>& runtime,
                         std::shared_ptr<LoadedModel>& model,
                         const TestCase& fixture, const Tokens& expected, size_t emptyRuntimeBytes) {
    std::cout << "[RUN] A partially prefilling Hybrid request survives external model/runtime/session owners" << std::endl;
    auto session = model->CreateSession();
    const std::weak_ptr<SessionState> weakSession = session;
    auto request = RequestState::Create(session, fixture.prompt, fixture.initialOptions);
    const auto initial = request->Step();
    expect(initial.status == RequestStatus::Prefill && !initial.tokenId && !initial.becameTerminal,
           "Owner-lifetime fixture did not stop inside a multi-chunk prefill");
    session.reset();
    model.reset();
    runtime.reset();
    expect(!weakSession.expired(), "Dropping external owners destroyed a live Hybrid request");
    // A new public handle observes the still-live execution domain retained by
    // the request's session; it does not reload the original weights.
    auto observer = DeviceRuntime::Create();
    expect(observer->Memory().currentBytes > emptyRuntimeBytes, "Surviving request lost its device resources");
    Tokens result;
    int terminalEvents = 0;
    for (int i = 0; i < kMaxSteps; ++i) {
        const auto step = request->Step();
        if (step.tokenId) result.push_back(*step.tokenId);
        if (step.becameTerminal) ++terminalEvents;
        if (IsTerminal(step.status)) break;
    }
    expect(request->Snapshot().status == RequestStatus::Completed && terminalEvents == 1,
           "Surviving Hybrid request failed: " + request->Snapshot().error);
    expect_tokens(result, expected, "Surviving Hybrid request output");
    {
        const auto retainedSession = weakSession.lock();
        expect(retainedSession && !retainedSession->Busy() && retainedSession->ContextLength() ==
                   static_cast<int>(fixture.prompt.size() + result.size()),
               "Surviving Hybrid request did not finalize its independent state");
    }
    request.reset();
    expect(weakSession.expired() && observer->Memory().currentBytes == emptyRuntimeBytes,
           "Last Hybrid request destruction retained model/cache device allocations");
    runtime = std::move(observer);
}

size_t one_byte_below_two(size_t bytes) {
    expect(bytes > 1 && bytes <= std::numeric_limits<size_t>::max() / 2,
           "Invalid reservation size for byte-budget test");
    return bytes * 2 - 1;
}

void test_byte_budget_and_disabled_snapshots(const std::shared_ptr<DeviceRuntime>& runtime,
                                             const std::string& modelDirectory,
                                             const Tokens& source, const References& reference,
                                             size_t emptyRuntimeBytes, bool hostBudget) {
    const std::string label = hostBudget ? "Hybrid host snapshot byte budget" : "Hybrid device-state byte budget";
    std::cout << "[RUN] " << label << " rejects one byte below the second reservation" << std::endl;
    expect(runtime->Memory().currentBytes == emptyRuntimeBytes,
           label + ": previous model must be released before reloading");
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 4;
    options.maxRecurrentSnapshots = hostBudget ? kSnapshots : 0;
    if (hostBudget) options.maxSessionHostBytes = one_byte_below_two(reference.hostBytes);
    else options.maxSessionCacheBytes = one_byte_below_two(reference.cacheBytes);
    auto model = LoadedModel::LoadQwen35Nf4(runtime, modelDirectory, options);
    expect(model->SessionReservationBytes() == reference.cacheBytes,
           label + ": snapshot policy changed the Full KV/recurrent device payload");
    if (hostBudget) {
        expect(model->SessionHostReservationBytes() == reference.hostBytes,
               "Reload changed the configured host mirror/snapshot reservation");
    } else {
        expect(reference.hostBytes % (kSnapshots + 1) == 0 &&
                   model->SessionHostReservationBytes() == reference.hostBytes / (kSnapshots + 1),
               "Disabling snapshots must retain exactly one recurrent host mirror payload");
    }
    auto first = model->CreateSession();
    const auto before = runtime->Memory();
    expect_error(ErrorCode::ResourceLimit, [&] { model->CreateSession(); }, label);
    const auto after = runtime->Memory();
    expect(model->ActiveSessions() == 1 && model->ActiveSessions() < options.maxSessions &&
               first->ContextLength() == 0 && before.currentBytes == after.currentBytes &&
               before.peakBytes == after.peakBytes,
           label + ": rejection allocated device memory or was actually the session-count limit");

    const auto plan = make_plan(source, {17, 1, 1, 1});
    const auto trace = capture_plan(first, plan, label + " reference");
    const int restored = first->Truncate(17);
    expect(restored == (hostBudget ? 17 : 0),
           hostBudget ? "Host-limited model did not preserve its prefill snapshot"
                      : "maxRecurrentSnapshots=0 still retained a prefill snapshot");
    replay_from(first, restored, plan, trace, label + " replay");
    const auto resident = runtime->Memory().currentBytes - first->CacheBytes();
    first.reset();
    expect(model->ActiveSessions() == 0 && runtime->Memory().currentBytes == resident,
           label + ": destroying the first session did not reclaim its device payload");
    auto replacement = model->CreateSession();
    expect(model->ActiveSessions() == 1 && replacement->ContextLength() == 0 &&
               runtime->Memory().currentBytes == resident + reference.cacheBytes,
           label + ": session destruction failed to return its byte reservation");
    replacement.reset();
    model.reset();
    expect(runtime->Memory().currentBytes == emptyRuntimeBytes,
           label + ": model teardown retained tracked device resources");
}

void run_tests(const std::string& modelDirectory, const std::string& selectedCase) {
    ScopedLogLevel quiet(aila::LogLevel::Error);
    ScopedEnvironment globalKv("AILA_KV_QUANT", "0");
    ScopedEnvironment scopedKv("AILA_VLM_KV_QUANT", "0");
    ScopedKvOverride engineKv(false);
    ScopedSnapshotStep snapshotStep;
    Tokenizer tokenizer;
    expect(tokenizer.load(modelDirectory), "Could not load Hybrid model tokenizer");
    const auto cases = make_cases(tokenizer);
    const bool fullFixture = selectedCase == "all";
    std::cout << "[INFO] Fixture case: " << selectedCase << std::endl;
    const auto legacyPlan = make_plan(cases[0].prompt, {11, 3, 7, 16, 19, 1, 3, 1});
    LogitsTrace legacyTrace;
    if (fullFixture || selectedCase == "legacy-last-rows") {
        legacyTrace = capture_legacy_last_rows(modelDirectory, legacyPlan);
    }
    auto runtime = DeviceRuntime::Create();
    std::cout << "[INFO] Device: " << runtime->DeviceName() << std::endl;
    const size_t emptyRuntimeBytes = runtime->Memory().currentBytes;
    ModelOptions options;
    options.maxSequenceLength = kCapacity;
    options.maxSessions = 4;
    options.maxRecurrentSnapshots = kSnapshots;
    if (fullFixture) {
        std::cout << "[RUN] Effective quantized KV is rejected before shared Hybrid model allocation" << std::endl;
        quantized_matrix(runtime, [&] { LoadedModel::LoadQwen35Nf4(runtime, modelDirectory, options); },
                           "Hybrid model admission");
    }
    std::shared_ptr<LoadedModel> model;
    {
        std::cout << "[RUN] Scoped BF16 overrides global quantized KV for shared Hybrid model load" << std::endl;
        ScopedEnvironment global("AILA_KV_QUANT", "1");
        ScopedKvOverride override(true);
        model = LoadedModel::LoadQwen35Nf4(runtime, modelDirectory, options);
    }
    expect(model->MaxSequenceLength() == kCapacity && model->VocabSize() > 0,
           "Loaded Hybrid model has invalid metadata");
    auto finishSelectedCase = [&] {
        expect(model->ActiveSessions() == 0, "Selected Hybrid fixture retained session ownership");
        model.reset();
        expect(runtime->Memory().currentBytes == emptyRuntimeBytes,
               "Selected Hybrid fixture retained tracked device allocations");
    };
    if (selectedCase == "interleaved-logits") {
        test_interleaved_logits(model, cases[0].prompt, cases[1].prompt);
        finishSelectedCase();
        return;
    }
    if (selectedCase == "legacy-last-rows") {
        test_legacy_last_rows(model, legacyPlan, legacyTrace);
        finishSelectedCase();
        return;
    }
    if (fullFixture) test_quantized_session_admission(runtime, model);
    // Keep this before larger forwards: its first 96-token chunk grows the
    // loaded slot from 64 to 128 and provides the deterministic Scheduler gate.
    const auto reference = test_four_sessions(runtime, model, cases);
    if (selectedCase == "four-sessions") {
        finishSelectedCase();
        return;
    }
    test_legacy_last_rows(model, legacyPlan, legacyTrace);
    test_interleaved_logits(model, cases[0].prompt, cases[1].prompt);
    test_checkpoint_rollback(model, cases[0].prompt, cases[1].prompt);
    // Each case grows the slot beyond the previous maximum; peers use shapes
    // already warmed above so their continuation cannot confound cache bytes.
    test_cancel_or_drop(runtime, model, cases, reference, false, 192);
    test_cancel_or_drop(runtime, model, cases, reference, true, 256);
    test_repeated_reclamation(runtime, model, cases[0].prompt, reference.cacheBytes);
    expect(model->ActiveSessions() == 0, "Completed fixture retained Hybrid sessions");
    test_owner_lifetime(runtime, model, cases[1], reference.initial[1], emptyRuntimeBytes);
    test_byte_budget_and_disabled_snapshots(runtime, modelDirectory, cases[0].prompt,
                                             reference, emptyRuntimeBytes, false);
    test_byte_budget_and_disabled_snapshots(runtime, modelDirectory, cases[0].prompt,
                                             reference, emptyRuntimeBytes, true);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string modelDirectory;
        std::string selectedCase = "all";
        bool caseSpecified = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "Usage: Qwen35SessionTests [--model <Qwen3.5 Hybrid NF4 directory>]\n"
                             "                         [--case all|four-sessions|interleaved-logits|legacy-last-rows]\n"
                             "Fallback environment variable: AILA_QWEN35_TEST_MODEL\n"
                             "Real GPU fixture: four shared sessions, Scheduler, recurrent rollback,\n"
                             "resource budgets and BF16 scoped/global KV admission. Context 512.\n"
                             "Default: all. A selected case runs before other inference scenarios,\n"
                             "so each isolated repro starts from a fresh workspace.\n";
                return 0;
            }
            if (argument == "--case") {
                expect(i + 1 < argc && !caseSpecified, "Expected one optional --case <name>");
                selectedCase = argv[++i];
                caseSpecified = true;
                expect(selectedCase == "all" || selectedCase == "four-sessions" ||
                           selectedCase == "interleaved-logits" || selectedCase == "legacy-last-rows",
                       "Unknown fixture case: " + selectedCase);
                continue;
            }
            expect(argument == "--model" && i + 1 < argc && modelDirectory.empty(),
                   "Expected --model <directory> or --case <name>");
            modelDirectory = argv[++i];
            expect(!modelDirectory.empty(), "Model directory must not be empty");
        }
        if (modelDirectory.empty()) modelDirectory = environment_value("AILA_QWEN35_TEST_MODEL").value_or("");
        if (modelDirectory.empty()) {
            std::cout << "[SKIP] Specify --model <directory> or AILA_QWEN35_TEST_MODEL for the real Hybrid GPU fixture.\n";
            return 77;
        }
        expect(std::filesystem::is_directory(modelDirectory), "Model directory does not exist: " + modelDirectory);
        run_tests(modelDirectory, selectedCase);
        std::cout << "Qwen3.5 shared-session and scheduler tests passed." << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.5 shared-session tests failed: " << error.what() << std::endl;
        return 1;
    }
}
