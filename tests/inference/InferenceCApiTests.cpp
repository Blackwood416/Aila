#include "aila_inference.h"
#include "runtime/InferenceClient.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_BUILD_ID
#error "C ABI GPU fixture requires the matching Worker identity"
#endif

namespace {
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
constexpr std::array<int32_t, 2> CloseTokens{1002, 1003};
constexpr std::array<int32_t, 4> Budgets{-1, 0, 2, 5};

void Expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template <typename T>
T Output() { T result{}; result.struct_size = sizeof(T); result.abi_version = AILA_INFERENCE_ABI_VERSION; return result; }
template <typename Function>
void Call(Function function) {
    auto error = Output<AilaInferenceError>();
    const auto result = function(&error);
    Expect(result == AILA_INFERENCE_OK && error.code == result,
           "C ABI error " + std::to_string(result) + ": " + error.message);
}
template <typename Predicate>
void Until(Predicate predicate, const char* message) {
    const auto deadline = Clock::now() + 10s;
    while (!predicate()) {
        Expect(Clock::now() < deadline, message);
        std::this_thread::sleep_for(2ms);
    }
}

template <auto Release>
class Handle final {
public:
    ~Handle() { Reset(); }
    Handle() = default;
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    uint64_t Get() const { return value; }
    uint64_t* Out() { Expect(value == 0, "Overwriting a live test handle"); return &value; }
    void Reset() noexcept { if (value) { Release(value, nullptr); value = 0; } }
private:
    uint64_t value = 0;
};
using Client = Handle<aila_inference_client_release>;
using Model = Handle<aila_inference_model_release>;
using Session = Handle<aila_inference_session_release>;
using Request = Handle<aila_inference_request_release>;

struct Parameters {
    std::filesystem::path worker;
    std::string model;
    uint32_t backend = AILA_INFERENCE_QWEN3_NF4;
    bool contextOperations = false;
};
struct Capture {
    std::vector<int32_t> tokens;
    uint64_t promptTokens = 0;
    uint64_t generatedTokens = 0;
    uint32_t contextLength = 0;
};
struct Reference {
    std::array<Capture, 4> first;
    std::array<Capture, 4> continuation;
    Capture peer;
};

std::vector<int32_t> Prompt(size_t index) {
    std::vector<int32_t> result(3 + index * 7);
    for (size_t i = 0; i < result.size(); ++i) result[i] = static_cast<int32_t>(11 + index * 17 + i);
    return result;
}
host::InferenceRequestOptions Options(size_t index, uint32_t count, bool thinking) {
    host::InferenceRequestOptions result;
    result.generation.maxNewTokens = count;
    result.generation.doSample = true;
    result.generation.useFixedSeed = true;
    result.generation.samplingSeed = 921 + index;
    result.generation.temperature = 0.7;
    result.generation.topK = 20;
    result.generation.topP = 0.9;
    result.generation.repetitionPenalty = 1.1;
    result.generation.presencePenalty = 0.2;
    result.generation.frequencyPenalty = 0.05;
    result.prefillChunkSize = 7;
    if (thinking && Budgets[index] >= 0) {
        wire::ThinkingBudgetOptions budget;
        budget.maxTokens = static_cast<uint32_t>(Budgets[index]);
        budget.openTokenId = 1001;
        budget.closeTokenId = CloseTokens[0];
        budget.promptOpensThink = true;
        budget.closeTokenIds.assign(CloseTokens.begin(), CloseTokens.end());
        result.thinkingBudget = std::move(budget);
    }
    return result;
}
AilaInferenceRequestOptions COptions(size_t index, uint32_t count, bool thinking) {
    AilaInferenceRequestOptions result{};
    Expect(aila_inference_default_request_options(&result, sizeof(result)) == AILA_INFERENCE_OK, "Request defaults failed");
    const auto source = Options(index, count, thinking);
    const auto& generation = source.generation;
    result.max_new_tokens = generation.maxNewTokens;
    result.do_sample = generation.doSample;
    result.use_fixed_seed = generation.useFixedSeed;
    result.sampling_seed = generation.samplingSeed;
    result.temperature = generation.temperature;
    result.top_k = generation.topK;
    result.top_p = generation.topP;
    result.repetition_penalty = generation.repetitionPenalty;
    result.presence_penalty = generation.presencePenalty;
    result.frequency_penalty = generation.frequencyPenalty;
    result.prefill_chunk_size = source.prefillChunkSize;
    if (source.thinkingBudget) {
        result.thinking_budget_tokens = static_cast<int32_t>(source.thinkingBudget->maxTokens);
        result.prompt_opens_think = 1;
        result.thinking_open_token_id = source.thinkingBudget->openTokenId;
        result.thinking_close_token_id = source.thinkingBudget->closeTokenId;
        result.thinking_close_token_ids = CloseTokens.data();
        result.thinking_close_token_count = static_cast<uint32_t>(CloseTokens.size());
    }
    return result;
}
Capture Drain(const std::shared_ptr<host::InferenceRequest>& request) {
    Capture result;
    const auto deadline = Clock::now() + 120s;
    while (Clock::now() < deadline) {
        const auto read = request->Read(100ms);
        if (read.status == host::InferenceReadStatus::Timeout) continue;
        Expect(read.status == host::InferenceReadStatus::Event && read.event.has_value(), "Internal reference stream closed early");
        const auto& event = *read.event;
        if (!event.terminal) {
            result.tokens.push_back(event.tokenId);
            Expect(event.sequence == result.tokens.size(), "Internal reference token sequence mismatch");
            continue;
        }
        Expect(!event.localFailure && event.summary.status == wire::RequestStatus::Completed &&
               event.summary.endReason == wire::EndReason::Length && !event.summary.errorCode,
               "Internal reference did not complete normally: " + event.summary.message);
        Expect(event.sequence == result.tokens.size() + 1 && event.summary.generatedTokens == result.tokens.size(),
               "Internal reference terminal count mismatch");
        result.promptTokens = event.summary.promptTokensProcessed;
        result.generatedTokens = event.summary.generatedTokens;
        Expect(request->Read(0ms).status == host::InferenceReadStatus::Closed, "Reference terminal repeated");
        request->Release();
        return result;
    }
    throw std::runtime_error("Internal reference read timed out");
}
Capture Drain(uint64_t request, bool cancelled = false) {
    Capture result;
    const auto deadline = Clock::now() + 120s;
    while (Clock::now() < deadline) {
        auto event = Output<AilaInferenceEvent>();
        auto error = Output<AilaInferenceError>();
        const auto status = aila_inference_request_read(request, 100, &event, &error);
        if (status == AILA_INFERENCE_TIMEOUT) continue;
        Expect(status == AILA_INFERENCE_OK, "C ABI read failed: " + std::string(error.message));
        if (event.kind == AILA_INFERENCE_EVENT_TOKEN) {
            result.tokens.push_back(event.token_id);
            Expect(event.sequence == result.tokens.size(), "C ABI token sequence mismatch");
            continue;
        }
        Expect(event.kind == AILA_INFERENCE_EVENT_TERMINAL && !event.local_failure, "Missing C ABI wire terminal");
        Expect(event.progress.status == (cancelled ? AILA_INFERENCE_CANCELLED : AILA_INFERENCE_COMPLETED) &&
               event.progress.end_reason == (cancelled ? AILA_INFERENCE_END_CANCELLED : AILA_INFERENCE_END_LENGTH) &&
               event.progress.remote_error == AILA_INFERENCE_REMOTE_NONE, "Unexpected C ABI terminal: " + std::string(event.progress.message));
        Expect(event.sequence == result.tokens.size() + 1 && event.progress.generated_tokens == result.tokens.size() &&
               event.delivered_tokens == result.tokens.size() && event.discarded_published_tokens == 0,
               "C ABI terminal count mismatch");
        result.promptTokens = event.progress.prompt_tokens_processed;
        result.generatedTokens = event.progress.generated_tokens;
        Expect(aila_inference_request_read(request, 0, &event, &error) == AILA_INFERENCE_END_OF_STREAM,
               "C ABI terminal repeated");
        return result;
    }
    throw std::runtime_error("C ABI read timed out");
}
uint32_t ContextLength(uint64_t session) {
    auto info = Output<AilaInferenceSessionInfo>();
    Call([&](auto* error) { return aila_inference_session_query(session, 0, &info, error); });
    Expect(info.has_context_length && !info.busy, "C ABI Session did not become idle");
    return info.context_length;
}
void Compare(Capture actual, const Capture& reference, uint64_t session) {
    actual.contextLength = ContextLength(session);
    Expect(actual.tokens == reference.tokens && actual.promptTokens == reference.promptTokens &&
           actual.generatedTokens == reference.generatedTokens && actual.contextLength == reference.contextLength,
           "C ABI tokens, usage or cache length differs from the independent internal-client run");
}
void Submit(uint64_t session, const std::vector<int32_t>& prompt, const AilaInferenceRequestOptions& options, Request& request) {
    Call([&](auto* error) {
        return aila_inference_request_submit(session, prompt.data(), static_cast<uint32_t>(prompt.size()),
                                             &options, 0, request.Out(), error);
    });
}

Reference CaptureReference(const Parameters& parameters) {
    auto client = host::InferenceClient::Start(parameters.worker.parent_path(), parameters.worker, AILA_BUILD_ID);
    struct CloseOnFailure {
        std::shared_ptr<host::InferenceClient> client;
        ~CloseOnFailure() { if (client) { try { client->Shutdown(15s); } catch (...) {} } }
    } cleanup{client};
    host::InferenceModelOptions settings;
    settings.maxSequenceLength = 512;
    settings.allocationLimits = wire::AllocationLimits{8ULL << 30, 16ULL << 20, 256ULL << 20};
    std::shared_ptr<host::InferenceModel> model;
    if (parameters.backend == AILA_INFERENCE_QWEN3_NF4) model = client->LoadQwen3(parameters.model, settings, 120s);
    else if (parameters.backend == AILA_INFERENCE_QWEN35_NF4) model = client->LoadQwen35(parameters.model, settings, 120s);
    else model = client->LoadQwen35Bf16(parameters.model, settings, 120s);
    Reference reference;
    for (size_t index = 0; index < 4; ++index) {
        const auto session = model->CreateSession();
        reference.first[index] = Drain(session->Submit(Prompt(index), Options(index, 16 + static_cast<uint32_t>(index), true)));
        reference.first[index].contextLength = session->Query().contextLength.value();
        reference.continuation[index] = Drain(session->Submit({71, 72}, Options(index, 4, false)));
        reference.continuation[index].contextLength = session->Query().contextLength.value();
        if (index == 1) {
            reference.peer = Drain(session->Submit({81, 82}, Options(index, 8, false)));
            reference.peer.contextLength = session->Query().contextLength.value();
        }
        session->Release();
    }
    model->Release();
    model.reset();
    client->Shutdown(15s); // Wait for this Worker to exit before starting the C ABI Worker.
    cleanup.client.reset();
    client.reset();
    std::cout << "[PASS] independent internal-client reference and Worker shutdown\n" << std::flush;
    return reference;
}

void VerifyCandidate(const Parameters& parameters, const Reference& reference) {
    Client client;
    const auto runtime = parameters.worker.parent_path().u8string();
    const auto executable = parameters.worker.u8string();
    Call([&](auto* error) { return aila_inference_client_start(runtime.c_str(), executable.c_str(), nullptr, client.Out(), error); });
    try {
        Model model;
        AilaInferenceModelOptions settings{};
        Expect(aila_inference_default_model_options(&settings, sizeof(settings)) == AILA_INFERENCE_OK, "Model defaults failed");
        settings.max_sequence_length = 512;
        settings.allocation_limits_enabled = 1;
        settings.max_device_bytes = 8ULL << 30;
        settings.max_pinned_host_bytes = 16ULL << 20;
        settings.max_session_host_bytes = 256ULL << 20;
        Call([&](auto* error) {
            return aila_inference_model_load(client.Get(), parameters.backend, parameters.model.c_str(), &settings, 120000,
                                             model.Out(), error);
        });
        auto info = Output<AilaInferenceModelInfo>();
        Call([&](auto* error) { return aila_inference_model_info(model.Get(), &info, error); });
        Expect(info.backend == parameters.backend && info.vocab_size > 1003 && info.has_memory_info &&
               info.resident_device_bytes > 0 && info.max_device_bytes == settings.max_device_bytes,
               "C ABI model/memory metadata changed");
        std::array<Session, 4> sessions;
        std::array<Request, 4> requests;
        for (size_t index = 0; index < 4; ++index) {
            Call([&](auto* error) { return aila_inference_session_create(model.Get(), 0, sessions[index].Out(), error); });
            Submit(sessions[index].Get(), Prompt(index), COptions(index, 16 + static_cast<uint32_t>(index), true), requests[index]);
        }
        std::array<std::future<Capture>, 4> captures;
        for (size_t index = 0; index < 4; ++index)
            captures[index] = std::async(std::launch::async, [&, index] { return Drain(requests[index].Get()); });
        for (size_t index = 0; index < 4; ++index) {
            Compare(captures[index].get(), reference.first[index], sessions[index].Get());
            requests[index].Reset();
        }
        std::cout << "[PASS] C ABI four concurrent fixed-seed requests with independent thinking budgets\n" << std::flush;
        for (size_t index = 0; index < 4; ++index)
            Submit(sessions[index].Get(), {71, 72}, COptions(index, 4, false), requests[index]);
        for (size_t index = 0; index < 4; ++index)
            captures[index] = std::async(std::launch::async, [&, index] { return Drain(requests[index].Get()); });
        for (size_t index = 0; index < 4; ++index) {
            Compare(captures[index].get(), reference.continuation[index], sessions[index].Get());
            requests[index].Reset();
        }
        std::cout << "[PASS] C ABI cache continuation, exact tokens and usage\n" << std::flush;
        if (parameters.contextOperations) {
            for (bool reset : {true, false}) {
                for (size_t index = 0; index < 4; ++index) {
                    auto info = Output<AilaInferenceSessionInfo>();
                    Call([&](auto* error) {
                        return reset ? aila_inference_session_reset(sessions[index].Get(), 0, &info, error)
                                     : aila_inference_session_truncate(sessions[index].Get(), 0, 0, &info, error);
                    });
                    Expect(!info.busy && info.has_context_length && info.context_length == 0,
                           "C ABI reset/truncate-to-zero retained context");
                    Submit(sessions[index].Get(), Prompt(index), COptions(index, 16 + static_cast<uint32_t>(index), true), requests[index]);
                }
                for (size_t index = 0; index < 4; ++index)
                    captures[index] = std::async(std::launch::async, [&, index] { return Drain(requests[index].Get()); });
                for (size_t index = 0; index < 4; ++index) {
                    Compare(captures[index].get(), reference.first[index], sessions[index].Get());
                    requests[index].Reset();
                }
            }
            for (size_t index = 0; index < 4; ++index)
                Submit(sessions[index].Get(), {71, 72}, COptions(index, 4, false), requests[index]);
            for (size_t index = 0; index < 4; ++index)
                captures[index] = std::async(std::launch::async, [&, index] { return Drain(requests[index].Get()); });
            for (size_t index = 0; index < 4; ++index) {
                Compare(captures[index].get(), reference.continuation[index], sessions[index].Get());
                requests[index].Reset();
            }
            std::cout << "[PASS] C ABI reset, truncate-to-zero, thinking/seed replay and continuation\n" << std::flush;
        }
        Submit(sessions[0].Get(), {81, 82}, COptions(0, 256, false), requests[0]);
        if (parameters.contextOperations) {
            auto info = Output<AilaInferenceSessionInfo>();
            auto error = Output<AilaInferenceError>();
            Expect(aila_inference_session_reset(sessions[0].Get(), 0, &info, &error) == AILA_INFERENCE_BUSY &&
                   aila_inference_session_truncate(sessions[0].Get(), 0, 0, &info, &error) == AILA_INFERENCE_BUSY,
                   "C ABI allowed mutation of an active GPU Session");
        }
        Submit(sessions[1].Get(), {81, 82}, COptions(1, 8, false), requests[1]);
        uint32_t cancellation = 0;
        Call([&](auto* error) { return aila_inference_request_cancel(requests[0].Get(), 0, &cancellation, error); });
        Expect(cancellation != 0, "Cancellation did not reach an active request");
        (void)Drain(requests[0].Get(), true);
        Compare(Drain(requests[1].Get()), reference.peer, sessions[1].Get());
        requests[0].Reset();
        requests[1].Reset();
        std::cout << "[PASS] C ABI cancellation leaves peer output exact\n" << std::flush;
        for (auto& session : sessions) session.Reset();
        Until([&] {
            auto snapshot = Output<AilaInferenceClientInfo>();
            Call([&](auto* error) { return aila_inference_client_snapshot(client.Get(), &snapshot, error); });
            return !snapshot.outstanding_requests && !snapshot.retained_request_handles &&
                   !snapshot.reserved_tokens && !snapshot.transport_bytes;
        }, "C ABI did not return request reservations");
        Session child;
        // Session release is asynchronous; its slot is available only after
        // the Worker lifecycle executor has finished reaping it.
        Until([&] {
            auto error = Output<AilaInferenceError>();
            const auto result = aila_inference_session_create(model.Get(), 0, child.Out(), &error);
            if (result == AILA_INFERENCE_RESOURCE_LIMIT) return false;
            Expect(result == AILA_INFERENCE_OK, "Session reuse failed: " + std::string(error.message));
            return true;
        }, "Released Session slots did not become reusable");
        model.Reset();
        if (parameters.contextOperations) {
            auto info = Output<AilaInferenceSessionInfo>();
            Call([&](auto* error) { return aila_inference_session_reset(child.Get(), 0, &info, error); });
            Expect(info.has_context_length && info.context_length == 0, "Released parent prevented C ABI reset");
        }
        Request descendant;
        Submit(child.Get(), Prompt(0), COptions(0, 16, true), descendant);
        child.Reset();
        const auto actual = Drain(descendant.Get());
        Expect(actual.tokens == reference.first[0].tokens && actual.generatedTokens == reference.first[0].generatedTokens,
               "Child request changed after parent releases");
        descendant.Reset();
        Call([&](auto* error) { return aila_inference_client_shutdown(client.Get(), 15000, error); });
        std::cout << "[PASS] C ABI slot reuse, parent release and orderly Worker shutdown\n"
                  << "backend=" << parameters.backend << " resident_device_bytes=" << info.resident_device_bytes
                  << " context=512 sessions=4 exact_tokens=true\n" << std::flush;
    } catch (...) {
        aila_inference_client_shutdown(client.Get(), 15000, nullptr);
        throw;
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        Parameters parameters;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--session-context") {
                Expect(!parameters.contextOperations, "Repeated --session-context");
                parameters.contextOperations = true;
                continue;
            }
            if (i + 1 >= argc) throw std::invalid_argument("Missing argument value");
            if (argument == "--worker") parameters.worker = std::filesystem::u8path(argv[++i]);
            else if (argument == "--model") parameters.model = argv[++i];
            else if (argument == "--backend") {
                const std::string backend = argv[++i];
                if (backend == "qwen3_nf4") parameters.backend = AILA_INFERENCE_QWEN3_NF4;
                else if (backend == "qwen35_nf4") parameters.backend = AILA_INFERENCE_QWEN35_NF4;
                else if (backend == "qwen35_bf16") parameters.backend = AILA_INFERENCE_QWEN35_BF16;
                else throw std::invalid_argument("Unknown backend");
            } else throw std::invalid_argument("Unknown argument");
        }
        if (parameters.worker.empty() || parameters.model.empty()) {
            std::cout << "SKIP: --worker and --model are required; run under the machine GPU guard\n";
            return 77;
        }
        parameters.worker = std::filesystem::absolute(parameters.worker).lexically_normal();
        const auto reference = CaptureReference(parameters);
        VerifyCandidate(parameters, reference);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL C ABI GPU fixture: " << error.what() << '\n';
        return 1;
    }
}
