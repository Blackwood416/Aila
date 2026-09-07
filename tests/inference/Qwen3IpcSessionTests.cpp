#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "runtime/InferenceClient.hpp"
#include "profile/Profiling.hpp"
#include "utils/Tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef AILA_IPC_WORKER_PATH
#error "Qwen3 IPC tests require the actual Worker target path"
#endif
#ifndef AILA_BUILD_ID
#error "Qwen3 IPC tests require the Worker build identity"
#endif

namespace {

namespace core = aila::inference;
namespace host = aila::runtime;
namespace wire = aila::ipc::v2;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Tokens = std::vector<int32_t>;
constexpr int Capacity = 512;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Case {
    Tokens prompt;
    Tokens continuation;
    host::InferenceRequestOptions options;
    host::InferenceRequestOptions continuationOptions;
    Tokens expected;
    Tokens expectedContinuation;
};

core::RequestOptions DirectOptions(const host::InferenceRequestOptions& input) {
    core::RequestOptions result;
    result.prefillChunkSize = static_cast<int>(input.prefillChunkSize);
    result.stopTokenIds.assign(input.stopTokenIds.begin(), input.stopTokenIds.end());
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

std::vector<Case> References(const std::string& directory) {
    Tokenizer tokenizer;
    Expect(tokenizer.load(directory), "Could not load tokenizer");
    const std::array<std::string, 4> prompts{
        "List the first few prime numbers and explain why they are prime. /no_think",
        "用中文简短说明共享模型权重与独立会话缓存的关系。 /no_think",
        "A train leaves at noon and travels 60 kilometres in an hour. Explain its speed, then give a short example. /no_think",
        "Write a small C++ function that sums a vector of integers. Explain the time complexity and one possible overflow issue. /no_think"};
    std::vector<Case> cases;
    auto runtime = core::DeviceRuntime::Create();
    core::ModelOptions options;
    options.maxSequenceLength = Capacity;
    options.maxSessions = 4;
    auto model = core::LoadedModel::LoadQwen3Nf4(runtime, directory, options);
    for (size_t index = 0; index < prompts.size(); ++index) {
        Case test;
        const auto prompt = tokenizer.apply_chat_template("You are a helpful assistant.", prompts[index]);
        test.prompt.assign(prompt.begin(), prompt.end());
        const auto continuation = tokenizer.encode(" Continue with one more useful example.");
        test.continuation.assign(continuation.begin(), continuation.end());
        test.options.prefillChunkSize = index % 2 ? 32 : 16;
        test.options.generation.maxNewTokens = static_cast<uint32_t>(20 + index * 3);
        test.options.generation.doSample = index >= 2;
        test.options.generation.useFixedSeed = true;
        test.options.generation.samplingSeed = 987654321ull + index;
        test.options.generation.repetitionPenalty = index >= 2 ? 1.08 : 1.0;
        test.options.generation.presencePenalty = index >= 2 ? 0.1 : 0.0;
        test.options.generation.frequencyPenalty = index >= 2 ? 0.05 : 0.0;
        test.continuationOptions = test.options;
        test.continuationOptions.generation.maxNewTokens = 5;
        auto session = model->CreateSession();
        test.expected = Direct(session, test.prompt, test.options);
        test.expectedContinuation = Direct(session, test.continuation, test.continuationOptions);
        cases.push_back(std::move(test));
    }
    std::cout << "[PASS] direct greedy, seeded/penalized and continuation references\n";
    return cases; // All direct device owners are destroyed before Worker starts.
}

struct Output {
    Tokens tokens;
    host::InferenceReadEvent terminal;
    double firstTokenMs = 0;
};

std::vector<Output> Drain(const std::vector<std::shared_ptr<host::InferenceRequest>>& requests,
                          Clock::time_point start = Clock::now()) {
    std::vector<Output> output(requests.size());
    std::vector<bool> finished(requests.size());
    size_t remaining = requests.size();
    const auto deadline = Clock::now() + 120s;
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
        try { return model->CreateSession(); }
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
                      << " generated=" << output[index].tokens.size() << " caller_first_token_ms=" << output[index].firstTokenMs << '\n';
        }
        requests.clear();
        for (size_t index = 0; index < count; ++index)
            requests.push_back(sessions[index]->Submit(cases[index].continuation, cases[index].continuationOptions));
        const auto continued = Drain(requests);
        for (size_t index = 0; index < count; ++index)
            Same(continued[index].tokens, cases[index].expectedContinuation, "IPC continuation");
        requests.clear(); sessions.clear();
        WaitReclaimed(client);
        std::cout << "[PASS] " << count << " IPC sessions and continuation\n";
    }
    Expect(observedOverlap, "real Worker did not show overlapping accepted execution");
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

std::shared_ptr<host::InferenceClient> StartWorker() {
    const std::filesystem::path executable(AILA_IPC_WORKER_PATH);
    host::InferenceClientOptions options;
    options.handshakeTimeout = 10s;
    options.commandTimeout = 120s;
    options.shutdownTimeout = 20s;
    return host::InferenceClient::Start(executable.parent_path(), executable, AILA_BUILD_ID, options);
}

void Run(const std::string& directory) {
    aila::set_log_level(aila::LogLevel::Warning);
    const auto cases = References(directory);
    auto client = StartWorker();
    const auto firstEpoch = client->Snapshot().epoch;
    host::InferenceModelOptions modelOptions;
    modelOptions.maxSequenceLength = Capacity;
    modelOptions.maxSessions = 4;
    auto model = client->LoadQwen3(directory, modelOptions);
    TestConcurrency(client, model, cases);
    TestCancelAndBackpressure(client, model, cases);
    TestParentReleaseAndShutdown(client, model, cases);
    auto restarted = StartWorker();
    Expect(restarted->Snapshot().epoch != firstEpoch, "restarted Worker reused epoch");
    auto newModel = restarted->LoadQwen3(directory, modelOptions);
    auto session = newModel->CreateSession();
    auto request = session->Submit(cases[0].prompt, cases[0].options);
    Same(Drain({request})[0].tokens, cases[0].expected, "new epoch inference");
    request.reset(); session.reset(); newModel.reset();
    WaitReclaimed(restarted);
    restarted->Shutdown(20s);
    std::cout << "[PASS] fresh Worker epoch, reload and token parity\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string directory;
        for (int index = 1; index < argc; ++index) {
            if (std::string(argv[index]) != "--model" || index + 1 >= argc || !directory.empty())
                throw std::invalid_argument("Usage: AilaQwen3IpcSessionTests --model <NF4 model directory>");
            directory = argv[++index];
        }
        if (directory.empty()) {
            char* value = nullptr;
            size_t length = 0;
            if (_dupenv_s(&value, &length, "AILA_QWEN3_TEST_MODEL") == 0 && value) directory = value;
            std::free(value);
        }
        if (directory.empty()) {
            std::cout << "[SKIP] Set AILA_QWEN3_TEST_MODEL or pass --model.\n";
            return 77;
        }
        directory = std::filesystem::absolute(directory).lexically_normal().string();
        Expect(std::filesystem::is_directory(directory), "Model directory does not exist");
        Run(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
