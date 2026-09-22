#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "inference/Session.hpp"
#include "profile/Profiling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace aila::inference;
using Tokens = std::vector<int>;

constexpr int Capacity = 128;
constexpr size_t DeviceBudget = size_t{8} * 1024 * 1024 * 1024;
constexpr size_t PinnedBudget = size_t{16} * 1024 * 1024;
constexpr size_t RowCount = 4;

struct Arguments {
    std::string backend;
    std::string model;
    bool policyRejection = false;
};

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void ExpectError(ErrorCode code, Operation&& operation, const std::string& label) {
    try {
        operation();
    } catch (const Error& error) {
        Expect(error.Code() == code, label + ": wrong error code: " + error.what());
        return;
    }
    throw std::runtime_error(label + ": expected an inference Error");
}

void PrintTokens(const std::string& label, size_t row, const Tokens& tokens) {
    std::cout << "[TOKENS] run=" << label << " row=" << row << " values=";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << tokens[i];
    }
    std::cout << '\n';
}

void PrintMemory(const MemorySnapshot& memory, const std::string& label) {
    std::cout << "[MEMORY] label=" << label
              << " ready_bytes=" << memory.currentBytes
              << " peak_bytes=" << memory.peakBytes
              << " pinned_bytes=" << memory.pinnedHostBytes
              << " peak_pinned_bytes=" << memory.peakPinnedHostBytes
              << " device_allocations=" << memory.deviceAllocations
              << " pinned_allocations=" << memory.pinnedHostAllocations
              << " device_limit=" << memory.deviceLimitBytes
              << " pinned_limit=" << memory.pinnedHostLimitBytes << '\n';
}

void ExpectEmpty(const std::shared_ptr<DeviceRuntime>& runtime, const std::string& label) {
    const auto memory = runtime->Memory();
    PrintMemory(memory, label);
    Expect(memory.currentBytes == 0 && memory.pinnedHostBytes == 0 &&
               memory.deviceAllocations == 0 && memory.pinnedHostAllocations == 0,
           label + ": explicit USM allocations remain after model destruction");
    Expect(!runtime->FaultMessage(), label + ": device runtime is faulted");
}

std::shared_ptr<LoadedModel> Load(const Arguments& arguments, const std::shared_ptr<DeviceRuntime>& runtime) {
    ModelOptions options;
    options.maxSequenceLength = Capacity;
    options.maxSessions = RowCount;
    options.maxRecurrentSnapshots = 0;
    if (arguments.backend == "qwen3_nf4") return LoadedModel::LoadQwen3Nf4(runtime, arguments.model, options);
    if (arguments.backend == "qwen35_nf4") return LoadedModel::LoadQwen35Nf4(runtime, arguments.model, options);
    if (arguments.backend == "qwen35_bf16") return LoadedModel::LoadQwen35Bf16(runtime, arguments.model, options);
    throw std::runtime_error("Unknown backend: " + arguments.backend);
}

std::array<Tokens, RowCount> Prompts() {
    // Short, distinct low vocabulary IDs avoid any tokenizer/template policy.
    return {{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10, 11, 12, 13, 14}}};
}

RequestOptions Options(size_t row) {
    RequestOptions options;
    options.prefillChunkSize = 16; // Exactly one complete prefill per fixture prompt.
    auto& generation = options.generation;
    generation.max_new_tokens = static_cast<int>(6 + row);
    generation.do_sample = row != 0;
    generation.use_fixed_seed = true;
    generation.sampling_seed = 1001 + 97 * row;
    generation.temperature = row == 0 ? 1.0f : 0.75f + static_cast<float>(row) * 0.05f;
    generation.top_k = row == 0 ? 1 : 16 + static_cast<int>(row) * 8;
    generation.top_p = row == 0 ? 1.0f : 0.9f;
    if (row == 2) {
        options.forceHostSampling = true;
        generation.repetition_penalty = 1.15f;
        generation.presence_penalty = 0.2f;
        generation.frequency_penalty = 0.1f;
    } else if (row == 3) {
        // Penalties require host sampling even without the explicit force flag.
        generation.repetition_penalty = 1.05f;
        generation.frequency_penalty = -0.15f;
    }
    return options;
}

struct RunResult {
    std::array<Tokens, RowCount> tokens;
    std::array<size_t, RowCount> terminalCounts{};
    std::array<int, RowCount> contexts{};
    std::vector<std::vector<size_t>> submittedRows;
};

RunResult Run(const std::shared_ptr<LoadedModel>& model,
              const std::vector<size_t>& includedRows,
              const std::vector<size_t>& rowOrder,
              const std::string& label, bool cancelFirstRow = false) {
    Expect(model->ActiveSessions() == 0, label + ": a prior run retained sessions");
    Expect(includedRows.size() == rowOrder.size(), label + ": invalid fixture row order");
    auto sortedIncluded = includedRows, sortedOrder = rowOrder;
    std::sort(sortedIncluded.begin(), sortedIncluded.end());
    std::sort(sortedOrder.begin(), sortedOrder.end());
    Expect(sortedIncluded == sortedOrder &&
               std::adjacent_find(sortedIncluded.begin(), sortedIncluded.end()) == sortedIncluded.end() &&
               !sortedIncluded.empty() && sortedIncluded.back() < RowCount,
           label + ": fixture row IDs must be a permutation without duplicates");
    Expect(!cancelFirstRow || std::find(includedRows.begin(), includedRows.end(), size_t{0}) != includedRows.end(),
           label + ": cancellation row is not present");
    const auto prompts = Prompts();
    std::array<std::shared_ptr<SessionState>, RowCount> sessions;
    std::array<std::shared_ptr<RequestState>, RowCount> requests;
    RunResult output;
    for (const size_t row : includedRows) sessions[row] = model->CreateSession();
    const void* key = sessions[includedRows.front()]->DecodeBatchKey();
    Expect(key != nullptr, label + ": model configuration does not support decode batching");
    for (const size_t row : includedRows) {
        Expect(sessions[row]->DecodeBatchKey() == key, label + ": sessions did not retain the same model identity");
        requests[row] = RequestState::Create(sessions[row], prompts[row], Options(row));
    }
    const auto observe = [&](size_t row, const StepResult& step) {
        if (step.tokenId) {
            Expect(!IsTerminal(step.status) && *step.tokenId >= 0 && *step.tokenId < model->VocabSize(),
                   label + ": token row has invalid metadata or vocabulary ID");
            output.tokens[row].push_back(*step.tokenId);
        }
        if (step.becameTerminal) ++output.terminalCounts[row];
        if (step.status == RequestStatus::Failed) {
            throw std::runtime_error(label + ": row " + std::to_string(row) + " failed: " +
                                     requests[row]->Snapshot().error);
        }
    };

    // Prefill order stays fixed while decode row order changes in the replay.
    // This compares identical calculation shapes and per-request RNG histories.
    for (const size_t row : includedRows) {
        const auto step = requests[row]->Step();
        observe(row, step);
        Expect(step.status == RequestStatus::Decode && step.tokenId &&
                   requests[row]->Progress().promptTokensProcessed == prompts[row].size() &&
                   sessions[row]->ContextLength() == static_cast<int>(prompts[row].size()),
               label + ": short prefill did not complete in one independent step");
    }
    if (cancelFirstRow) requests[0]->Cancel();

    for (size_t turn = 0; turn < 32; ++turn) {
        std::vector<std::shared_ptr<RequestState>> batch;
        std::vector<size_t> selected;
        std::vector<size_t> submitted;
        for (const size_t row : rowOrder) {
            if (!IsTerminal(requests[row]->Progress().status)) {
                selected.push_back(row);
                batch.push_back(requests[row]);
                if (!(cancelFirstRow && turn == 0 && row == 0)) submitted.push_back(row);
            }
        }
        if (batch.empty()) break;
        output.submittedRows.push_back(submitted);
        std::cout << "[STEP] run=" << label << " turn=" << turn
                  << " selected_width=" << batch.size() << " forward_width=" << submitted.size() << " rows=";
        for (size_t i = 0; i < submitted.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << submitted[i];
        }
        std::cout << '\n';
        const auto steps = RequestState::StepDecodeBatch(batch);
        Expect(steps.size() == selected.size(), label + ": decode result row count changed");
        for (size_t i = 0; i < selected.size(); ++i) observe(selected[i], steps[i]);
    }

    for (const size_t row : includedRows) {
        const auto snapshot = requests[row]->Snapshot();
        const bool cancelled = cancelFirstRow && row == 0;
        Expect(snapshot.status == (cancelled ? RequestStatus::Cancelled : RequestStatus::Completed) &&
                   snapshot.endReason == (cancelled ? RequestEndReason::Cancelled : RequestEndReason::Length) &&
                   !snapshot.errorCode && output.terminalCounts[row] == 1 &&
                   snapshot.promptTokensProcessed == prompts[row].size() &&
                   snapshot.generatedIds == output.tokens[row] && !sessions[row]->Busy(),
               label + ": terminal, stream history, prompt accounting, or lease is inconsistent");
        output.contexts[row] = sessions[row]->ContextLength();
        Expect(output.tokens[row].size() == (cancelled ? size_t{1} : size_t{6} + row) &&
                   output.contexts[row] == static_cast<int>(prompts[row].size() +
                       (cancelled ? size_t{0} : output.tokens[row].size())),
               label + ": cancellation/finalization consumed the wrong context length");
        const auto repeated = requests[row]->Step();
        Expect(!repeated.becameTerminal && !repeated.tokenId && repeated.status == snapshot.status,
               label + ": a terminal request published a second event");
        PrintTokens(label, row, output.tokens[row]);
    }
    requests = {};
    sessions = {};
    Expect(model->ActiveSessions() == 0, label + ": terminated requests retained rebuilt sessions");
    std::cout << "[PASS] run=" << label << " sessions_released=" << includedRows.size() << '\n';
    return output;
}

std::vector<size_t> WidthChanges(const RunResult& result) {
    std::vector<size_t> changes;
    for (const auto& rows : result.submittedRows) {
        if (changes.empty() || changes.back() != rows.size()) changes.push_back(rows.size());
    }
    return changes;
}

int Functional(const Arguments& arguments) {
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "functional-initial");
    runtime->SetAllocationLimits(DeviceBudget, PinnedBudget);
    auto model = Load(arguments, runtime);
    Expect(model->VocabSize() > 14, "Fixture prompt IDs exceed the model vocabulary");
    std::cout << "[RUN] backend=" << arguments.backend << " capacity=" << Capacity
              << " max_sessions=" << RowCount << " model=" << arguments.model << '\n';

    const auto initial = Run(model, {0, 1, 2, 3}, {0, 1, 2, 3}, "mixed-order-0123");
    Expect(WidthChanges(initial) == std::vector<size_t>({4, 3, 2, 1}),
           "Mixed decode run did not exercise batch shrink 4->3->2->1");
    const auto reordered = Run(model, {0, 1, 2, 3}, {2, 0, 3, 1}, "mixed-order-2031");
    Expect(initial.tokens == reordered.tokens && initial.contexts == reordered.contexts &&
               WidthChanges(reordered) == std::vector<size_t>({4, 3, 2, 1}),
           "Reordering independent batch rows changed complete token sequences or final contexts");
    std::cout << "[PASS] row-order-replay complete_sequences=4\n";

    const auto cancellation = Run(model, {0, 1, 2, 3}, {0, 1, 2, 3}, "cancel-row-0", true);
    const auto oracle = Run(model, {1, 2, 3}, {1, 2, 3}, "three-row-oracle");
    Expect(cancellation.submittedRows == oracle.submittedRows &&
               WidthChanges(cancellation) == std::vector<size_t>({3, 2, 1}),
           "Cancellation replay did not preserve the independent three-row batch schedule");
    for (const size_t row : {size_t{1}, size_t{2}, size_t{3}}) {
        Expect(cancellation.tokens[row] == oracle.tokens[row] &&
                   cancellation.contexts[row] == oracle.contexts[row],
               "Cancelling row 0 changed an unaffected row's complete output or cache");
    }
    std::cout << "[PASS] cancelled-row-replay complete_sequences=3\n";
    Expect(!runtime->FaultMessage(), "Successful functional batches left the runtime faulted");
    PrintMemory(runtime->Memory(), "functional-ready-after-replays");
    model.reset();
    ExpectEmpty(runtime, "functional-final");
    std::cout << "[PASS] AilaBatchInferenceTests functional backend=" << arguments.backend << '\n';
    return 0;
}

RequestOptions PolicyGreedy(int generated) {
    RequestOptions options;
    options.prefillChunkSize = 16;
    options.generation.max_new_tokens = generated;
    options.generation.do_sample = false;
    options.generation.use_fixed_seed = true;
    options.generation.sampling_seed = 1001;
    options.generation.temperature = 1.0f;
    options.generation.top_k = 1;
    options.generation.top_p = 1.0f;
    return options;
}

bool SamePolicySnapshot(const RequestSnapshot& left, const RequestSnapshot& right) {
    return left.status == right.status && left.endReason == right.endReason &&
        left.promptTokensProcessed == right.promptTokensProcessed && left.generatedIds == right.generatedIds &&
        left.error == right.error && left.errorCode == right.errorCode;
}

bool SamePolicyMemory(const MemorySnapshot& left, const MemorySnapshot& right) {
    return left.currentBytes == right.currentBytes && left.peakBytes == right.peakBytes &&
        left.deviceAllocations == right.deviceAllocations && left.pinnedHostBytes == right.pinnedHostBytes &&
        left.peakPinnedHostBytes == right.peakPinnedHostBytes && left.pinnedHostAllocations == right.pinnedHostAllocations &&
        left.deviceLimitBytes == right.deviceLimitBytes && left.pinnedHostLimitBytes == right.pinnedHostLimitBytes;
}

void PrintPolicyMemory(const MemorySnapshot& memory, const char* stage) {
    std::cout << "[POLICY] {\"event\":\"memory\",\"stage\":\"" << stage
              << "\",\"device_bytes\":" << memory.currentBytes << ",\"peak_device_bytes\":" << memory.peakBytes
              << ",\"device_allocations\":" << memory.deviceAllocations << ",\"device_limit\":" << memory.deviceLimitBytes
              << ",\"pinned_bytes\":" << memory.pinnedHostBytes << ",\"peak_pinned_bytes\":" << memory.peakPinnedHostBytes
              << ",\"pinned_allocations\":" << memory.pinnedHostAllocations << ",\"pinned_limit\":" << memory.pinnedHostLimitBytes
              << "}\n";
}

Tokens FinishPolicyRequest(const std::shared_ptr<RequestState>& request, const std::shared_ptr<SessionState>& session,
                           Tokens observed, size_t expectedTokens, const std::string& label) {
    Expect(request->Snapshot().generatedIds == observed, label + ": starting token history differs from observed output");
    size_t terminalCount = 0;
    for (size_t turn = 0; turn < 32 && !IsTerminal(request->Progress().status); ++turn) {
        const auto step = request->Step();
        if (step.tokenId) {
            Expect(!IsTerminal(step.status) && *step.tokenId >= 0 && *step.tokenId < session->VocabSize(),
                   label + ": invalid token event");
            observed.push_back(*step.tokenId);
        }
        if (step.becameTerminal) ++terminalCount;
        Expect(step.status != RequestStatus::Failed, label + ": " + request->Snapshot().error);
    }
    const auto snapshot = request->Snapshot();
    Expect(snapshot.status == RequestStatus::Completed && snapshot.endReason == RequestEndReason::Length &&
               !snapshot.errorCode && terminalCount == 1 && snapshot.generatedIds == observed &&
               observed.size() == expectedTokens && !session->Busy(),
           label + ": incomplete sequence, duplicate terminal or retained lease");
    const auto repeated = request->Step();
    Expect(repeated.status == RequestStatus::Completed && !repeated.tokenId && !repeated.becameTerminal &&
               SamePolicySnapshot(snapshot, request->Snapshot()), label + ": terminal event was republished");
    return observed;
}

Tokens RunPolicyGreedy(const std::shared_ptr<SessionState>& session, const Tokens& prompt,
                       int generated, const std::string& label) {
    const int before = session->ContextLength();
    auto request = RequestState::Create(session, prompt, PolicyGreedy(generated));
    auto result = FinishPolicyRequest(request, session, {}, static_cast<size_t>(generated), label);
    Expect(request->Progress().promptTokensProcessed == prompt.size() &&
               session->ContextLength() == before + static_cast<int>(prompt.size()) + generated,
           label + ": prompt accounting or final context differs from the complete sequence");
    return result;
}

int PolicyRejection(const Arguments& arguments) {
    constexpr int OpenToken = 13;
    constexpr int CloseToken = 14;
    constexpr size_t SelectedRows = 3;
    const Tokens recoveryPrefix{9}, recoveryPrompt{5}, peerPrompt{10, 11}, peerContinuation{12};
    const std::string pinnedError = "Pinned host allocation budget exceeded";
    auto runtime = DeviceRuntime::Create();
    ExpectEmpty(runtime, "policy-initial");
    // Loading and greedy/logits-only paths use no pinned sampling buffer. The
    // first host sample will reject after a completed native batch forward.
    runtime->SetAllocationLimits(DeviceBudget, 1);
    auto model = Load(arguments, runtime);
    Expect(model->VocabSize() > CloseToken, "Policy fixture marker IDs exceed vocabulary");
    std::cout << "[POLICY] {\"event\":\"config\",\"scenario\":\"batch_host_sampling_limit_v1\",\"backend\":\""
              << arguments.backend << "\",\"selected_rows\":[0,1,2],\"host_row\":1,\"pinned_limit\":1,\"device_limit\":"
              << DeviceBudget << ",\"close_token\":" << CloseToken << ",\"context_capacity\":" << Capacity
              << ",\"max_snapshots\":0,\"vocabulary\":" << model->VocabSize() << "}\n";

    std::vector<float> recoveryLogits;
    Tokens recoveryTokens, peerInitialTokens, peerContinuationTokens;
    {
        // An independent B1 session supplies complete references. Destroy it
        // before creating the four fixture sessions; trace creation order then
        // identifies oracle, selected rows 0..2 and nonmember row 3 exactly.
        auto oracle = model->CreateSession();
        recoveryLogits = oracle->ForwardLogits(recoveryPrefix);
        Expect(recoveryLogits.size() == static_cast<size_t>(model->VocabSize()) &&
                   std::all_of(recoveryLogits.begin(), recoveryLogits.end(), [](float value) { return std::isfinite(value); }),
               "Policy B1 recovery logits are not complete and finite");
        recoveryTokens = RunPolicyGreedy(oracle, recoveryPrompt, 3, "policy-oracle-recovery");
        oracle->Reset();
        peerInitialTokens = RunPolicyGreedy(oracle, peerPrompt, 4, "policy-oracle-peer-initial");
        peerContinuationTokens = RunPolicyGreedy(oracle, peerContinuation, 2, "policy-oracle-peer-continuation");
    }
    Expect(model->ActiveSessions() == 0, "Policy oracle retained its session");
    PrintTokens("policy-oracle-recovery", 0, recoveryTokens);
    PrintTokens("policy-oracle-peer-initial", 3, peerInitialTokens);
    PrintTokens("policy-oracle-peer-continuation", 3, peerContinuationTokens);

    std::array<std::shared_ptr<SessionState>, RowCount> sessions;
    std::array<std::shared_ptr<RequestState>, RowCount> requests;
    for (auto& session : sessions) session = model->CreateSession();
    const void* key = sessions[0]->DecodeBatchKey();
    Expect(key != nullptr, "Policy fixture requires a native decode-batch-capable model");
    for (const auto& session : sessions) Expect(session->DecodeBatchKey() == key, "Policy session model identity changed");

    // Warm the exact row width and position without stochastic sampling. This
    // avoids an earlier token-buffer/activation rejection masking the target.
    for (size_t row = 0; row < SelectedRows; ++row) sessions[row]->ForwardLogits({static_cast<int>(row + 1)});
    const auto warm = SessionState::ForwardDecodeBatchLogits({sessions[0], sessions[1], sessions[2]}, {5, 6, 7});
    Expect(warm.size() == SelectedRows, "Policy warm batch did not return three logits rows");
    for (size_t row = 0; row < SelectedRows; ++row) {
        Expect(sessions[row]->ContextLength() == 2, "Policy warm batch did not consume each row");
        sessions[row]->Reset();
    }

    std::array<RequestSnapshot, SelectedRows> beforeRows;
    for (size_t row = 0; row < SelectedRows; ++row) {
        auto options = PolicyGreedy(row == 2 ? 1 : 3);
        options.generation.do_sample = row == 1;
        options.generation.temperature = 0.7f;
        options.generation.top_k = 4;
        options.forceHostSampling = row == 1;
        // Explicit synthetic marker metadata postpones the first sample until
        // decode. The fixture does not depend on the checkpoint tokenizer.
        aila::generation::ThinkingBudgetOptions budget;
        budget.maxTokens = 0;
        budget.openTokenId = OpenToken;
        budget.closeTokenId = CloseToken;
        budget.promptOpensThink = true;
        budget.closeTokenIds = {CloseToken};
        options.thinkingBudget = budget;
        requests[row] = RequestState::Create(sessions[row], {static_cast<int>(row + 1)}, options);
        const auto step = requests[row]->Step();
        beforeRows[row] = requests[row]->Snapshot();
        Expect(step.status == (row == 2 ? RequestStatus::Finalizing : RequestStatus::Decode) &&
                   step.tokenId == std::optional<int>{CloseToken} && !step.becameTerminal &&
                   beforeRows[row].generatedIds == Tokens{CloseToken} && beforeRows[row].promptTokensProcessed == 1 &&
                   sessions[row]->ContextLength() == 1 && sessions[row]->Busy(),
               "Policy prefill did not postpone sampling and retain its one-token prefix");
    }
    requests[3] = RequestState::Create(sessions[3], peerPrompt, PolicyGreedy(4));
    const auto peerFirst = requests[3]->Step();
    Expect(peerFirst.status == RequestStatus::Decode && peerFirst.tokenId && !peerFirst.becameTerminal,
           "Policy nonmember did not reach its independent decode boundary");
    const auto beforePeer = requests[3]->Snapshot();
    const int peerPosition = sessions[3]->ContextLength();
    Expect(peerPosition == 2 && beforePeer.generatedIds == Tokens{*peerFirst.tokenId} && sessions[3]->Busy(),
           "Policy nonmember initial accounting is incorrect");

    const auto ready = runtime->Memory();
    Expect(ready.currentBytes > 0 && ready.peakBytes <= DeviceBudget && ready.deviceLimitBytes == DeviceBudget &&
               ready.pinnedHostBytes == 0 && ready.peakPinnedHostBytes == 0 && ready.pinnedHostAllocations == 0 &&
               ready.pinnedHostLimitBytes == 1 && !runtime->FaultMessage(), "Policy ready state has an unexpected allocation or fault");
    PrintPolicyMemory(ready, "ready");
    std::cout << "[POLICY] {\"event\":\"ready\",\"contexts\":[1,1,1],\"generated\":[1,1,1],"
                 "\"statuses\":[\"decode\",\"decode\",\"finalizing\"],\"peer_context\":2}\n";

    const auto failed = RequestState::StepDecodeBatch({requests[0], requests[1], requests[2]});
    Expect(failed.size() == SelectedRows, "Policy failure omitted a submitted row");
    std::array<RequestSnapshot, SelectedRows> failedSnapshots;
    for (size_t row = 0; row < SelectedRows; ++row) {
        const auto& step = failed[row];
        failedSnapshots[row] = requests[row]->Snapshot();
        const auto& snapshot = failedSnapshots[row];
        Expect(step.status == RequestStatus::Failed && step.becameTerminal && !step.tokenId &&
                   snapshot.status == RequestStatus::Failed && snapshot.endReason == RequestEndReason::Failed &&
                   snapshot.errorCode == ErrorCode::ResourceLimit && snapshot.error == pinnedError &&
                   snapshot.generatedIds == beforeRows[row].generatedIds && snapshot.promptTokensProcessed == 1 &&
                   sessions[row]->ContextLength() == 0 && !sessions[row]->Busy(),
               "Post-forward pinned policy failure did not reset/fail every row without publishing its computed result: " + snapshot.error);
        std::cout << "[POLICY] {\"event\":\"row_failed\",\"row\":" << row
                  << ",\"error_code\":\"ResourceLimit\",\"error_message\":\"Pinned host allocation budget exceeded\","
                     "\"new_tokens\":0,\"generated_before\":1,\"generated_after\":1,\"terminal_count\":1,\"context\":0,\"busy\":false}\n";
    }
    Expect(SamePolicySnapshot(beforePeer, requests[3]->Snapshot()) && sessions[3]->ContextLength() == peerPosition &&
               sessions[3]->Busy() && !runtime->FaultMessage(), "Policy failure changed its nonmember or faulted the runtime");
    const auto after = runtime->Memory();
    Expect(SamePolicyMemory(ready, after), "Warmed post-forward policy rejection changed the USM ledger or limits");
    PrintPolicyMemory(after, "after_rejection");

    for (size_t row = 0; row < SelectedRows; ++row) {
        const auto repeated = requests[row]->Step();
        Expect(repeated.status == RequestStatus::Failed && !repeated.tokenId && !repeated.becameTerminal &&
                   SamePolicySnapshot(failedSnapshots[row], requests[row]->Snapshot()), "Policy row republished its failure");
    }
    const auto repeatedBatch = RequestState::StepDecodeBatch({requests[0], requests[1], requests[2]});
    Expect(repeatedBatch.size() == SelectedRows, "Terminal policy batch omitted a row");
    for (size_t row = 0; row < SelectedRows; ++row) {
        Expect(repeatedBatch[row].status == RequestStatus::Failed && !repeatedBatch[row].tokenId &&
                   !repeatedBatch[row].becameTerminal && SamePolicySnapshot(failedSnapshots[row], requests[row]->Snapshot()),
               "Terminal-only policy batch republished output or terminal state");
    }
    Expect(SamePolicyMemory(after, runtime->Memory()), "Terminal policy replay submitted new allocations");
    PrintPolicyMemory(runtime->Memory(), "after_replay");
    std::cout << "[POLICY] {\"event\":\"terminal_replay\",\"rows\":3,\"new_tokens\":0,\"new_terminals\":0}\n";

    const auto peerInitial = FinishPolicyRequest(requests[3], sessions[3], {*peerFirst.tokenId}, 4, "policy-peer-initial");
    Expect(peerInitial == peerInitialTokens && sessions[3]->ContextLength() == 6, "Policy nonmember initial sequence differs from B1 oracle");
    const auto peerContinued = RunPolicyGreedy(sessions[3], peerContinuation, 2, "policy-peer-continuation");
    Expect(peerContinued == peerContinuationTokens && sessions[3]->ContextLength() == 9,
           "Policy nonmember complete continuation differs from B1 oracle");
    PrintTokens("policy-peer-initial", 3, peerInitial);
    PrintTokens("policy-peer-continuation", 3, peerContinued);
    std::cout << "[POLICY] {\"event\":\"peer_recovered\",\"initial_tokens\":4,\"continuation_tokens\":2,"
                 "\"context\":9,\"oracle_equal\":true}\n";

    for (size_t row = 0; row < SelectedRows; ++row) {
        // There is no explicit Reset before this read: it must observe the
        // batch catch's actual KV/recurrent reset, rather than just length zero.
        const auto logits = sessions[row]->ForwardLogits(recoveryPrefix);
        Expect(logits == recoveryLogits && sessions[row]->ContextLength() == 1,
               "Policy member did not reproduce complete B1 logits after implicit reset");
        const auto tokens = RunPolicyGreedy(sessions[row], recoveryPrompt, 3, "policy-member-recovery");
        Expect(tokens == recoveryTokens && sessions[row]->ContextLength() == 5 && !runtime->FaultMessage(),
               "Policy member could not complete a new request equal to the B1 recovery oracle");
        PrintTokens("policy-member-recovery", row, tokens);
        std::cout << "[POLICY] {\"event\":\"member_recovered\",\"row\":" << row
                  << ",\"tokens\":3,\"context\":5,\"logits_equal\":true,\"oracle_equal\":true}\n";
    }
    requests = {};
    sessions = {};
    Expect(model->ActiveSessions() == 0, "Policy fixture retained selected or nonmember sessions");
    model.reset();
    ExpectEmpty(runtime, "policy-final");
    PrintPolicyMemory(runtime->Memory(), "final");
    std::cout << "[POLICY] {\"event\":\"cleanup\",\"active_sessions\":0,\"runtime_fault\":false}\n";
    std::cout << "[PASS] AilaBatchInferenceTests policy-rejection backend=" << arguments.backend << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::unitbuf;
    try {
        Arguments arguments;
        bool incomplete = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--policy-rejection") {
                arguments.policyRejection = true;
            } else if (option == "--backend" || option == "--model") {
                if (i + 1 == argc) {
                    incomplete = true;
                    break;
                }
                auto& destination = option == "--backend" ? arguments.backend : arguments.model;
                Expect(destination.empty(), "Duplicate argument: " + option);
                destination = argv[++i];
            } else {
                throw std::runtime_error("Unknown argument: " + option);
            }
        }
        if (incomplete || arguments.backend.empty() || arguments.model.empty()) {
            std::cout << "[SKIP] Usage: AilaBatchInferenceTests --backend qwen3_nf4|qwen35_nf4|qwen35_bf16"
                         " --model <directory> [--policy-rejection]\n";
            return 77;
        }
        Expect(arguments.backend == "qwen3_nf4" || arguments.backend == "qwen35_nf4" ||
                   arguments.backend == "qwen35_bf16", "Unsupported backend: " + arguments.backend);
        Expect(std::filesystem::is_directory(std::filesystem::path(arguments.model)),
               "Model directory does not exist: " + arguments.model);
        aila::set_log_level(aila::LogLevel::Info);
        return arguments.policyRejection ? PolicyRejection(arguments) : Functional(arguments);
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] AilaBatchInferenceTests: " << error.what() << '\n';
        return 1;
    }
}
