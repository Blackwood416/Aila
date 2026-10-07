#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "utils/Tokenizer.hpp"

#include <iostream>
#include <cstdlib>
#include <stdexcept>

namespace {
using namespace aila::inference;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void complete(const std::shared_ptr<RequestState>& request) {
    for (int i = 0; i < 32 && !IsTerminal(request->Progress().status); ++i) request->Step();
    const auto result = request->Snapshot();
    if (result.status != RequestStatus::Completed) throw std::runtime_error(result.error);
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "SKIP: pass Qwen4Exp NF4 model directory\n";
        return 77;
    }
    try {
        using namespace aila::inference;
        auto runtime = DeviceRuntime::Create();
        ModelOptions options;
        options.maxSequenceLength = 128;
        options.maxSessions = 3;
        options.maxRecurrentSnapshots = 0;
        const bool mtp = std::getenv("AILA_Q4E_SERVER_MTP") && std::atoi(std::getenv("AILA_Q4E_SERVER_MTP")) != 0;
        const size_t reservation = 121675776 + (mtp ? 128 * 2304 + 32 * 256 + 20480 : 0);
        // Exactly two sessions fit the device reservation budget.
        options.maxSessionCacheBytes = 2 * reservation;
        auto model = LoadedModel::LoadQwen4ExpNf4(runtime, argv[1], options);
        expect(model->SessionReservationBytes() == reservation, "Device reservation mismatch");
        expect(model->SessionHostReservationBytes() == 117669888, "Host reservation mismatch");
        const auto baseline = runtime->Memory();
        auto first = model->CreateSession();
        auto second = model->CreateSession();
        expect(first->CacheBytes() == model->SessionReservationBytes(), "Actual cache differs from reservation");
        expect(runtime->Memory().currentBytes == baseline.currentBytes + 2 * first->CacheBytes(),
               "Session allocations differ from reservation");
        expect(!first->DecodeBatchKey(), "Qwen4Exp must not join cross-session decode batch");
        bool limited = false;
        try { auto third = model->CreateSession(); }
        catch (const Error& error) { limited = error.Code() == ErrorCode::ResourceLimit; }
        expect(limited && model->ActiveSessions() == 2, "Session budget admission failed");

        Tokenizer tokenizer;
        expect(tokenizer.load(argv[1]), "Tokenizer failed");
        auto prompt = tokenizer.encode("The lighthouse overlooks the ocean. At sunrise");
        expect(!prompt.empty() && prompt.size() < 64, "Unexpected prompt size");
        const auto reference = first->ForwardLogits(prompt);
        expect(reference == second->ForwardLogits(prompt), "Interleaved prefill logits differ");
        const auto continuation = tokenizer.encode(" the keeper watches the waves.");
        const auto appended = first->ForwardLogits(continuation);
        expect(appended == second->ForwardLogits(continuation), "Interleaved continuation logits differ");
        expect(first->Truncate(1) == 0 && first->ContextLength() == 0, "Snapshot-free truncate must reset");
        second->Reset();
        expect(first->ForwardLogits(prompt) == reference, "Truncate replay differs");
        first->Reset();

        RequestOptions requestOptions;
        requestOptions.generation.do_sample = false;
        const int generatedCount = mtp ? 8 : 3;
        requestOptions.generation.max_new_tokens = generatedCount;
        requestOptions.generation.use_fixed_seed = true;
        auto requestA = RequestState::Create(first, prompt, requestOptions);
        auto referenceOptions = requestOptions;
        // Interleaving must preserve the same execution mode and private state.
        auto requestB = RequestState::Create(second, prompt, referenceOptions);
        for (int i = 0; i < 8; ++i) { requestA->Step(); requestB->Step(); }
        complete(requestA); complete(requestB);
        if (requestA->Snapshot().generatedIds != requestB->Snapshot().generatedIds) {
            std::cerr << "MTP:"; for (int t : requestA->Snapshot().generatedIds) std::cerr << ' ' << t;
            std::cerr << "\nT1:"; for (int t : requestB->Snapshot().generatedIds) std::cerr << ' ' << t;
            std::cerr << '\n';
        }
        expect(requestA->Snapshot().generatedIds == requestB->Snapshot().generatedIds,
               "Interleaved greedy requests differ");
        const auto initialGenerated = requestA->Snapshot().generatedIds;
        expect(first->ContextLength() == int(prompt.size()) + generatedCount &&
               second->ContextLength() == first->ContextLength(), "Pending token was not flushed exactly once");
        requestA.reset(); requestB.reset();
        auto followup = RequestState::Create(first, continuation, requestOptions);
        complete(followup);
        auto referenceFollowup = RequestState::Create(second, continuation, referenceOptions);
        complete(referenceFollowup);
        expect(followup->Snapshot().generatedIds == referenceFollowup->Snapshot().generatedIds, "Interleaved followup differs");
        expect(first->ContextLength() == int(prompt.size() + continuation.size()) + 2 * generatedCount,
               "Second turn cache length differs");
        followup.reset();
        referenceFollowup.reset();
        if (mtp) {
            second->Reset();
            auto t1Options = requestOptions; t1Options.forceHostSampling = true;
            auto t1 = RequestState::Create(second, prompt, t1Options);
            complete(t1);
            const auto t1Ids = t1->Snapshot().generatedIds;
            size_t differences = 0;
            for (size_t i = 0; i < initialGenerated.size(); ++i) {
                if (initialGenerated[i] != t1Ids[i]) {
                    ++differences;
                    std::cout << "MTP/T1 difference index=" << i << " mtp=" << initialGenerated[i] << " t1=" << t1Ids[i] << '\n';
                }
            }
            // B2 and B1 use different floating-point reduction paths. Report
            // this comparison separately; do not call experimental MTP lossless.
            std::cout << "MTP/T1 compared=" << t1Ids.size() << " differences=" << differences << '\n';
        }
        first->Reset(); second->Reset();
        // Sessions keep shared weights and the execution domain alive.
        model.reset(); runtime.reset();
        expect(first->ForwardLogits(prompt) == second->ForwardLogits(prompt), "Retained sessions differ");
        first.reset(); second.reset();
        runtime = DeviceRuntime::Create();
        expect(runtime->Memory().currentBytes == 0 && runtime->Memory().pinnedHostBytes == 0,
               "Shared model/session teardown leaked explicit USM");
        std::cout << "PASS Qwen4Exp shared " << (mtp ? "MTP isolation" : "T1")
                  << ": memory, interleave, append, reset, truncate, lifetime\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
