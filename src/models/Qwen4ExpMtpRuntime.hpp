#pragma once
#include "Qwen4ExpMtpDraft.hpp"
#include "inference/Session.hpp"
#include "ops/GreedyOps.hpp"

// Called only under the owning DeviceRuntime execution mutex. Weights and
// rollback scratch are shared; each session owns its draft KV and last hidden.
class Qwen4ExpMtpRuntime {
public:
    struct State {
        std::unique_ptr<Qwen4ExpMtpDraft::State> draft;
        Tensor lastHidden;
        int length = 0;
        bool valid = true;
    };
    Qwen4ExpMtpRuntime(Context&, Qwen4ExpBnb4Backend&, ModelWeights&, int capacity);
    ~Qwen4ExpMtpRuntime();
    static size_t SessionBytes(int capacity);
    std::unique_ptr<State> CreateState();
    void AfterForward(State&, int start, const std::vector<int>& tokens);
    aila::inference::DecodeWindowResult ForwardWindow(Qwen4ExpBnb4Backend::Session&,
        State&, int pending, const aila::inference::DecodeWindowSelector&);
private:
    Context& ctx;
    Qwen4ExpBnb4Backend& target;
    int capacity;
    std::unique_ptr<Qwen4ExpMtpDraft> draft;
    Tensor hiddenRows, checkpoint, linearInputs, plePrefix, ids, greedyPartials, greedyResult;
    std::array<int, 48> convHeads{}, blocks{};
    std::array<int, 2> pleTokens{};
    int pleHead = 0, length = 0, noMiss = 0;
    uint64_t windows = 0, accepted = 0;
    std::array<ops::GreedyResult, 2> Select(int rows);
    void Capture(Qwen4ExpBnb4Backend::Session&);
    void Restore(Qwen4ExpBnb4Backend::Session&);
};
