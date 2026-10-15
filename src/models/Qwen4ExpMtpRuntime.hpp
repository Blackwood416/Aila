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
    // Load-time policy for the draft window, clamped to the model's sequence
    // budget. AILA_Q4E_MTP_MAX_CONTEXT (default 2048) keeps today's bound;
    // raising it only widens which sessions may draft, never the T1 path.
    static int EffectiveCapacity(int maxSequenceLength);
    // MTP service window in tokens. Decode windows are used only while the
    // session fits: AfterForward invalidates State past this length and the
    // caller's admission check must reject windows at Capacity()-1 or above,
    // so longer sessions silently take the plain T1 path.
    int Capacity() const noexcept { return capacity; }
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
