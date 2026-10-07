#pragma once

// Checkpoint-backed Qwen4Exp MTP (multi-token prediction) draft head.
// Semantics source: vLLM qwen4_exp/nvidia/mtp.py, residual_linear_shared,
// scheme A. Target and draft share the embedding table and LM head, but never
// KV state. The shared instance owns the draft weights, projections, expert
// cache and reusable decode scratch; each State owns only the draft KV and
// indexer tensors.

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include "../ops/Bnb4BitLinear.hpp"
#include "../ops/Ops.hpp"
#include "../utils/SafeTensors.hpp"
#include "Qwen4ExpBnb4Backend.hpp"
#include "Qwen4ExpExpertCache.hpp"

#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>

class Qwen4ExpMtpDraft {
public:
    struct State {
        Qwen4ExpLayerCache cache;
    };

    Qwen4ExpMtpDraft(Context& ctx, Qwen4ExpBnb4Backend& target, ModelWeights& weights);
    ~Qwen4ExpMtpDraft();

    std::unique_ptr<State> CreateState(int capacity);

    // Input hidden is [4,H] produced by the target (or a prior draft step)
    // BEFORE its final mixer, at the hidden's own position; the input token ID
    // is shifted by one. The caller supplies the visible position every step,
    // so an attention-only suffix overwrite implements draft rollback: the
    // visible length is always position+1, never the state capacity. Writes the
    // draft logits into the target's logits buffer unless kvOnly is set, in
    // which case it returns after the persistent KV/indexer rows are updated.
    void Step(State& state, const sycl::ext::oneapi::bfloat16* hidden, int token,
              int position, bool kvOnly = false);

private:
    static void Require(bool ok, const std::string& message);
    Tensor Allocate(int rows, int cols,
                    dnnl::memory::data_type type = dnnl::memory::data_type::bf16);
    Tensor* Norm(const std::string& name, bool addOneBf16 = false);
    void Quant(const std::string& name, Bnb4BitLinear& projection);
    void EraseFusedSource(const std::string& name);
    void Hc(const std::string& prefix, Qwen4ExpHyperConnectionLayer& hc, bool inject);

    Context& ctx_;
    Qwen4ExpBnb4Backend& target_;
    ModelWeights& weights_;

    Qwen4ExpLayer layer_;
    Qwen4ExpHyperConnectionLayer mixer_;
    Bnb4BitLinear fcEmbedding_;
    Bnb4BitLinear fcHidden_;
    Tensor* normEmbedding_ = nullptr;
    Tensor* normHidden_ = nullptr;
    Tensor embed_;
    Tensor embedNorm_;
    Tensor embedFc_;
    Tensor hiddenNorm_;
    Qwen4ExpExpertCache experts_;
    std::vector<bf16> hostEmbed_;
    HostTensorView embeddingTable_;

    // Route-scoped descriptors retained across the asynchronous expert submit;
    // the kernels read these slot pointers after Step returns.
    std::array<ops::MoeExpertSlotDev, 10> slots_{};
    std::array<int, 10> slotIds_{};
};
