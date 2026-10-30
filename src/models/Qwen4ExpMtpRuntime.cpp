#include "Qwen4ExpMtpRuntime.hpp"
#include "memory/SessionMemory.hpp"
#include "utils/EnvUtils.hpp"
#include "profile/Profiling.hpp"
#include <algorithm>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

// Recurrent checkpoint geometry, fixed by the Qwen4Exp layer shapes: every
// DeltaNet layer owns a [48, 128, 128] f32 state and a [3, 10240] f32 conv
// ring, and the PLE owns a [9, 10240] f32 conv ring. The checkpoint tensor is
// sized from these numbers, so a geometry change fails the bind checks below
// instead of silently truncating a rollback.
constexpr int kMtpLinearLayers = 36;
constexpr int kMtpStateFloats = 48 * 128 * 128;
constexpr int kMtpConvFloats = 3 * 10240;
constexpr int kMtpPleFloats = 9 * 10240;
constexpr size_t kMtpLinearStateBytes = size_t(kMtpStateFloats) * sizeof(float);
constexpr size_t kMtpLinearConvBytes = size_t(kMtpConvFloats) * sizeof(float);
constexpr size_t kMtpPleCheckpointBytes = size_t(kMtpPleFloats) * sizeof(float);
constexpr size_t kMtpRecurrentCheckpointBytes =
    size_t(kMtpLinearLayers) * (kMtpLinearStateBytes + kMtpLinearConvBytes);
// One work-item moves one 16-byte chunk, so the whole checkpoint is a single
// kMtpRecurrentChunks-wide launch instead of 73 queue memcpys.
constexpr int kMtpLayerChunks = (kMtpStateFloats + kMtpConvFloats) / 4;   // 204,288
constexpr int kMtpStateChunks = kMtpStateFloats / 4;                      // 196,608
constexpr int kMtpRecurrentChunks = static_cast<int>(
    (kMtpRecurrentCheckpointBytes + kMtpPleCheckpointBytes) / (4 * sizeof(float)));
static_assert(kMtpRecurrentCheckpointBytes + kMtpPleCheckpointBytes == 118038528,
              "MTP recurrent checkpoint size must match the checkpoint tensor");

}  // namespace

struct Qwen4ExpMtpRuntime::RecurrentCheckpointPtrs {
    float* linear_state[kMtpLinearLayers];
    float* linear_conv[kMtpLinearLayers];
    float* ple_conv;
};

Qwen4ExpMtpRuntime::Qwen4ExpMtpRuntime(Context& c, Qwen4ExpBnb4Backend& b,
    ModelWeights& weights, int cap) : ctx(c), target(b), capacity(cap) {
    try {
        if (cap < 2 || cap > target.max_seq_len())
            throw std::invalid_argument("MTP requires a draft window within the sequence budget");
        target.ensure_runtime_buffers(ctx, cap);
        draft = std::make_unique<Qwen4ExpMtpDraft>(ctx, target, weights);
        hiddenRows = Tensor::allocate(ctx, {cap, 10240});
        checkpoint = Tensor::allocate(ctx, {117669888 + 368640}, dnnl::memory::data_type::u8);
        linearInputs = Tensor::allocate(ctx, {48, 4, 16480});
        plePrefix = Tensor::allocate(ctx, {1, 10240}, dnnl::memory::data_type::f32);
        ids = Tensor::allocate(ctx, {2}, dnnl::memory::data_type::s32);
        greedyPartials = Tensor::allocate(ctx, {2 * ops::GreedyGroups * int64_t(sizeof(ops::GreedyResult))}, dnnl::memory::data_type::u8);
        greedyResult = Tensor::allocate(ctx, {2 * int64_t(sizeof(ops::GreedyResult))}, dnnl::memory::data_type::u8);
        if (target.buf_.logits.shape(0) < 2)
            target.buf_.logits = Tensor::allocate(ctx, {2, target.vocab_size()});
        target.PrepareDecodeVerification(ctx);
        ctx.synchronize();
        AILA_LOG_INFO("[Q4E Server MTP] enabled=1 experimental=1 bitwise_t1_guarantee=0 width=2 capacity=%d draft_slots=64", cap);
    } catch (...) { try { ctx.synchronize(); } catch (...) {} throw; }
}
Qwen4ExpMtpRuntime::~Qwen4ExpMtpRuntime() {
    try { ctx.synchronize(); } catch (...) {}
    AILA_LOG_INFO("[Q4E Server MTP] windows=%llu accepted=%llu",
        static_cast<unsigned long long>(windows), static_cast<unsigned long long>(accepted));
}
size_t Qwen4ExpMtpRuntime::SessionBytes(int cap) {
    return aila::memory::checked_memory_sum(
        aila::memory::checked_memory_product(size_t(cap), 2304),
        size_t(std::max(1, cap / 4)) * 256 + 10240 * sizeof(bf16));
}
int Qwen4ExpMtpRuntime::EffectiveCapacity(int maxSequenceLength) {
    // Read once per process like the other load-time switches. The default
    // reproduces the original hard gate (dense draft within the QSA indexer
    // budget); raising it must pay for draft KV (2.25 KB/token) and one draft
    // catch-up step per prompt token, see qwen38-r1-mtp-longctx-2026-10-15.md.
    static const int requested = aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT", 2048);
    return std::clamp(requested, 2, std::max(2, maxSequenceLength));
}
std::unique_ptr<Qwen4ExpMtpRuntime::State> Qwen4ExpMtpRuntime::CreateState() {
    auto state = std::make_unique<State>();
    try {
        state->draft = draft->CreateState(capacity);
        state->lastHidden = Tensor::allocate(ctx, {1, 10240});
    } catch (...) { try { ctx.synchronize(); } catch (...) {} throw; }
    return state;
}
void Qwen4ExpMtpRuntime::AfterForward(State& state, int start, const std::vector<int>& tokens) {
    if (!state.valid || state.length != start) { state.valid = false; return; }
    const size_t count = tokens.size();
    // Per-request enablement: a session (or one prefill chunk) reaching past
    // the draft window silently falls back to T1. Invalidate before staging so
    // the check also bounds hiddenRows: the copy below writes count rows and
    // draft->Step writes position start+count-1, so count and position must
    // both stay inside capacity. Invalidate, then keep length truthful so a
    // later Truncate/Reset can restart catch-up.
    if (start < 0 || int64_t(start) + int64_t(count) > int64_t(capacity)) {
        state.valid = false;
        state.length = start + int(count);
        return;
    }
    ctx.queue().memcpy(hiddenRows.data(), target.buf_.hidden4.data(), count * 10240 * sizeof(bf16));
    if (start) draft->Step(*state.draft, state.lastHidden.data_as<bf16>(), tokens.front(), start - 1, true);
    for (size_t r = 0; r + 1 < count; ++r)
        draft->Step(*state.draft, hiddenRows.data_as<bf16>() + r * 10240, tokens[r + 1], start + int(r), true);
    ctx.queue().memcpy(state.lastHidden.data(), hiddenRows.data_as<bf16>() + (count - 1) * 10240, 10240 * sizeof(bf16));
    state.length = start + int(count);
}
std::array<ops::GreedyResult, 2> Qwen4ExpMtpRuntime::Select(int rows) {
    ops::greedy_bf16(ctx, target.buf_.logits.data_as<bf16>(), target.vocab_size(),
        greedyPartials.data_as<ops::GreedyResult>(), greedyResult.data_as<ops::GreedyResult>(), rows);
    std::array<ops::GreedyResult, 2> result{};
    ctx.memcpy_d2h(result.data(), greedyResult.data(), rows * sizeof(result[0]));
    for (int r = 0; r < rows; ++r)
        if (result[r].nonfinite || result[r].index < 0 || result[r].index >= target.vocab_size())
            throw std::runtime_error("MTP nonfinite/invalid greedy logits");
    return result;
}
void Qwen4ExpMtpRuntime::BindRecurrentCheckpoint(Qwen4ExpBnb4Backend::Session& s,
                                                RecurrentCheckpointPtrs& ptrs) {
    if (s.layerCaches_.size() != convHeads.size())
        throw std::runtime_error("MTP recurrent checkpoint layer count changed");
    ptrs = RecurrentCheckpointPtrs{};
    int linear = 0;
    for (size_t l = 0; l < s.layerCaches_.size(); ++l) {
        auto& c = s.layerCaches_[l];
        if (!c.linear_state.valid()) continue;
        if (linear >= kMtpLinearLayers || !c.linear_conv_state.valid() ||
            c.linear_state.size_bytes() != kMtpLinearStateBytes ||
            c.linear_conv_state.size_bytes() != kMtpLinearConvBytes) {
            throw std::runtime_error("MTP recurrent checkpoint geometry changed");
        }
        ptrs.linear_state[linear] = c.linear_state.data_as<float>();
        ptrs.linear_conv[linear] = c.linear_conv_state.data_as<float>();
        ++linear;
    }
    if (linear != kMtpLinearLayers ||
        s.ple_state_.conv_state.size_bytes() != kMtpPleCheckpointBytes) {
        throw std::runtime_error("MTP recurrent checkpoint geometry changed");
    }
    ptrs.ple_conv = s.ple_state_.conv_state.data_as<float>();
}

void Qwen4ExpMtpRuntime::SubmitRecurrentCheckpoint(const RecurrentCheckpointPtrs& ptrs,
                                                   bool capture) {
    using vec4 = sycl::vec<float, 4>;
    float* ckpt = checkpoint.data_as<float>();
    // In-order queue: the launch is ordered after the work that produced the
    // session state and before the next forward reads it.
    ctx.queue().submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(kMtpRecurrentChunks)),
                              sycl::range<1>(256)),
            [=](sycl::nd_item<1> item) {
                const int idx = static_cast<int>(item.get_global_id(0));
                if (idx >= kMtpRecurrentChunks) return;
                float* layer_ptr = nullptr;
                if (idx < kMtpLinearLayers * kMtpLayerChunks) {
                    const int l = idx / kMtpLayerChunks;
                    const int rem = idx - l * kMtpLayerChunks;
                    layer_ptr = rem < kMtpStateChunks
                        ? ptrs.linear_state[l] + static_cast<size_t>(rem) * 4
                        : ptrs.linear_conv[l] + static_cast<size_t>(rem - kMtpStateChunks) * 4;
                } else {
                    layer_ptr = ptrs.ple_conv +
                        static_cast<size_t>(idx - kMtpLinearLayers * kMtpLayerChunks) * 4;
                }
                float* ckpt_ptr = ckpt + static_cast<size_t>(idx) * 4;
                if (capture) {
                    *reinterpret_cast<vec4*>(ckpt_ptr) =
                        *reinterpret_cast<const vec4*>(layer_ptr);
                } else {
                    *reinterpret_cast<vec4*>(layer_ptr) =
                        *reinterpret_cast<const vec4*>(ckpt_ptr);
                }
            });
    });
}

void Qwen4ExpMtpRuntime::Capture(Qwen4ExpBnb4Backend::Session& s) {
    if (s.token_graph_poisoned()) {
        throw std::logic_error(
            "MTP cannot checkpoint a poisoned token-graph session");
    }
    s.InvalidateTokenGraphState();
    length = s.currentLength_; noMiss = s.consecutive_no_miss_layers_;
    pleTokens = s.ple_state_.prev_tokens; pleHead = s.ple_state_.conv_head;
    if (s.ple_state_.conv_head_dev.valid()) {
        ctx.memcpy_d2h(&pleHead, s.ple_state_.conv_head_dev.data(),
                       sizeof(pleHead));
    }
    target.SynchronizeGdnDeviceHeadsToHost(ctx, s);
    for (size_t l = 0; l < s.layerCaches_.size(); ++l) {
        auto& c = s.layerCaches_[l]; convHeads[l] = c.linear_conv_head; blocks[l] = c.num_compressed_blocks;
    }
    RecurrentCheckpointPtrs ptrs{};
    BindRecurrentCheckpoint(s, ptrs);
    SubmitRecurrentCheckpoint(ptrs, true);
}
void Qwen4ExpMtpRuntime::Restore(Qwen4ExpBnb4Backend::Session& s) {
    // Roll back a rejected verification window without treating it as a new
    // session generation. The dedicated epoch rejects prefetch tickets that
    // were prepared against the speculative target state.
    s.InvalidateMtpRestoreState();
    for (size_t l = 0; l < s.layerCaches_.size(); ++l) {
        auto& c = s.layerCaches_[l]; c.linear_conv_head = convHeads[l]; c.num_compressed_blocks = blocks[l];
        if (c.linear_state.valid()) c.device_state_dirty = true;
    }
    RecurrentCheckpointPtrs ptrs{};
    BindRecurrentCheckpoint(s, ptrs);
    SubmitRecurrentCheckpoint(ptrs, false);
    target.PublishGdnHostHeadsToDevice(ctx, s);
    s.ple_state_.prev_tokens = pleTokens; s.ple_state_.conv_head = pleHead;
    if (s.ple_state_.conv_head_dev.valid()) {
        ctx.memcpy_h2d(s.ple_state_.conv_head_dev.data(), &pleHead,
                       sizeof(pleHead));
    }
    s.currentLength_ = length; s.consecutive_no_miss_layers_ = noMiss;
}
aila::inference::DecodeWindowResult Qwen4ExpMtpRuntime::ForwardWindow(
    Qwen4ExpBnb4Backend::Session& s, State& state, int pending,
    const aila::inference::DecodeWindowSelector& acceptFirst) {
    const int start = s.CurrentLength();
    const auto allocations = ctx.device_allocation_count();
    draft->Step(*state.draft, state.lastHidden.data_as<bf16>(), pending, start - 1);
    const std::array<int, 2> proposal{pending, Select(1)[0].index};
    Capture(s);
    {
        struct CaptureInputs {
            Qwen4ExpBnb4Backend& b;
            ~CaptureInputs() { b.verification_linear_inputs_for_test_ = nullptr; }
        } captureInputs{target};
        target.verification_linear_inputs_for_test_ = linearInputs.data_as<bf16>();
        ForwardOptions options{SnapshotPolicy::None, true}; options.decode_verification = true;
        // The proposal lives on the host: the host embedding gather reads it
        // directly, so no device copy (and no queue drain) separates the
        // capture from the verification window.
        options.host_token_ids = proposal.data();
        target.ForwardImpl(ctx, s, ids.data_as<int>(), 2, false, options);
    }
    ctx.queue().memcpy(hiddenRows.data(), target.buf_.hidden4.data(), 2 * 10240 * sizeof(bf16));
    const auto selected = Select(2);
    const bool takeDraft = proposal[1] == selected[0].index && acceptFirst(selected[0].index);
    if (!takeDraft) {
        // Keep only the pending input: replay causal GDN inputs and the first
        // PLE ring write from verification, hiding all appended KV tails.
        ctx.queue().memcpy(plePrefix.data(), s.ple_state_.conv_state.data_as<float>() + size_t(pleHead) * 10240, 10240 * sizeof(float));
        Restore(s);
        for (size_t l = 0; l < target.layers_.size(); ++l) {
            auto& layer = target.layers_[l]; auto& c = s.layerCaches_[l];
            if (!layer.is_linear) { c.num_compressed_blocks = (start + 1) / 4; continue; }
            ops::DeltaNetWeights weights{layer.linear_conv1d_weight->data_as<bf16>(), layer.linear_norm_weight->data_as<float>(),
                layer.linear_A_log->data_as<float>(), layer.linear_dt_bias->data_as<bf16>()};
            ops::DeltaNetState stateView{c.linear_state.data_as<float>(), c.linear_conv_state.data_as<float>(), c.linear_conv_head, true};
            static const bool verifyOrder = aila::env::read_flag("AILA_Q4E_VERIFY_GDN_ORDER", false);
            ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, weights, stateView,
                linearInputs.data_as<bf16>() + l * 4 * 16480, target.buf_.attn_out.data_as<bf16>(), 1, verifyOrder);
            c.linear_conv_head = stateView.conv_head; c.device_state_dirty = stateView.device_state_dirty;
            if (c.linear_conv_head_dev != nullptr) {
                c.linear_conv_head_device_authority = false;
                c.linear_conv_head_device_synced = false;
            }
        }
        ctx.queue().memcpy(s.ple_state_.conv_state.data_as<float>() + size_t(pleHead) * 10240, plePrefix.data(), 10240 * sizeof(float));
        s.ple_state_.conv_head = (pleHead + 1) % 9;
        std::array<int64_t, 64> ngrams{}; std::array<int, 2> prev{};
        target.ple_.compute_ngram_ids_pure(s.ple_state_, proposal.data(), 1, ngrams.data(), prev);
        s.ple_state_.prev_tokens = prev; s.currentLength_ = start + 1; s.consecutive_no_miss_layers_ = 0;
    } else {
        draft->Step(*state.draft, hiddenRows.data_as<bf16>(), proposal[1], start, true);
    }
    state.length = s.currentLength_;
    ctx.queue().memcpy(state.lastHidden.data(), hiddenRows.data_as<bf16>() + (takeDraft ? 10240 : 0), 10240 * sizeof(bf16));
    ++windows; accepted += takeDraft;
    ctx.synchronize();
    if (ctx.device_allocation_count() != allocations)
        throw std::runtime_error("MTP window unexpectedly allocated device memory");
    return {selected[0].index, takeDraft ? std::optional<int>(selected[1].index) : std::nullopt,
        takeDraft ? 2 : 1, s.currentLength_, takeDraft};
}
