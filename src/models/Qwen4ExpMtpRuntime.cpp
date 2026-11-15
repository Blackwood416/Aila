#include "Qwen4ExpMtpRuntime.hpp"
#include "memory/SessionMemory.hpp"
#include "utils/EnvUtils.hpp"
#include "profile/Profiling.hpp"
#include <algorithm>
#include <cstdlib>
#include <string>

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
constexpr int kMtpConvChunks = kMtpConvFloats / 4;                        // 7,680
constexpr int kMtpConvRecurrentChunks =
    (kMtpLinearLayers * kMtpConvFloats + kMtpPleFloats) / 4;              // 299,520
constexpr int kMtpRecurrentChunks = static_cast<int>(
    (kMtpRecurrentCheckpointBytes + kMtpPleCheckpointBytes) / (4 * sizeof(float)));
static_assert(kMtpRecurrentCheckpointBytes + kMtpPleCheckpointBytes == 118038528,
              "MTP recurrent checkpoint size must match the checkpoint tensor");

}  // namespace

struct Qwen4ExpMtpRuntime::RecurrentCheckpointPtrs {
    float* linear_state[kMtpLinearLayers];
    float* linear_conv[kMtpLinearLayers];
    int conv_heads[kMtpLinearLayers];
    float* ple_conv;
    int ple_head;
};

Qwen4ExpMtpRuntime::Qwen4ExpMtpRuntime(Context& c, Qwen4ExpBnb4Backend& b,
    ModelWeights& weights, int cap) : ctx(c), target(b), capacity(cap) {
    try {
        if (cap < 2 || cap > target.max_seq_len())
            throw std::invalid_argument("MTP requires a draft window within the sequence budget");
        const int serverWidth = aila::env::read_int_raw("AILA_Q4E_SERVER_MTP_WIDTH", 0);
        const int widthDefault = (serverWidth >= 2 && serverWidth <= 4) ? serverWidth : 2;
        configuredWidth = std::clamp(aila::env::read_int_raw("AILA_Q4E_MTP_WIDTH", widthDefault), 2, 4);
        target.ensure_runtime_buffers(ctx, std::min(cap, 64));
        draft = std::make_unique<Qwen4ExpMtpDraft>(ctx, target, weights);
        const int hiddenRowsCap = std::min(cap, Qwen4ExpMtpDraft::kMtpBatchEmbedTile);
        hiddenRows = Tensor::allocate(ctx, {hiddenRowsCap, 10240});
        checkpoint = Tensor::allocate(ctx, {117669888 + 368640}, dnnl::memory::data_type::u8);
        linearInputs = Tensor::allocate(ctx, {48, 4, 16480});
        plePrefix = Tensor::allocate(ctx, {1, 10240}, dnnl::memory::data_type::f32);
        ids = Tensor::allocate(ctx, {4}, dnnl::memory::data_type::s32);
        greedyPartials = Tensor::allocate(ctx, {4 * ops::GreedyGroups * int64_t(sizeof(ops::GreedyResult))}, dnnl::memory::data_type::u8);
        greedyResult = Tensor::allocate(ctx, {4 * int64_t(sizeof(ops::GreedyResult))}, dnnl::memory::data_type::u8);
        greedyResultHost = static_cast<ops::GreedyResult*>(ctx.alloc_host(4 * sizeof(ops::GreedyResult)));
        if (target.buf_.logits.shape(0) < 4)
            target.buf_.logits = Tensor::allocate(ctx, {4, target.vocab_size()});
        target.PrepareDecodeVerification(ctx);
        ctx.synchronize();
        AILA_LOG_INFO("[Q4E MTP] enabled=1 width=%d capacity=%d draft_slots=64", configuredWidth, cap);
    } catch (...) {
        try { ctx.synchronize(); } catch (...) {}
        if (greedyResultHost) { ctx.free_host(greedyResultHost); greedyResultHost = nullptr; }
        throw;
    }
}
Qwen4ExpMtpRuntime::~Qwen4ExpMtpRuntime() {
    try { ctx.synchronize(); } catch (...) {}
    if (greedyResultHost) { ctx.free_host(greedyResultHost); greedyResultHost = nullptr; }
    AILA_LOG_DEBUG("[Q4E MTP] windows=%llu accepted=%llu",
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
    static const int requested = aila::env::read_int("AILA_Q4E_MTP_MAX_CONTEXT", 32768);
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
    static const bool s_batch_catchup = aila::env::read_flag("AILA_Q4E_MTP_BATCH_CATCHUP", true);
    const size_t capRows = static_cast<size_t>(std::max<int64_t>(1, hiddenRows.shape(0)));
    if (start) draft->Step(*state.draft, state.lastHidden.data_as<bf16>(), tokens.front(), start - 1, true);
    for (size_t base = 0; base < count; base += capRows) {
        const size_t slice = std::min(capRows, count - base);
        ctx.queue().memcpy(hiddenRows.data(),
                           target.buf_.hidden4.data_as<bf16>() + base * 10240,
                           slice * 10240 * sizeof(bf16));
        const size_t stepEnd = std::min(slice, count - 1 - base);
        // StepBatchKvOnly reuses target.buf_.hidden4[0 .. slice - 1], which has
        // already been copied into hiddenRows; later rows of target.buf_.hidden4
        // remain untouched, so every slice <= kMtpBatchEmbedTile can batch.
        if (s_batch_catchup && stepEnd > 1) {
            draft->StepBatchKvOnly(*state.draft, hiddenRows.data_as<bf16>(),
                                   tokens.data() + base + 1, static_cast<int>(stepEnd),
                                   start + static_cast<int>(base));
        } else {
            for (size_t r = 0; r < stepEnd; ++r) {
                draft->Step(*state.draft, hiddenRows.data_as<bf16>() + r * 10240,
                            tokens[base + r + 1], start + int(base + r), true);
            }
        }
        if (base + slice == count) {
            ctx.queue().memcpy(state.lastHidden.data(),
                               hiddenRows.data_as<bf16>() + (slice - 1) * 10240,
                               10240 * sizeof(bf16));
        }
    }
    state.length = start + int(count);
}
std::array<ops::GreedyResult, 4> Qwen4ExpMtpRuntime::Select(int rows) {
    ops::GreedyResult* dst = greedyResultHost ? greedyResultHost : greedyResult.data_as<ops::GreedyResult>();
    sycl::event ev = ops::greedy_bf16(ctx, target.buf_.logits.data_as<bf16>(), target.vocab_size(),
        greedyPartials.data_as<ops::GreedyResult>(), dst, rows);
    std::array<ops::GreedyResult, 4> result{};
    if (greedyResultHost) {
        ev.wait_and_throw();
        for (int r = 0; r < rows; ++r) result[r] = greedyResultHost[r];
    } else {
        ctx.memcpy_d2h(result.data(), greedyResult.data(), rows * sizeof(result[0]));
    }
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
        ptrs.conv_heads[linear] = convHeads[l] % 3;
        ++linear;
    }
    if (linear != kMtpLinearLayers ||
        s.ple_state_.conv_state.size_bytes() != kMtpPleCheckpointBytes) {
        throw std::runtime_error("MTP recurrent checkpoint geometry changed");
    }
    ptrs.ple_conv = s.ple_state_.conv_state.data_as<float>();
    ptrs.ple_head = pleHead % 9;
}

void Qwen4ExpMtpRuntime::SubmitRecurrentCheckpoint(const RecurrentCheckpointPtrs& ptrs,
                                                   bool capture, int keepPleRows,
                                                   bool skipLinearState) {
    using vec4 = sycl::vec<float, 4>;
    float* ckpt = checkpoint.data_as<float>();
    if (!capture && skipLinearState) {
        ctx.queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(kMtpConvRecurrentChunks)),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item) {
                    const int idx = static_cast<int>(item.get_global_id(0));
                    if (idx >= kMtpConvRecurrentChunks) return;
                    float* layer_ptr = nullptr;
                    const float* ckpt_ptr = nullptr;
                    if (idx < kMtpLinearLayers * kMtpConvChunks) {
                        const int l = idx / kMtpConvChunks;
                        const int rem = idx - l * kMtpConvChunks;
                        layer_ptr = ptrs.linear_conv[l] + static_cast<size_t>(rem) * 4;
                        ckpt_ptr = ckpt + static_cast<size_t>(l * kMtpLayerChunks + kMtpStateChunks + rem) * 4;
                    } else {
                        const int ple_rem = idx - kMtpLinearLayers * kMtpConvChunks;
                        const int ple_row = (ple_rem * 4) / 10240;
                        const int rel_row = (ple_row - ptrs.ple_head + 9) % 9;
                        if (rel_row < keepPleRows) return;
                        layer_ptr = ptrs.ple_conv + static_cast<size_t>(ple_rem) * 4;
                        ckpt_ptr = ckpt + static_cast<size_t>(kMtpLinearLayers * kMtpLayerChunks + ple_rem) * 4;
                    }
                    *reinterpret_cast<vec4*>(layer_ptr) = *reinterpret_cast<const vec4*>(ckpt_ptr);
                });
        });
        return;
    }
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
                    const int ple_rem = idx - kMtpLinearLayers * kMtpLayerChunks;
                    if (!capture) {
                        const int ple_row = (ple_rem * 4) / 10240;
                        const int rel_row = (ple_row - ptrs.ple_head + 9) % 9;
                        if (rel_row < keepPleRows) return;
                    }
                    layer_ptr = ptrs.ple_conv + static_cast<size_t>(ple_rem) * 4;
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
void Qwen4ExpMtpRuntime::Restore(Qwen4ExpBnb4Backend::Session& s, int keepPleRows, bool skipLinearState) {
    // Roll back a rejected verification window without treating it as a new
    // session generation. The dedicated epoch rejects prefetch tickets that
    // were prepared against the speculative target state.
    s.InvalidateMtpRestoreState();
    for (size_t l = 0; l < s.layerCaches_.size(); ++l) {
        auto& c = s.layerCaches_[l]; c.linear_conv_head = convHeads[l];
        if (c.indexer_k_tail_only) {
            c.num_compressed_blocks = std::min(c.num_compressed_blocks, (length + keepPleRows) / 4);
        } else {
            c.num_compressed_blocks = blocks[l];
        }
        if (c.linear_state.valid()) c.device_state_dirty = true;
    }
    RecurrentCheckpointPtrs ptrs{};
    BindRecurrentCheckpoint(s, ptrs);
    SubmitRecurrentCheckpoint(ptrs, false, keepPleRows, skipLinearState);
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
    const aila::inference::DecodeWindowSelector& acceptFirst, int maxWidth) {
    static const bool s_work_summary = aila::env::read_flag("AILA_Q4E_WORK_SUMMARY", false);
    // Top-2 logit margin gates. A drafted token whose margin is below its
    // threshold rarely survives verification, so the window stops one column
    // earlier instead of paying for another draft step plus verify column.
    // Unset, empty or negative means disabled.
    const auto read_margin_threshold = [](const char* name) {
        const std::string raw = aila::env::read_string(name, "");
        return raw.empty() ? -1.0f : std::strtof(raw.c_str(), nullptr);
    };
    static const float s_margin_th1 = read_margin_threshold("AILA_Q4E_MTP_MARGIN_TH1");
    static const float s_margin_th2 = read_margin_threshold("AILA_Q4E_MTP_MARGIN_TH2");
    static const float s_verify_margin_th2 = read_margin_threshold("AILA_Q4E_MTP_VERIFY_MARGIN_TH2");
    static const float s_verify_margin_th3 = read_margin_threshold("AILA_Q4E_MTP_VERIFY_MARGIN_TH3");
    const bool apply_margin_gates = target.cold_epoch_enabled_;
    using Clock = std::chrono::steady_clock;
    const auto t0 = s_work_summary ? Clock::now() : Clock::time_point{};
    const int start = s.CurrentLength();
    const int reqWidth = maxWidth > 0 ? maxWidth : configuredWidth;
    int width = std::clamp(reqWidth, 2, std::min(4, capacity - start));
    const auto allocations = ctx.device_allocation_count();
    struct CaptureHook {
        Qwen4ExpMtpRuntime* self;
        Qwen4ExpBnb4Backend::Session* session;
        bool captured = false;
        static void Run(void* p) {
            auto* h = static_cast<CaptureHook*>(p);
            if (!h->captured) {
                h->self->Capture(*h->session);
                h->captured = true;
            }
        }
    } captureHook{this, &s, false};
    std::array<int, 4> proposal{};
    proposal[0] = pending;
    draft->Step(*state.draft, state.lastHidden.data_as<bf16>(), proposal[0], start - 1, false,
                &CaptureHook::Run, &captureHook);
    const ops::GreedyResult sel1 = Select(1)[0];
    proposal[1] = sel1.index;
    const float margin1 = ops::greedy_margin(sel1);
    const int savedDraftBlocksAfterStep1 = state.draft->cache.num_compressed_blocks;
    if (apply_margin_gates && width >= 3 && s_margin_th1 > 0.0f && margin1 < s_margin_th1) width = 2;
    float margin2 = -1.0f;
    float margin3 = -1.0f;
    for (int d = 2; d < width; ++d) {
        draft->Step(*state.draft, target.buf_.hidden4.data_as<bf16>(), proposal[d - 1], start + d - 2, false);
        const ops::GreedyResult selD = Select(1)[0];
        proposal[d] = selD.index;
        const float mD = ops::greedy_margin(selD);
        if (d == 2) {
            margin2 = mD;
            if (apply_margin_gates && width >= 4 && s_margin_th2 > 0.0f && margin2 < s_margin_th2) width = 3;
        }
        if (d == 3) margin3 = mD;
    }
    if (apply_margin_gates && width == 3 && s_verify_margin_th2 > 0.0f && margin2 >= 0.0f && margin2 < s_verify_margin_th2) width = 2;
    if (apply_margin_gates && width == 4 && s_verify_margin_th3 > 0.0f && margin3 >= 0.0f && margin3 < s_verify_margin_th3) width = 3;
    if (!captureHook.captured) Capture(s);
    const auto t_draft = s_work_summary ? Clock::now() : Clock::time_point{};
    struct DeferSwapGuard {
        Qwen4ExpBnb4Backend& backend;
        Context& ctx;
        DeferSwapGuard(Qwen4ExpBnb4Backend& b, Context& c) : backend(b), ctx(c) {
            backend.defer_adaptive_swap_to_caller_ = true;
        }
        ~DeferSwapGuard() {
            backend.defer_adaptive_swap_to_caller_ = false;
            try { backend.RunPendingAdaptiveSwap(ctx); } catch (...) {}
        }
    } deferSwapGuard(target, ctx);
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
        target.ForwardImpl(ctx, s, ids.data_as<int>(), width, false, options);
    }
    const auto t_verify = s_work_summary ? Clock::now() : Clock::time_point{};
    std::array<ops::GreedyResult, 4> selected{};
    bool kvOnlyPreSubmitted = false;
    if (width >= 2 && width <= 4 && greedyResultHost != nullptr) {
        ctx.queue().memcpy(hiddenRows.data(), target.buf_.hidden4.data(),
                           size_t(width) * 10240 * sizeof(bf16));
        sycl::event verifyGreedyEv = ops::greedy_bf16(
            ctx, target.buf_.logits.data_as<bf16>(), target.vocab_size(),
            greedyPartials.data_as<ops::GreedyResult>(), greedyResultHost, width);
        try {
            (void)verifyGreedyEv.get_info<sycl::info::event::command_execution_status>();
        } catch (...) {}
        // The speculative draft steps above advanced the draft block counter to
        // the last proposed position, so the pre-submitted kv-only step for
        // j = 0 must start again from the step-1 checkpoint.
        state.draft->cache.num_compressed_blocks = savedDraftBlocksAfterStep1;
        draft->Step(*state.draft, hiddenRows.data_as<bf16>(), proposal[1], start, true);
        kvOnlyPreSubmitted = true;
        try {
            (void)ctx.queue().ext_oneapi_submit_barrier()
                .get_info<sycl::info::event::command_execution_status>();
        } catch (...) {}
        target.RunPendingAdaptiveSwap(ctx);
        verifyGreedyEv.wait_and_throw();
        for (int r = 0; r < width; ++r) {
            selected[r] = greedyResultHost[r];
            if (selected[r].nonfinite || selected[r].index < 0 || selected[r].index >= target.vocab_size())
                throw std::runtime_error("MTP nonfinite/invalid greedy logits");
        }
    } else {
        target.RunPendingAdaptiveSwap(ctx);
        selected = Select(width);
    }
    int emitted = 1;
    for (int d = 1; d < width; ++d) {
        if (proposal[d] == selected[d - 1].index && acceptFirst(selected[d - 1].index)) {
            emitted = d + 1;
        } else {
            break;
        }
    }
    if (kvOnlyPreSubmitted) {
        ctx.queue().memcpy(state.lastHidden.data(),
                           hiddenRows.data_as<bf16>() + size_t(emitted - 1) * 10240,
                           10240 * sizeof(bf16));
        if (emitted == 1) {
            // Nothing was accepted, so the pre-submitted block is overwritten by
            // the next window: roll the draft counter back with the window.
            state.draft->cache.num_compressed_blocks = savedDraftBlocksAfterStep1;
        }
    } else {
        ctx.queue().memcpy(state.lastHidden.data(),
                           target.buf_.hidden4.data_as<bf16>() + size_t(emitted - 1) * 10240,
                           10240 * sizeof(bf16));
        if (emitted > 2) {
            ctx.queue().memcpy(hiddenRows.data(), target.buf_.hidden4.data(),
                               size_t(emitted - 1) * 10240 * sizeof(bf16));
        }
        state.draft->cache.num_compressed_blocks = savedDraftBlocksAfterStep1;
    }
    if (emitted < width) {
        static const bool verifyOrder = aila::env::read_flag("AILA_Q4E_VERIFY_GDN_ORDER", false);
        Restore(s, emitted, !verifyOrder);
        const float* ckpt_base = checkpoint.data_as<float>();
        int linear_idx = 0;
        for (size_t l = 0; l < target.layers_.size(); ++l) {
            auto& layer = target.layers_[l]; auto& c = s.layerCaches_[l];
            if (!layer.is_linear) continue;
            const float* ckpt_state = !verifyOrder
                ? (ckpt_base + static_cast<size_t>(linear_idx) * kMtpLayerChunks * 4)
                : nullptr;
            ++linear_idx;
            ops::DeltaNetWeights weights{layer.linear_conv1d_weight->data_as<bf16>(), layer.linear_norm_weight->data_as<float>(),
                layer.linear_A_log->data_as<float>(), layer.linear_dt_bias->data_as<bf16>()};
            ops::DeltaNetState stateView{c.linear_state.data_as<float>(), c.linear_conv_state.data_as<float>(), c.linear_conv_head, true};
            ops::run_linear_delta_prefill_gpu_batched(ctx, layer.geom, weights, stateView,
                linearInputs.data_as<bf16>() + l * 4 * 16480, target.buf_.attn_out.data_as<bf16>(),
                emitted, verifyOrder, false, false, ckpt_state);
            c.linear_conv_head = stateView.conv_head; c.device_state_dirty = stateView.device_state_dirty;
            if (c.linear_conv_head_dev != nullptr) {
                c.linear_conv_head_device_authority = false;
                c.linear_conv_head_device_synced = false;
            }
        }
        s.ple_state_.conv_head = (pleHead + emitted) % 9;
        if (s.ple_state_.conv_head_dev.valid()) {
            ctx.memcpy_h2d(s.ple_state_.conv_head_dev.data(), &s.ple_state_.conv_head,
                           sizeof(s.ple_state_.conv_head));
        }
        std::array<int64_t, 64> ngrams{}; std::array<int, 2> prev{};
        target.ple_.compute_ngram_ids_pure(s.ple_state_, proposal.data(), emitted, ngrams.data(), prev);
        s.ple_state_.prev_tokens = prev; s.currentLength_ = start + emitted; s.consecutive_no_miss_layers_ = 0;
    }
    target.CommitStreamedIndexerTail(ctx, s, start, start + emitted);
    if (kvOnlyPreSubmitted) {
        // j = 0 was submitted before the verify barrier from the same staged
        // rows, so only the rest of the accepted prefix is replayed here.
        for (int j = 1; j + 1 < emitted; ++j) {
            draft->Step(*state.draft, hiddenRows.data_as<bf16>() + size_t(j) * 10240,
                        proposal[j + 1], start + j, true);
        }
    } else {
        const bf16* kvOnlyHiddenBase = (emitted == 2)
            ? target.buf_.hidden4.data_as<bf16>()
            : hiddenRows.data_as<bf16>();
        for (int j = 0; j + 1 < emitted; ++j) {
            draft->Step(*state.draft, kvOnlyHiddenBase + size_t(j) * 10240,
                        proposal[j + 1], start + j, true);
        }
    }
    state.length = s.currentLength_;
    ++windows; accepted += (emitted > 1 ? 1 : 0);
    try {
        (void)ctx.queue().ext_oneapi_submit_barrier()
            .get_info<sycl::info::event::command_execution_status>();
    } catch (...) {}
    if (s_work_summary) {
        const auto t_end = Clock::now();
        auto ms = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        AILA_LOG_INFO("[Q4E MtpWindow] pos=%d count=%d emitted=%d accepted=%d window_ms=%.3f draft_ms=%.3f verify_ms=%.3f tail_ms=%.3f m1=%.2f m2=%.2f m3=%.2f",
            start, width, emitted, emitted - 1,
            ms(t0, t_end), ms(t0, t_draft), ms(t_draft, t_verify), ms(t_verify, t_end),
            margin1, margin2, margin3);
    }
    if (ctx.device_allocation_count() != allocations)
        throw std::runtime_error("MTP window unexpectedly allocated device memory");
    aila::inference::DecodeWindowResult res{};
    res.tokenId = selected[0].index;
    if (emitted >= 2) res.bonusTokenId = selected[1].index;
    res.consumedInputs = emitted;
    res.cacheLength = s.currentLength_;
    res.firstTokenCached = (emitted >= 2);
    res.extraBonusCount = std::max(0, emitted - 2);
    for (int i = 0; i < res.extraBonusCount; ++i) {
        res.extraBonusTokenIds[i] = selected[2 + i].index;
    }
    return res;
}
