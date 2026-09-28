#include "engine/Engine.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include "Qwen4ExpCpuDiagnostic.hpp"

struct Qwen4ExpGenerationTestAccess {
    static void InitFixture(InferenceEngine& engine) {
        engine.ctx_ = std::make_unique<Context>();
        engine.model_spec_.family = ModelFamily::Qwen4Exp;
        engine.token_input_staging_ = Tensor::allocate(*engine.ctx_, {64}, dnnl::memory::data_type::s32);
    }
    static Context& Ctx(InferenceEngine& engine) { return *engine.ctx_; }
    static void Install(InferenceEngine& engine, std::unique_ptr<IModelBackend> backend) {
        engine.backend_ = std::move(backend);
    }
    static Qwen4ExpBnb4Backend& Backend(InferenceEngine& engine) {
        return dynamic_cast<Qwen4ExpBnb4Backend&>(*engine.backend_);
    }
    static Qwen4ExpBnb4Backend::Session& Session(InferenceEngine& engine) {
        return *Backend(engine).defaultSession_;
    }
    static const std::vector<int>& CachedIds(InferenceEngine& engine) { return engine.cached_ids_; }
    static int* Staging(InferenceEngine& engine) { return engine.token_input_staging_.data_as<int>(); }
};

namespace {
using Access = Qwen4ExpGenerationTestAccess;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class RecordingBackend : public IModelBackend {
public:
    struct Call {
        int count;
        ForwardOptions options;
        const int* input;
        size_t allocations;
        std::array<int, 64> ids{};
    };
    std::vector<Call> calls;
    bool throw_next = false;
    bool interior_enabled = false;
    explicit RecordingBackend(Context& ctx)
        : output_(Tensor::allocate(ctx, {1, 4}, dnnl::memory::data_type::f32)) {}
    bool load(Context&, ModelWeights&, const ModelSpec&, int, std::string*) override { return true; }
    bool supports_interior_checkpoint(int count, int anchor) const override {
        return interior_enabled && count <= 64 && anchor >= 2 && count - anchor >= 2;
    }
    Tensor& forward(Context& ctx, const int* ids, int count) override { return forward(ctx, ids, count, {}); }
    Tensor& forward(Context& ctx, const int* ids, int count, const ForwardOptions& options) override {
        Call call{count, options, ids, ctx.device_allocation_count()};
        ctx.memcpy_d2h(call.ids.data(), ids, count * sizeof(int));
        calls.push_back(call);
        if (throw_next) {
            throw_next = false;
            ctx.queue().memset(output_.data(), 0, output_.size_bytes());
            throw std::runtime_error("injected forward failure");
        }
        return options.need_logits ? output_ : unused_;
    }
    void reset() override {}
    bool truncate_kv_cache(int) override { return true; }
    int max_seq_len() const override { return 64; }
    int vocab_size() const override { return 4; }
    ModelFamily family() const override { return ModelFamily::Qwen4Exp; }
private:
    Tensor output_, unused_;
};

void test_prefill_slices() {
    InferenceEngine engine;
    Access::InitFixture(engine);
    auto backend = std::make_unique<RecordingBackend>(Access::Ctx(engine));
    auto& recording = *backend;
    Access::Install(engine, std::move(backend));
    const size_t allocations = Access::Ctx(engine).device_allocation_count();
    std::vector<int> ids(130);
    std::iota(ids.begin(), ids.end(), 100);

    auto* logits = engine.forward_prompt_tokens(ids, 0, 30, 26, true);
    expect(logits && logits->valid(), "Final slice must produce logits");
    expect(recording.calls.size() == 2 && recording.calls[0].count == 26 &&
           recording.calls[1].count == 4, "Reusable prefix must retain the 26+4 split");
    expect(!recording.calls[0].options.need_logits && recording.calls[1].options.need_logits,
           "Only final slice may request logits");
    expect(recording.calls[0].options.capture_snapshot, "Anchor must request a snapshot");
    expect(recording.calls[0].ids[25] == ids[25] && recording.calls[1].ids[0] == ids[26],
           "Staged token slices must preserve offsets");
    for (const auto& call : recording.calls) {
        expect(call.input == Access::Staging(engine) && call.allocations == allocations,
               "Forward helper must reuse preallocated input without extra USM");
    }

    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 0, 30, 26, false, {SnapshotPolicy::None, true});
    expect(recording.calls.size() == 1 && recording.calls[0].count == 30 &&
           !recording.calls[0].options.capture_snapshot, "One-shot must ignore the anchor");

    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 25, 5, 26);
    expect(recording.calls.size() == 2 && recording.calls[0].count == 1 &&
           recording.calls[0].options.capture_snapshot, "One-token anchor must be explicit");

    recording.calls.clear();
    logits = engine.forward_prompt_tokens(ids, 0, 30, 26, false, {SnapshotPolicy::ReusablePrefix, false});
    expect(!logits && !recording.calls[0].options.need_logits && !recording.calls[1].options.need_logits,
           "State-only replay must not return stale logits");

    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 0, 130);
    expect(recording.calls.size() == 3 && recording.calls[0].count == 64 &&
           recording.calls[1].count == 64 && recording.calls[2].count == 2,
           "Slices must fit staging capacity");
    expect(!recording.calls[0].options.need_logits && !recording.calls[1].options.need_logits &&
           recording.calls[2].options.need_logits, "Capacity splitting must compute only final logits");

    recording.calls.clear();
    recording.interior_enabled = true;
    engine.forward_prompt_tokens(ids, 0, 30, 26);
    expect(recording.calls.size() == 1 && recording.calls[0].count == 30 &&
           recording.calls[0].options.checkpoint_after == 26 && recording.calls[0].options.need_logits,
           "Interior checkpoint must preserve one full-length prefill");
    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 25, 5, 26);
    expect(recording.calls.size() == 2 && recording.calls[0].count == 1 &&
           recording.calls[0].options.checkpoint_after == 0,
           "Single-token interior boundary must fall back to split prefill");
    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 0, 30, 29);
    expect(recording.calls.size() == 2 && recording.calls[1].count == 1,
           "Single-token suffix must fall back to split prefill");
    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 0, 130, 80);
    expect(recording.calls.size() == 3 && recording.calls[0].options.checkpoint_after == 0 &&
           recording.calls[1].options.checkpoint_after == 16 && recording.calls[2].options.checkpoint_after == 0,
           "Only the chunk containing the anchor may capture it");
    recording.calls.clear();
    engine.forward_prompt_tokens(ids, 0, 30, 26, false, {SnapshotPolicy::None, true});
    expect(recording.calls.size() == 1 && recording.calls[0].options.checkpoint_after == 0,
           "One-shot must not request an interior checkpoint");
    recording.interior_enabled = false;

    recording.throw_next = true;
    bool failed = false;
    try { engine.forward_prompt_tokens(ids, 0, 30, 26, true); }
    catch (const std::runtime_error&) { failed = true; }
    expect(failed, "Injected failure must propagate");
    logits = engine.forward_prompt_tokens(ids, 0, 4);
    expect(logits && logits->valid() && Access::Ctx(engine).device_allocation_count() == allocations,
           "Staging must remain usable after failure and drain");
    std::cout << "[PASS] Prefill slices, logits demand, anchor and staging lifetime\n";
}

std::vector<uint16_t> read_logits(Context& ctx, const Tensor& tensor) {
    expect(tensor.valid() && tensor.dtype() == dnnl::memory::data_type::bf16, "Expected BF16 logits");
    std::vector<uint16_t> data(tensor.numel());
    ctx.memcpy_d2h(data.data(), tensor.data(), tensor.size_bytes());
    return data;
}

void test_real_model(const std::string& directory) {
    InferenceEngine engine;
    expect(engine.init(directory, 512), "Real Q4E model initialization failed");
    auto& ctx = Access::Ctx(engine);
    auto& backend = Access::Backend(engine);
    auto& session = Access::Session(engine);
    GenerationConfig config;
    config.do_sample = false;
    config.thinking_budget_tokens = 0;
    config.max_new_tokens = 8;
    aila::chat::ChatRequest request;
    std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "Cannot parse fixture");
    aila::chat::ChatFormatTextResult rendered;
    aila::chat::ChatFormatter formatter;
    expect(formatter.render_text(engine.make_chat_format_input(request), true, rendered, &error),
           "Cannot render fixture");
    Tokenizer tokenizer;
    expect(tokenizer.load(directory), "Cannot load fixture tokenizer");
    const auto ids = tokenizer.encode(rendered.text);
    expect(ids.size() == 30, "Story fixture must contain 30 tokens");
    ctx.memcpy_h2d(Access::Staging(engine), ids.data(), ids.size() * sizeof(int));
    backend.forward(ctx, Access::Staging(engine), 26);
    const auto reference_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine) + 26, 4));
    backend.reset();
    const auto previous_logits = read_logits(ctx, backend.buf_for_testing().logits);
    Tensor& unused = backend.forward(ctx, Access::Staging(engine), 26, {SnapshotPolicy::ReusablePrefix, false});
    expect(!unused.valid(), "State-only forward must return invalid logits");
    expect(session.CurrentLength() == 26 && session.snapshot_for_testing(26),
           "State-only prefix must publish its checkpoint");
    expect(previous_logits == read_logits(ctx, backend.buf_for_testing().logits),
           "Skipped output head must leave logits buffer untouched");
    const auto skipped_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine) + 26, 4));
    expect(reference_final == skipped_final, "Skip-logits must preserve every final logit bit");
    expect(backend.Truncate(session, 26) == 26, "Must restore the skipped-head prefix");
    const auto restored_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine) + 26, 4));
    expect(reference_final == restored_final, "Rollback must preserve every final logit bit");
    std::cout << "[PASS] Real 26+4 skip-logits and rollback: " << reference_final.size() << " BF16 logits identical\n";

    backend.reset();
    const auto full_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine), 30));
    for (const int anchor : {2, 16, 26, 28}) {
        backend.reset();
        backend.forward(ctx, Access::Staging(engine), anchor, {SnapshotPolicy::ReusablePrefix, false, true});
        const auto expected_snapshot = *session.snapshot_for_testing(anchor);
        const auto split_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine) + anchor, 30 - anchor));
        backend.reset();
        const auto inline_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine), 30,
            {SnapshotPolicy::ReusablePrefix, true, false, anchor}));
        expect(inline_final == full_final, "Interior checkpoint must preserve full-prefill logits");
        const auto* actual_snapshot = session.snapshot_for_testing(anchor);
        expect(actual_snapshot && session.snapshot_for_testing(30), "Interior and final checkpoints must both be retained");
        expect(actual_snapshot->ple_snapshot.prev_tokens == expected_snapshot.ple_snapshot.prev_tokens &&
               actual_snapshot->ple_snapshot.conv_head == expected_snapshot.ple_snapshot.conv_head,
               "Interior PLE token history must match the split prefix");
        for (size_t i = 0; i < actual_snapshot->layers.size(); ++i) {
            expect(actual_snapshot->layers[i].linear_conv_head == expected_snapshot.layers[i].linear_conv_head &&
                   actual_snapshot->layers[i].num_compressed_blocks == expected_snapshot.layers[i].num_compressed_blocks,
                   "Interior layer metadata must match the split prefix");
        }
        expect(backend.Truncate(session, anchor) == anchor, "Interior checkpoint must be restorable");
        for (size_t i = 0; i < actual_snapshot->layers.size(); ++i) {
            const auto& layer = actual_snapshot->layers[i];
            if (layer.linear_state_host.empty()) continue;
            const auto& cache = session.layer_caches_for_testing()[i];
            std::vector<float> restored(layer.linear_state_host.size());
            ctx.memcpy_d2h(restored.data(), cache.linear_state.data(), restored.size() * sizeof(float));
            expect(restored == layer.linear_state_host, "Interior recurrent state must restore exactly");
            restored.resize(layer.linear_conv_state_host.size());
            ctx.memcpy_d2h(restored.data(), cache.linear_conv_state.data(), restored.size() * sizeof(float));
            expect(restored == layer.linear_conv_state_host, "Interior convolution state must restore exactly");
        }
        const auto replay_final = read_logits(ctx, backend.forward(ctx, Access::Staging(engine) + anchor, 30 - anchor));
        size_t mismatches = 0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < replay_final.size(); ++i) {
            if (replay_final[i] != split_final[i]) ++mismatches;
            uint32_t left_bits = static_cast<uint32_t>(replay_final[i]) << 16;
            uint32_t right_bits = static_cast<uint32_t>(split_final[i]) << 16;
            float left, right;
            std::memcpy(&left, &left_bits, sizeof(float));
            std::memcpy(&right, &right_bits, sizeof(float));
            max_abs_diff = std::max(max_abs_diff, std::abs(left - right));
        }
        std::cout << "[DIAGNOSTIC] Interior anchor=" << anchor << " split-reference mismatches="
                  << mismatches << " max_abs_diff=" << max_abs_diff << '\n';
        if (anchor == 26 || anchor == 28) {
            expect(mismatches == 0, "Golden story anchor rollback must match split logits exactly");
        }
        std::cout << "[PASS] Interior checkpoint anchor=" << anchor << " full-prefill logits and state restore identical\n";
    }
    backend.reset();
    for (const int rejected_anchor : {1, 29}) {
        bool invalid_checkpoint = false;
        try { backend.forward(ctx, Access::Staging(engine), 30,
                               {SnapshotPolicy::ReusablePrefix, true, false, rejected_anchor}); }
        catch (const std::invalid_argument&) { invalid_checkpoint = true; }
        expect(invalid_checkpoint && session.CurrentLength() == 0,
               "Unsupported interior checkpoint must be rejected before advancing state");
    }

    engine.reset_context();
    const std::string first = engine.generate_chat_request_text(request);
    expect(!first.empty() && engine.last_error_code() == EngineErrorCode::Ok, "First turn failed");
    aila::chat::ChatMessage assistant;
    assistant.role = aila::chat::ChatRole::Assistant;
    assistant.content.push_back(aila::chat::ChatContentPart::text_part(first));
    request.messages.push_back(assistant);
    aila::chat::ChatMessage user;
    user.role = aila::chat::ChatRole::User;
    user.content.push_back(aila::chat::ChatContentPart::text_part("Continue with one short sentence about the waves."));
    request.messages.push_back(user);
    const std::string reused = engine.generate_chat_request_text(request);
    expect(!reused.empty() && engine.last_error_code() == EngineErrorCode::Ok, "Second turn failed");
    const auto reused_ids = Access::CachedIds(engine);
    engine.reset_context();
    const std::string fresh = engine.generate_chat_request_text(request);
    expect(fresh == reused && Access::CachedIds(engine) == reused_ids,
           "Two-turn incremental generation must match fresh replay token IDs");
    std::cout << "[PASS] Real two-turn prefix reuse matches fresh replay token IDs\n";

    engine.reset_context();
    auto first_request = request;
    first_request.messages.resize(1);
    expect(engine.generate_chat_request_text(first_request) == first, "Rollback seed turn must be reproducible");
    expect(session.snapshot_for_testing(30) != nullptr, "Rollback seed must retain the first prompt checkpoint");
    request.messages[1].content[0].text = "A crimson sunrise brightens the calm ocean.";
    const auto before_rollback = session.generation();
    const std::string edited = engine.generate_chat_request_text(request);
    expect(!edited.empty() && session.generation() > before_rollback,
           "Edited history must trigger recurrent rollback");
    expect(session.snapshot_for_testing(30) != nullptr,
           "Edited history must retain the reused checkpoint instead of resetting to zero");
    const auto edited_ids = Access::CachedIds(engine);
    engine.reset_context();
    const std::string edited_fresh = engine.generate_chat_request_text(request);
    expect(edited_fresh == edited && Access::CachedIds(engine) == edited_ids,
           "Edited-history rollback must match fresh replay token IDs");
    std::cout << "[PASS] Real edited-history rollback matches fresh replay token IDs\n";

    request.generation_config.snapshot_policy = SnapshotPolicy::None;
    expect(!engine.generate_chat_request_text(request).empty(), "One-shot after reusable request failed");
    expect(engine.context_length() == 0 && session.CurrentLength() == 0, "One-shot must release reusable state");
    for (const auto& slot : session.snapshot_slots_for_testing()) {
        expect(slot.length == -1, "One-shot must not leave a valid checkpoint");
    }
    request.generation_config.snapshot_policy = SnapshotPolicy::ReusablePrefix;
    expect(engine.generate_chat_request_text(request) == edited_fresh && Access::CachedIds(engine) == edited_ids,
           "Reusable generation after one-shot must rebuild correctly");
    std::cout << "[PASS] Snapshot-policy transitions preserve generation\n";
}

void diagnose_real_cpu(const std::string& directory) {
    InferenceEngine engine;
    expect(engine.init(directory, 512), "Real Q4E model initialization failed");
    auto& executor = Access::Backend(engine).expert_executor();
    q4e_cpu_diagnostic::State state;
    executor.set_cpu_observer_for_testing(q4e_cpu_diagnostic::observe, &state);
    struct Detach { aila::Qwen4ExpExpertExecutor& executor;
        ~Detach() { executor.set_cpu_observer_for_testing(nullptr, nullptr); } } detach{executor};
    GenerationConfig config;
    config.do_sample = false;
    config.thinking_budget_tokens = 0;
    config.max_new_tokens = 48;
    aila::chat::ChatRequest request;
    std::string error;
    expect(aila::chat::parse_chat_request_json(
        R"([{"role":"user","content":"Write a short paragraph (about 80 words) describing a sunrise over the ocean."}])",
        config, request, &error), "Cannot parse diagnostic fixture");
    try { engine.generate_chat_request_text(request); }
    catch (const q4e_cpu_diagnostic::Complete&) {}
    expect(state.samples == state.limit, "Diagnostic did not observe enough CPU tasks; enable CPU dispatch");
    std::cout << "[DIAGNOSTIC COMPLETE] samples=" << state.samples << "; not a numerical acceptance or timing run\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        test_prefill_slices();
        if (argc == 3 && std::string(argv[1]) == "--model") test_real_model(argv[2]);
        else if (argc == 3 && std::string(argv[1]) == "--diagnose-cpu") diagnose_real_cpu(argv[2]);
        else if (argc != 1) throw std::invalid_argument("Usage: AilaQwen4ExpGenerationTests [--model|--diagnose-cpu directory]");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
