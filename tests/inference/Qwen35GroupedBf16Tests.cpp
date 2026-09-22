#include "models/Qwen35HybridTextBackend.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/SafeTensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Run the SAME executable twice, serially, in fresh processes:
//   AilaQwen35GroupedBf16Tests --mode host --reference grouped-bf16.ref
//   AilaQwen35GroupedBf16Tests --mode gpu  --reference grouped-bf16.ref
// Both modes use the GPU for the ordinary BF16 projections/norms. Only the
// DeltaNet recurrence/causal convolution switches to the independent host
// implementation in host mode. This is a correctness fixture, not a benchmark.
//
// The public backend accepts synthetic ModelWeights, so no tokenizer, model
// download or 4B checkpoint is required. The single active DeltaNet layer has
// the 4B model's 16 key / 32 value heads, 128-dimensional heads and 4-tap conv.
// A small residual width and identity LM head expose all 32 value heads while
// avoiding unrelated full-attention and multi-layer error accumulation. This
// does not replace an end-to-end real-checkpoint Hybrid test.
//
// Do not enable AILA_DEBUG_Q35_LINEAR_COMPARE here: that helper mixes GPU ring
// storage and host chronological storage without restoring/normalizing the
// ring head. Independent, exclusively host/GPU sessions avoid that ambiguity.

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;
using Tokens = std::vector<int>;
using Logits = std::vector<float>;
using Plan = std::vector<Tokens>;
using Backend = Qwen35HybridTextBackend;

constexpr char FixtureId[] = "aila-q35-grouped-bf16-16x32-d128-k4-v1";
constexpr int Hidden = 128;
constexpr int Intermediate = 128;
constexpr int Vocab = 128;
constexpr int KeyHeads = 16;
constexpr int ValueHeads = 32;
constexpr int HeadDim = 128;
constexpr int ConvKernel = 4;
constexpr int ConvRows = ConvKernel - 1;
constexpr int QDim = KeyHeads * HeadDim;
constexpr int VDim = ValueHeads * HeadDim;
constexpr int QkvDim = 2 * QDim + VDim;
constexpr int Capacity = 256;
constexpr int MaxChunk = 128;
constexpr size_t MaxRecords = 256;

// BF16 has seven explicit fraction bits. The output is BF16 after DeltaNet,
// then a sparse one-to-one projection, residual/norm and identity LM head.
// Permit two output ULPs plus a small absolute allowance for cancellation in
// the residual. Also bound aggregate normalized RMS error to half of 1/128;
// a broad elementwise envelope alone must not hide systematic state drift.
// These constants are fixed before GPU evaluation, with no CLI relaxation.
constexpr int AllowedBf16Ulps = 2;
constexpr double AbsoluteFloor = 1.0 / 512.0;
constexpr double NormalizedRmsLimit = 1.0 / 256.0;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

uint32_t Mix(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

class Fingerprint {
public:
    void Add(const void* data, size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < bytes; ++i) {
            value ^= p[i];
            value *= 1099511628211ull;
        }
    }
    std::string Hex() const {
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << value;
        return out.str();
    }
private:
    uint64_t value = 14695981039346656037ull;
};

class WeightBuilder {
public:
    WeightBuilder(Context& context, ModelWeights& weights)
        : context(context), weights(weights) {}

    template <typename Generator>
    void Bf16(const std::string& name, std::vector<int64_t> shape, Generator generate) {
        Put<bf16>(name, std::move(shape), dnnl::memory::data_type::bf16, generate);
    }
    template <typename Generator>
    void F32(const std::string& name, std::vector<int64_t> shape, Generator generate) {
        Put<float>(name, std::move(shape), dnnl::memory::data_type::f32, generate);
    }
    void ZeroBf16(const std::string& name, std::vector<int64_t> shape) {
        Bf16(name, std::move(shape), [](size_t) { return 0.0f; });
    }
    std::string Hash() const { return fingerprint.Hex(); }
    size_t Bytes() const { return bytes; }

private:
    template <typename T, typename Generator>
    void Put(const std::string& name, std::vector<int64_t> shape,
             dnnl::memory::data_type dtype, Generator generate) {
        size_t elements = 1;
        for (const auto dimension : shape) {
            Expect(dimension > 0 && static_cast<size_t>(dimension) <=
                       std::numeric_limits<size_t>::max() / elements,
                   "Invalid synthetic weight dimensions");
            elements *= static_cast<size_t>(dimension);
        }
        std::vector<T> host(elements);
        for (size_t i = 0; i < elements; ++i) host[i] = T(generate(i));
        auto tensor = Tensor::allocate(context, shape, dtype);
        context.memcpy_h2d(tensor.data(), host.data(), host.size() * sizeof(T));
        fingerprint.Add(name.data(), name.size());
        fingerprint.Add(shape.data(), shape.size() * sizeof(int64_t));
        fingerprint.Add(host.data(), host.size() * sizeof(T));
        bytes += host.size() * sizeof(T);
        weights.put(name, std::move(tensor));
    }

    Context& context;
    ModelWeights& weights;
    Fingerprint fingerprint;
    size_t bytes = 0;
};

ModelSpec MakeSpec() {
    ModelSpec spec;
    spec.family = ModelFamily::Qwen35Hybrid;
    spec.model_type = "qwen3_5";
    auto& config = spec.qwen35_text;
    config.hidden_size = Hidden;
    config.intermediate_size = Intermediate;
    config.vocab_size = Vocab;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = HeadDim;
    config.max_position_embeddings = Capacity;
    config.eos_token_id = Vocab - 1;
    config.tie_word_embeddings = false;
    config.attn_output_gate = false;
    config.linear_num_key_heads = KeyHeads;
    config.linear_num_value_heads = ValueHeads;
    config.linear_key_head_dim = HeadDim;
    config.linear_value_head_dim = HeadDim;
    config.linear_conv_kernel_dim = ConvKernel;
    config.layer_types = {"linear_attention"};
    return spec;
}

void BuildWeights(WeightBuilder& builder) {
    const std::string prefix = "model.language_model.layers.0.";
    builder.Bf16("model.language_model.embed_tokens.weight", {Vocab, Hidden}, [](size_t i) {
        return static_cast<float>(static_cast<int>(Mix(static_cast<uint32_t>(i) + 71u) % 63u) - 31) / 64.0f;
    });
    // These Qwen3.5 norm weights are stored as zero, then load adds one.
    builder.ZeroBf16(prefix + "input_layernorm.weight", {Hidden});
    builder.ZeroBf16(prefix + "post_attention_layernorm.weight", {Hidden});
    builder.ZeroBf16("model.language_model.norm.weight", {Hidden});

    builder.Bf16(prefix + "linear_attn.in_proj_qkv.weight", {QkvDim, Hidden}, [](size_t i) {
        const int row = static_cast<int>(i / Hidden);
        const int column = static_cast<int>(i % Hidden);
        const int kind = row < QDim ? 0 : row < 2 * QDim ? 1 : 2;
        const int local = row - (kind == 0 ? 0 : kind == 1 ? QDim : 2 * QDim);
        const int head = local / HeadDim;
        const int dimension = local % HeadDim;
        const int source = (dimension * 17 + head * 11 + kind * 29) % Hidden;
        const int second = (source + 37) % Hidden;
        if (column == source) return ((head + kind) % 3 == 0) ? -0.25f : 0.25f;
        if (column == second) return 0.125f;
        return 0.0f;
    });
    builder.Bf16(prefix + "linear_attn.in_proj_z.weight", {VDim, Hidden}, [](size_t i) {
        const int row = static_cast<int>(i / Hidden);
        const int source = (row * 13 + (row / HeadDim) * 7 + 19) % Hidden;
        return static_cast<int>(i % Hidden) == source ? 0.5f : 0.0f;
    });
    // a=dt_bias=A_log=b=0 gives beta=1/2 and decay=exp(-log(2)). It keeps
    // recurrence nontrivial while minimizing unrelated exp/softplus error.
    builder.ZeroBf16(prefix + "linear_attn.in_proj_a.weight", {ValueHeads, Hidden});
    builder.ZeroBf16(prefix + "linear_attn.in_proj_b.weight", {ValueHeads, Hidden});
    builder.ZeroBf16(prefix + "linear_attn.dt_bias", {ValueHeads});
    builder.F32(prefix + "linear_attn.A_log", {ValueHeads}, [](size_t) { return 0.0f; });
    builder.F32(prefix + "linear_attn.norm.weight", {HeadDim}, [](size_t i) {
        return 1.0f + static_cast<float>(static_cast<int>(i % 7) - 3) / 32.0f;
    });
    builder.Bf16(prefix + "linear_attn.conv1d.weight", {QkvDim, 1, ConvKernel}, [](size_t i) {
        constexpr std::array<float, ConvKernel> taps{{0.5f, -0.375f, 0.25f, 0.75f}};
        const float scale = (i / ConvKernel) % 3 == 0 ? 0.75f : 1.0f;
        return taps[i % ConvKernel] * scale;
    });
    // Read four distinct channels from EACH value head. An averaging readout
    // could hide a broken follower head behind its correctly updated leader.
    builder.Bf16(prefix + "linear_attn.out_proj.weight", {Hidden, VDim}, [](size_t i) {
        const int row = static_cast<int>(i / VDim);
        const int head = row / 4;
        const int dimension = ((row % 4) * 31 + head * 7) % HeadDim;
        return static_cast<int>(i % VDim) == head * HeadDim + dimension ? 1.0f : 0.0f;
    });
    builder.ZeroBf16(prefix + "mlp.gate_proj.weight", {Intermediate, Hidden});
    builder.ZeroBf16(prefix + "mlp.up_proj.weight", {Intermediate, Hidden});
    builder.ZeroBf16(prefix + "mlp.down_proj.weight", {Hidden, Intermediate});
    builder.Bf16("lm_head.weight", {Vocab, Hidden}, [](size_t i) {
        return i / Hidden == i % Hidden ? 1.0f : 0.0f;
    });
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    const int result = _putenv_s(name, value);
#else
    const int result = setenv(name, value, 1);
#endif
    Expect(result == 0, std::string("Could not configure fixture environment: ") + name);
}

void Configure(bool host) {
    SetEnvironment("AILA_Q35_FORCE_HOST_GROUPED_LINEAR", host ? "1" : "0");
    SetEnvironment("AILA_Q35_LINEAR_DELTA", "1");
    SetEnvironment("AILA_Q35_EXPERIMENTAL_GROUPED_LINEAR_GPU", "0");
    SetEnvironment("AILA_Q35_EXPERIMENTAL_GQA_FASTPATH", "0");
    SetEnvironment("AILA_DEBUG_Q35_LINEAR_COMPARE", "0");
    SetEnvironment("AILA_DEBUG_Q35_LAYER_STATS", "0");
    SetEnvironment("AILA_DEBUG_Q35_LAYER_DETAIL", "-1");
    SetEnvironment("AILA_PROFILE_Q35_DECODE", "0");
    SetEnvironment("AILA_PROFILE_Q35_PREFILL", "0");
    SetEnvironment("AILA_BF16_GEMV", "1");
    SetEnvironment("AILA_KV_QUANT", "0");
    SetEnvironment("AILA_VLM_KV_QUANT", "0");
    aila::env::g_kv_quant_override = false;
    aila::env::disable_persistent_sycl_cache();
}

class PathEvidence {
public:
    PathEvidence() {
        aila::set_log_level(aila::LogLevel::Info);
        aila::set_log_callback(&Receive, this);
    }
    ~PathEvidence() { aila::set_log_callback(nullptr); }
    void Check(bool host) const {
        Expect(host ? (hostPaths == 1 && gpuPaths == 0) : (gpuPaths == 1 && hostPaths == 0),
               "Backend log did not confirm the requested grouped DeltaNet implementation");
        Expect(!debugComparison, "An invalid mixed host/GPU debug oracle was activated");
    }
private:
    static void Receive(int, const char* message, void* data) {
        auto& self = *static_cast<PathEvidence*>(data);
        if (!message) return;
        if (std::strstr(message, "Grouped linear delta backend: host-reference")) ++self.hostPaths;
        if (std::strstr(message, "Grouped linear delta backend: gpu-fastpath-default")) ++self.gpuPaths;
        if (std::strstr(message, "[Q35LinearCompare]")) self.debugComparison = true;
        std::printf("%s\n", message);
    }
    int hostPaths = 0;
    int gpuPaths = 0;
    bool debugComparison = false;
};

struct Fixture {
    Context context;
    ModelWeights weights;
    Backend backend;
    Tensor tokens;
    std::string weightHash;
    size_t weightBytes = 0;

    Fixture() {
        Expect(context.queue().get_device().is_gpu(), "This regression must exercise a GPU device");
        WeightBuilder builder(context, weights);
        BuildWeights(builder);
        weightHash = builder.Hash();
        weightBytes = builder.Bytes();
        Expect(weightBytes < 8 * 1024 * 1024, "Synthetic source weights unexpectedly exceed 8 MiB");
        std::string error;
        Expect(backend.load(context, weights, MakeSpec(), Capacity, &error), "Synthetic BF16 load failed: " + error);
        tokens = Tensor::allocate(context, {MaxChunk}, dnnl::memory::data_type::s32);
        context.synchronize();
    }
    ~Fixture() {
        // A failure still must not free USM referenced by queued work.
        try { context.synchronize(); } catch (...) {}
    }
    std::unique_ptr<Backend::Session> Session() {
        auto result = backend.CreateSession(context, 0); // No snapshots/mirror transfers between steps.
        context.synchronize();
        Expect(result->CurrentLength() == 0, "New synthetic Session is not empty");
        return result;
    }
    Logits Forward(Backend::Session& session, const Tokens& input) {
        Expect(!input.empty() && input.size() <= MaxChunk, "Invalid fixture input size");
        const int before = session.CurrentLength();
        context.memcpy_h2d(tokens.data(), input.data(), input.size() * sizeof(int));
        Tensor& output = backend.Forward(context, session, tokens.data_as<int>(), static_cast<int>(input.size()));
        Expect(output.dtype() == dnnl::memory::data_type::bf16 && output.shape(0) == 1 &&
                   output.shape(1) == Vocab && output.numel() == Vocab,
               "Explicit Session did not expose exactly the last valid logits row");
        std::vector<bf16> host(Vocab);
        context.memcpy_d2h(host.data(), output.data(), host.size() * sizeof(bf16));
        context.synchronize();
        Expect(session.CurrentLength() == before + static_cast<int>(input.size()), "Incorrect Session advance");
        Logits result(Vocab);
        for (int i = 0; i < Vocab; ++i) {
            result[i] = static_cast<float>(host[i]);
            Expect(std::isfinite(result[i]), "Non-finite synthetic logits");
        }
        return result;
    }
};

Plan MakePlan(uint32_t salt, const std::vector<int>& lengths) {
    Plan result;
    uint32_t position = 0;
    for (const int count : lengths) {
        Tokens tokens;
        for (int i = 0; i < count; ++i, ++position) {
            tokens.push_back(static_cast<int>(Mix(salt + position * 173u) % Vocab));
        }
        result.push_back(std::move(tokens));
    }
    Expect(position <= Capacity, "Fixture plan exceeds the context capacity");
    return result;
}

struct Record {
    std::string label;
    int before = 0;
    Tokens input;
    Logits logits;
};
using Trace = std::vector<Record>;

std::string Location(const Record& record) {
    return record.label + " input_positions=[" + std::to_string(record.before) + "," +
           std::to_string(record.before + static_cast<int>(record.input.size())) + ")" +
           " expected_gpu_ring_head=" + std::to_string(record.before % ConvRows);
}

double Bf16Ulp(double magnitude) {
    if (magnitude == 0.0) return std::ldexp(1.0, -133);
    return std::ldexp(1.0, std::max(-126, std::ilogb(std::abs(magnitude))) - 7);
}

double AllowedError(double expected, double actual) {
    return AbsoluteFloor + AllowedBf16Ulps * Bf16Ulp(std::max(std::abs(expected), std::abs(actual)));
}

struct ErrorMetrics {
    size_t changed = 0;
    size_t firstChanged = Vocab;
    size_t firstFailed = Vocab;
    size_t worst = 0;
    double maxAbsolute = 0.0;
    double normalizedRms = 0.0;
};

ErrorMetrics CompareValues(const Logits& actual, const Logits& expected, bool exact) {
    Expect(actual.size() == Vocab && expected.size() == Vocab, "Invalid reference logits width");
    ErrorMetrics result;
    double errorSquared = 0.0;
    double referenceSquared = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        Expect(std::isfinite(actual[i]) && std::isfinite(expected[i]), "Non-finite comparison operand");
        const double difference = std::abs(static_cast<double>(actual[i]) - expected[i]);
        if (actual[i] != expected[i]) {
            if (!result.changed) result.firstChanged = i;
            ++result.changed;
        }
        if (difference > result.maxAbsolute) {
            result.maxAbsolute = difference;
            result.worst = i;
        }
        if (difference > (exact ? 0.0 : AllowedError(expected[i], actual[i])) && result.firstFailed == Vocab) {
            result.firstFailed = i;
        }
        errorSquared += difference * difference;
        referenceSquared += static_cast<double>(expected[i]) * expected[i];
    }
    result.normalizedRms = std::sqrt(errorSquared / std::max(referenceSquared, 1e-20));
    return result;
}

void RequireExact(const Record& record, const Logits& expected) {
    const auto error = CompareValues(record.logits, expected, true);
    if (!error.changed) return;
    const size_t index = error.firstChanged;
    std::ostringstream message;
    message << std::setprecision(9) << "Same-path Session isolation/replay mismatch: " << Location(record)
            << " first_logit=" << index << " expected=" << expected[index]
            << " actual=" << record.logits[index] << " max_abs=" << error.maxAbsolute
            << " normalized_rms=" << error.normalizedRms;
    throw std::runtime_error(message.str());
}

Record RunStep(Fixture& fixture, Backend::Session& session, const Tokens& input, std::string label) {
    Record record{std::move(label), session.CurrentLength(), input, {}};
    record.logits = fixture.Forward(session, input);
    return record;
}

Trace RunSuite(Fixture& fixture) {
    const std::array<Plan, 3> plans{{
        MakePlan(11u, {3, 7, 16, 97, 1, 1, 1, 3, 7, 16, 1, 1, 1}),
        MakePlan(257u, {1, 3, 7, 16, 65, 1, 1, 1, 7, 3, 16, 1, 1}),
        MakePlan(991u, {2, 16, 3, 7, 79, 1, 1, 1, 16, 7, 3, 1, 1}),
    }};
    const std::array<std::string, 3> names{{"A", "B", "C"}};
    std::array<std::vector<Logits>, 3> isolated;
    Trace trace;
    trace.reserve(96);
    std::array<bool, ConvRows> decodeHeads{};
    std::array<bool, ConvRows> prefillHeads{};

    // Independent sessions form an exact oracle for interleaving within each
    // mode. The cross-process comparison below uses the host mode's trace.
    for (size_t s = 0; s < plans.size(); ++s) {
        auto session = fixture.Session();
        for (size_t step = 0; step < plans[s].size(); ++step) {
            auto record = RunStep(fixture, *session, plans[s][step],
                                  "isolated/" + names[s] + "/" + std::to_string(step));
            (record.input.size() == 1 ? decodeHeads : prefillHeads)[record.before % ConvRows] = true;
            isolated[s].push_back(record.logits);
            trace.push_back(std::move(record));
        }
    }
    Expect(std::all_of(decodeHeads.begin(), decodeHeads.end(), [](bool v) { return v; }) &&
               std::all_of(prefillHeads.begin(), prefillHeads.end(), [](bool v) { return v; }),
           "Plans failed to cover decode and prefill starting with every ring head");

    std::array<std::unique_ptr<Backend::Session>, 3> sessions;
    for (auto& session : sessions) session = fixture.Session();
    // Keep A's final decode pending while B is destroyed/recreated and C reset.
    for (size_t step = 0; step + 1 < plans[0].size(); ++step) {
        for (size_t turn = 0; turn < sessions.size(); ++turn) {
            const size_t s = (turn + step) % sessions.size();
            std::array<int, 3> lengths;
            for (size_t other = 0; other < sessions.size(); ++other) lengths[other] = sessions[other]->CurrentLength();
            auto record = RunStep(fixture, *sessions[s], plans[s][step],
                                  "interleaved/" + names[s] + "/" + std::to_string(step));
            RequireExact(record, isolated[s][step]);
            for (size_t other = 0; other < sessions.size(); ++other) {
                if (other != s) Expect(sessions[other]->CurrentLength() == lengths[other], "Peer Session advanced");
            }
            trace.push_back(std::move(record));
        }
    }

    Expect(sessions[1]->CurrentLength() % ConvRows != 0 && sessions[2]->CurrentLength() % ConvRows != 0,
           "Reset/replacement did not start from nonzero logical ring heads");
    fixture.context.synchronize();
    sessions[1].reset();
    sessions[1] = fixture.Session();
    fixture.backend.Reset(*sessions[2]);
    fixture.context.synchronize();
    Expect(sessions[2]->CurrentLength() == 0, "Reset did not clear the Session context");
    for (size_t step = 0; step < 3; ++step) {
        for (size_t s : {size_t{1}, size_t{2}}) {
            auto record = RunStep(fixture, *sessions[s], plans[s][step],
                                  (s == 1 ? "replacement/" : "reset/") + names[s] + "/" + std::to_string(step));
            RequireExact(record, isolated[s][step]);
            trace.push_back(std::move(record));
        }
    }
    auto survivor = RunStep(fixture, *sessions[0], plans[0].back(), "survivor/A/final_decode");
    RequireExact(survivor, isolated[0].back());
    trace.push_back(std::move(survivor));
    for (auto& session : sessions) session.reset();

    // A negative control ensures the fixture exposes history. Same first/last
    // input and identical probe, but a different order of the preceding tokens.
    // Nonzero, unequal conv taps and the per-head readout must make this visible.
    std::array<Logits, 2> historyProbes;
    const std::array<Tokens, 2> prefixes{{Tokens{7, 19, 43, 89}, Tokens{7, 43, 19, 89}}};
    for (size_t i = 0; i < prefixes.size(); ++i) {
        auto session = fixture.Session();
        trace.push_back(RunStep(fixture, *session, prefixes[i], "history_order/" + std::to_string(i) + "/prefix"));
        auto probe = RunStep(fixture, *session, Tokens{101}, "history_order/" + std::to_string(i) + "/probe");
        historyProbes[i] = probe.logits;
        trace.push_back(std::move(probe));
    }
    const auto sensitivity = CompareValues(historyProbes[0], historyProbes[1], false);
    const auto worst = sensitivity.worst;
    Expect(sensitivity.changed >= Vocab / 8 && sensitivity.maxAbsolute >
               8.0 * AllowedError(historyProbes[0][worst], historyProbes[1][worst]),
           "Synthetic fixture is insufficiently sensitive to history ordering");
    std::cout << "[PASS] history-order control max_abs=" << sensitivity.maxAbsolute
              << " changed_logits=" << sensitivity.changed << '\n';
    Expect(trace.size() <= MaxRecords, "Fixture produced too many trace records");
    return trace;
}

struct Reference {
    std::string device;
    std::string driver;
    std::string weightHash;
    size_t weightBytes = 0;
    Trace trace;
};

void WriteReference(const std::filesystem::path& path, const Reference& reference) {
    Expect(!std::filesystem::exists(path), "Refusing to overwrite an existing host reference");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    Expect(out.good(), "Could not create host reference: " + path.string());
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    out << FixtureId << "\nsource host\n";
    out << "device " << std::quoted(reference.device) << "\ndriver " << std::quoted(reference.driver) << '\n';
    out << "weights_fnv1a64 " << reference.weightHash << " bytes " << reference.weightBytes << '\n';
    out << "tolerance " << AllowedBf16Ulps << ' ' << AbsoluteFloor << ' ' << NormalizedRmsLimit << '\n';
    out << "records " << reference.trace.size() << '\n';
    for (const auto& record : reference.trace) {
        out << "step " << std::quoted(record.label) << ' ' << record.before << ' ' << record.input.size() << '\n';
        out << "tokens";
        for (const auto token : record.input) out << ' ' << token;
        out << "\nlogits " << record.logits.size();
        for (const auto value : record.logits) out << ' ' << value;
        out << '\n';
    }
    out << "end\n";
    out.flush();
    Expect(out.good(), "Could not finish writing the host reference");
}

void ReadTag(std::istream& input, const char* expected) {
    std::string actual;
    Expect(static_cast<bool>(input >> actual) && actual == expected,
           std::string("Invalid reference field; expected ") + expected);
}

Reference ReadReference(const std::filesystem::path& path) {
    Expect(std::filesystem::file_size(path) <= 16 * 1024 * 1024, "Oversized reference file");
    std::ifstream input(path);
    Expect(input.good(), "Could not open host reference: " + path.string());
    input.imbue(std::locale::classic());
    Reference result;
    ReadTag(input, FixtureId);
    ReadTag(input, "source");
    ReadTag(input, "host");
    ReadTag(input, "device");
    input >> std::quoted(result.device);
    ReadTag(input, "driver");
    input >> std::quoted(result.driver);
    ReadTag(input, "weights_fnv1a64");
    input >> result.weightHash;
    ReadTag(input, "bytes");
    input >> result.weightBytes;
    ReadTag(input, "tolerance");
    int ulps = 0;
    double absolute = 0.0, rms = 0.0;
    input >> ulps >> absolute >> rms;
    Expect(ulps == AllowedBf16Ulps && absolute == AbsoluteFloor && rms == NormalizedRmsLimit,
           "Reference numerical contract does not match this fixture");
    ReadTag(input, "records");
    size_t count = 0;
    input >> count;
    Expect(count > 0 && count <= MaxRecords, "Invalid reference record count");
    for (size_t i = 0; i < count; ++i) {
        Record record;
        size_t tokens = 0, logits = 0;
        ReadTag(input, "step");
        input >> std::quoted(record.label) >> record.before >> tokens;
        Expect(!record.label.empty() && record.label.size() <= 128 && record.before >= 0 &&
                   record.before <= Capacity && tokens > 0 && tokens <= MaxChunk &&
                   tokens <= static_cast<size_t>(Capacity - record.before), "Invalid reference step");
        ReadTag(input, "tokens");
        record.input.resize(tokens);
        for (auto& token : record.input) {
            input >> token;
            Expect(token >= 0 && token < Vocab, "Invalid reference token");
        }
        ReadTag(input, "logits");
        input >> logits;
        Expect(logits == Vocab, "Invalid reference logits count");
        record.logits.resize(logits);
        for (auto& value : record.logits) {
            input >> value;
            Expect(std::isfinite(value), "Non-finite host reference logit");
        }
        Expect(input.good(), "Truncated reference record");
        result.trace.push_back(std::move(record));
    }
    ReadTag(input, "end");
    input >> std::ws;
    Expect(input.eof(), "Trailing data after host reference");
    return result;
}

void CompareReference(const Trace& actual, const Trace& expected) {
    Expect(actual.size() == expected.size(), "Host/GPU trace lengths differ");
    bool reportedFirst = false;
    size_t failedSteps = 0;
    size_t changedValues = 0;
    double worstAbsolute = 0.0;
    double worstRms = 0.0;
    for (size_t step = 0; step < actual.size(); ++step) {
        const auto& record = actual[step];
        const auto& reference = expected[step];
        Expect(record.label == reference.label && record.before == reference.before && record.input == reference.input,
               "Host/GPU input plan differs at record " + std::to_string(step));
        const auto error = CompareValues(record.logits, reference.logits, false);
        changedValues += error.changed;
        worstAbsolute = std::max(worstAbsolute, error.maxAbsolute);
        worstRms = std::max(worstRms, error.normalizedRms);
        if (!reportedFirst && error.changed) {
            const size_t index = error.firstChanged;
            std::cout << std::setprecision(9) << "[DIFF] first observable host/GPU divergence: " << Location(record)
                      << " logit=" << index << " host=" << reference.logits[index] << " gpu=" << record.logits[index]
                      << " abs=" << std::abs(record.logits[index] - reference.logits[index]) << '\n';
            reportedFirst = true;
        }
        if (error.firstFailed != Vocab || error.normalizedRms > NormalizedRmsLimit) {
            if (!failedSteps) {
                const size_t index = error.firstFailed != Vocab ? error.firstFailed : error.worst;
                std::cerr << std::setprecision(9) << "[FAIL] first out-of-tolerance boundary: " << Location(record)
                          << " logit=" << index << " host=" << reference.logits[index] << " gpu=" << record.logits[index]
                          << " allowed=" << AllowedError(reference.logits[index], record.logits[index])
                          << " max_abs=" << error.maxAbsolute << " normalized_rms=" << error.normalizedRms << '\n';
            }
            ++failedSteps;
        }
    }
    std::cout << std::setprecision(9) << "[COMPARE] steps=" << actual.size() << " changed_logits=" << changedValues
              << " max_abs=" << worstAbsolute << " max_normalized_rms=" << worstRms << " failed_steps=" << failedSteps << '\n';
    Expect(failedSteps == 0, "Grouped BF16 GPU execution failed the independent host oracle");
}

void PrintUsage() {
    std::cout << "Usage: AilaQwen35GroupedBf16Tests --mode host|gpu --reference <file>\n"
                 "Run host first, then gpu, in separate processes on the same GPU/driver.\n"
                 "No external model files are used. Host mode refuses an existing reference.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string mode;
        std::filesystem::path path;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                PrintUsage();
                return 0;
            }
            Expect((argument == "--mode" || argument == "--reference") && i + 1 < argc,
                   "Unknown or incomplete argument: " + argument);
            const std::string value = argv[++i];
            if (argument == "--mode") {
                Expect(mode.empty(), "Duplicate --mode");
                mode = value;
            } else {
                Expect(path.empty(), "Duplicate --reference");
                path = std::filesystem::u8path(value);
            }
        }
        Expect((mode == "host" || mode == "gpu") && !path.empty(), "Both --mode host|gpu and --reference are required");
        const bool host = mode == "host";
        std::optional<Reference> reference;
        if (host) Expect(!std::filesystem::exists(path), "Host reference already exists; use a fresh output path");
        else reference = ReadReference(path); // Validate the file before constructing a GPU Context.

        Configure(host);
        PathEvidence evidence;
        Fixture fixture;
        evidence.Check(host);
        const auto device = fixture.context.queue().get_device();
        const auto deviceName = device.get_info<sycl::info::device::name>();
        const auto driver = device.get_info<sycl::info::device::driver_version>();
        if (reference) {
            Expect(reference->device == deviceName && reference->driver == driver, "Reference GPU/driver differs");
            Expect(reference->weightHash == fixture.weightHash && reference->weightBytes == fixture.weightBytes,
                   "Reference synthetic weights differ");
        }
        std::cout << "[FIXTURE] " << FixtureId << " mode=" << mode << " device=" << deviceName
                  << " driver=" << driver << " source_weight_bytes=" << fixture.weightBytes
                  << " weights_fnv1a64=" << fixture.weightHash << '\n';
        auto trace = RunSuite(fixture);
        evidence.Check(host);
        fixture.context.synchronize();
        if (host) {
            WriteReference(path, {deviceName, driver, fixture.weightHash, fixture.weightBytes, std::move(trace)});
            std::cout << "[PASS] independent host reference written to " << path.string() << '\n';
        } else {
            CompareReference(trace, reference->trace);
            std::cout << "[PASS] grouped BF16 decode/prefill, Session isolation, reset/replacement and continuation\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "AilaQwen35GroupedBf16Tests: " << error.what() << '\n';
        return 1;
    }
}
