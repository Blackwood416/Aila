#include "models/Qwen35HybridBnb4Backend.hpp"
#include "models/Qwen35HybridTextBackend.hpp"
#include "utils/EnvUtils.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Opt-in, short GPU correctness fixture. Run under the machine GPU guard;
// this executable runs bounded scenarios rather than a sustained workload.
// Every oracle is a fresh, independent single-sequence Session and uses the
// original Forward. Batch outputs never choose subsequent fixture input IDs.
namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
using Tokens = std::vector<int>;
using Logits = std::vector<float>;
using Trace = std::vector<Logits>;

constexpr int Capacity = 128;
constexpr int DecodeSteps = 5;
constexpr std::array<int, 4> PrefixLengths{0, 1, 7, 33};
constexpr size_t DeviceBudgetBytes = size_t{8} * 1024 * 1024 * 1024;
constexpr size_t PinnedBudgetBytes = size_t{16} * 1024 * 1024;

// [HYPOTHESIS] Provisional whole-model BF16 envelope, fixed before any GPU
// evaluation, not a measured production accuracy claim. B>1 changes GEMV
// reduction order and uses the existing GEMM + BF16 + SwiGLU path; B=1 must
// instead remain bitwise exact. Failures require first-divergence review,
// not an automatic retry with relaxed thresholds. The aggregate limit stops
// an elementwise envelope from hiding systematic drift across the vocabulary.
constexpr double AbsoluteTolerance = 1.0 / 8.0;
constexpr double RelativeTolerance = 1.0 / 32.0;
constexpr double NormalizedRmsTolerance = 1.0 / 64.0;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct DrainOnExit {
    Context& context;
    ~DrainOnExit() { try { context.synchronize(); } catch (...) {} }
};

struct ScopedSnapshotStep {
    int previous = aila::env::g_q35_prefill_step_override;
    explicit ScopedSnapshotStep(int step) { aila::env::g_q35_prefill_step_override = step; }
    ~ScopedSnapshotStep() { aila::env::g_q35_prefill_step_override = previous; }
};

Tokens Prefix(int sequence, int count) {
    Tokens result;
    for (int i = 0; i < count; ++i) result.push_back(1000 + sequence * 71 + (i * 17) % 67);
    return result;
}

int InputToken(int sequence, int step) {
    return 2000 + sequence * 97 + step * 13;
}

template <typename Backend>
class Fixture {
public:
    using Session = typename Backend::Session;
    Fixture(Context& context, const std::string& modelDirectory, int capacity = Capacity)
        : context(context), capacity(capacity), drain{context} {
        Expect(capacity > 0, "Fixture capacity must be positive");
        ModelSpec spec;
        std::string error;
        Expect(aila::modelspec::load_from_dir(modelDirectory, spec, &error),
               "Could not read model specification: " + error);
        Expect(spec.family == ModelFamily::Qwen35Hybrid &&
                   spec.is_bitsandbytes_4bit() == std::is_same_v<Backend, Qwen35HybridBnb4Backend>,
               "The fixture requires weights matching its selected Hybrid backend");
        weights = LoadModelWeightsFromDir(modelDirectory, context);
        Expect(backend.load(context, weights, spec, capacity, &error),
               "Could not load Hybrid: " + error);
        Expect(backend.SupportsDecodeBatch(), "Selected Hybrid configuration does not support decode batches");
        Expect(backend.vocab_size() > 4096, "Fixture token IDs exceed the model vocabulary");
        // Decode batch width is independent of each Session's context limit.
        ids = Tensor::allocate(context, {std::max(capacity, 4)}, dnnl::memory::data_type::s32);
    }

    std::unique_ptr<Session> Create(int sequence, int prefixLength, size_t snapshots = 0) {
        auto session = backend.CreateSession(context, snapshots);
        DrainOnExit sessionDrain{context};
        if (prefixLength) Forward(*session, Prefix(sequence, prefixLength));
        return session;
    }

    const int* Upload(const Tokens& tokens) {
        Expect(!tokens.empty() && tokens.size() <= static_cast<size_t>(ids.numel()), "Invalid fixture input size");
        context.memcpy_h2d(ids.data(), tokens.data(), tokens.size() * sizeof(int));
        return ids.data_as<int>();
    }

    Logits ReadRow(Tensor& output, int row, int expectedRows) {
        const auto vocabulary = static_cast<size_t>(backend.vocab_size());
        Expect(output.ndim() == 2 && output.shape(0) == expectedRows &&
                   output.shape(1) == static_cast<int64_t>(vocabulary) &&
                   output.dtype() == dnnl::memory::data_type::bf16,
               "Forward returned the wrong dense logits layout");
        std::vector<bf16> raw(vocabulary);
        context.memcpy_d2h(raw.data(), output.data_as<bf16>() + static_cast<size_t>(row) * vocabulary,
                           vocabulary * sizeof(bf16));
        context.synchronize();
        Logits values(vocabulary);
        for (size_t i = 0; i < vocabulary; ++i) {
            values[i] = static_cast<float>(raw[i]);
            Expect(std::isfinite(values[i]), "Non-finite logit at " + std::to_string(i));
        }
        return values;
    }

    Logits Forward(Session& session, const Tokens& tokens) {
        const int before = session.CurrentLength();
        Tensor& output = backend.Forward(context, session, Upload(tokens), static_cast<int>(tokens.size()));
        auto result = ReadRow(output, 0, 1);
        Expect(session.CurrentLength() == before + static_cast<int>(tokens.size()),
               "Single-sequence context advance differs");
        return result;
    }

    Trace Batch(const std::vector<Session*>& sessions, const Tokens& tokens) {
        Expect(sessions.size() == tokens.size(), "Batch fixture rows and token IDs differ");
        std::vector<int> before;
        for (const auto* session : sessions) before.push_back(session->CurrentLength());
        Tensor& output = backend.ForwardDecodeBatch(context, sessions, Upload(tokens));
        Trace rows;
        for (size_t row = 0; row < sessions.size(); ++row) {
            rows.push_back(ReadRow(output, static_cast<int>(row), static_cast<int>(sessions.size())));
            Expect(sessions[row]->CurrentLength() == before[row] + 1,
                   "Batch advanced the wrong sequence length");
        }
        return rows;
    }

    Context& context;
    const int capacity;
    ModelWeights weights;
    Backend backend;
    Tensor ids;
    DrainOnExit drain;
};

void Compare(const Logits& actual, const Logits& expected, const std::string& label, bool bitwise) {
    Expect(actual.size() == expected.size() && !actual.empty(), label + ": vocabulary differs");
    if (bitwise) {
        Expect(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
               label + ": required bitwise logits equality failed");
        std::cout << "[PASS] " << label << " values=" << actual.size() << " bitwise=true\n";
        return;
    }

    double squaredError = 0.0, squaredReference = 0.0, maximumAbsolute = 0.0;
    size_t firstFailure = actual.size(), failureCount = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double error = std::abs(static_cast<double>(actual[index]) - expected[index]);
        maximumAbsolute = std::max(maximumAbsolute, error);
        squaredError += error * error;
        squaredReference += static_cast<double>(expected[index]) * expected[index];
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            error > AbsoluteTolerance + RelativeTolerance * std::abs(expected[index])) {
            if (firstFailure == actual.size()) firstFailure = index;
            ++failureCount;
        }
    }
    const double normalizedRms = std::sqrt(squaredError / std::max(squaredReference, 1.0));
    const auto referenceTop = static_cast<size_t>(std::max_element(expected.begin(), expected.end()) - expected.begin());
    const auto actualTop = static_cast<size_t>(std::max_element(actual.begin(), actual.end()) - actual.begin());
    std::cout << std::setprecision(9) << "[NUMERICS] " << label << " values=" << actual.size()
              << " failed_values=" << failureCount << " max_abs=" << maximumAbsolute
              << " normalized_rms=" << normalizedRms << " reference_argmax=" << referenceTop
              << " actual_argmax=" << actualTop << " greedy_changed=" << (referenceTop != actualTop) << '\n';
    if (firstFailure < actual.size()) {
        std::cerr << "[FIRST_FAILURE] " << label << " logit=" << firstFailure
                  << " reference=" << expected[firstFailure] << " actual=" << actual[firstFailure]
                  << " allowed=" << AbsoluteTolerance + RelativeTolerance * std::abs(expected[firstFailure]) << '\n';
    }
    Expect(failureCount == 0 && normalizedRms <= NormalizedRmsTolerance,
           label + ": batch logits failed the fixed numerical contract");
}

template <typename Backend>
std::array<Trace, 4> CaptureIndependentReferences(Fixture<Backend>& fixture) {
    std::array<Trace, 4> references;
    for (int sequence = 0; sequence < 4; ++sequence) {
        auto session = fixture.Create(sequence, PrefixLengths[sequence]);
        DrainOnExit drain{fixture.context};
        for (int step = 0; step < DecodeSteps; ++step) {
            references[sequence].push_back(fixture.Forward(*session, {InputToken(sequence, step)}));
        }
    }
    return references;
}

template <typename Backend>
void CheckFixedBatchSize(Fixture<Backend>& fixture, const std::array<Trace, 4>& references, int batchSize) {
    using Session = typename Backend::Session;
    if (batchSize == 1) {
        for (int sequence = 0; sequence < 4; ++sequence) {
            auto session = fixture.Create(sequence, PrefixLengths[sequence]);
            DrainOnExit drain{fixture.context};
            for (int step = 0; step < DecodeSteps; ++step) {
                const auto actual = fixture.Batch({session.get()}, {InputToken(sequence, step)});
                Compare(actual.front(), references[sequence][step],
                        "fixed B=1 sequence=" + std::to_string(sequence) +
                        " position=" + std::to_string(PrefixLengths[sequence] + step), true);
            }
        }
        return;
    }
    std::array<std::unique_ptr<Session>, 4> sessions;
    DrainOnExit drain{fixture.context};
    for (int sequence = 0; sequence < batchSize; ++sequence) {
        sessions[sequence] = fixture.Create(sequence, PrefixLengths[sequence]);
    }
    for (int step = 0; step < DecodeSteps; ++step) {
        std::vector<Session*> selected;
        Tokens tokens;
        for (int sequence = 0; sequence < batchSize; ++sequence) {
            selected.push_back(sessions[sequence].get());
            tokens.push_back(InputToken(sequence, step));
        }
        const auto actual = fixture.Batch(selected, tokens);
        for (int sequence = 0; sequence < batchSize; ++sequence) {
            Compare(actual[sequence], references[sequence][step],
                    "fixed B=" + std::to_string(batchSize) + " sequence=" + std::to_string(sequence) +
                    " position=" + std::to_string(PrefixLengths[sequence] + step), batchSize == 1);
        }
    }
}

template <typename Backend>
Trace CaptureDynamicMembers(Fixture<Backend>& fixture, bool batched) {
    using Session = typename Backend::Session;
    const std::vector<std::vector<int>> membership{
        {0, 1, 2, 3}, {3, 1, 0}, {2, 0}, {1, 3, 2, 0},
        {2}, {3, 2}, {0, 1, 2}, {1, 0, 3, 2}
    };
    std::array<std::unique_ptr<Session>, 4> sessions;
    DrainOnExit drain{fixture.context};
    for (int sequence = 0; sequence < 4; ++sequence) {
        sessions[sequence] = fixture.Create(sequence, PrefixLengths[sequence]);
    }
    Trace result;
    for (size_t step = 0; step < membership.size(); ++step) {
        if (step == 3) {
            // A stopped member is retired after the last read drained its
            // commands. Its replacement joins at an unrelated position.
            fixture.context.synchronize();
            sessions[1].reset();
            sessions[1] = fixture.Create(5, 5);
        }
        std::array<int, 4> before{};
        for (int sequence = 0; sequence < 4; ++sequence) before[sequence] = sessions[sequence]->CurrentLength();
        std::vector<Session*> selected;
        Tokens tokens;
        for (int sequence : membership[step]) {
            selected.push_back(sessions[sequence].get());
            tokens.push_back(InputToken(sequence, static_cast<int>(step)));
        }
        if (batched) {
            auto rows = fixture.Batch(selected, tokens);
            for (auto& row : rows) result.push_back(std::move(row));
        } else {
            for (size_t row = 0; row < selected.size(); ++row) {
                result.push_back(fixture.Forward(*selected[row], {tokens[row]}));
            }
        }
        for (int sequence = 0; sequence < 4; ++sequence) {
            const bool active = std::find(membership[step].begin(), membership[step].end(), sequence) != membership[step].end();
            Expect(sessions[sequence]->CurrentLength() == before[sequence] + (active ? 1 : 0),
                   "A dynamic batch advanced an inactive or wrong Session");
        }
    }
    return result;
}

template <typename Backend>
void CheckRejectedGroups(Fixture<Backend>& fixture) {
    using Session = typename Backend::Session;
    auto first = fixture.Create(0, 3);
    auto second = fixture.Create(1, 7);
    auto full = fixture.Create(2, Capacity);
    DrainOnExit drain{fixture.context};
    const auto* tokens = fixture.Upload({InputToken(0, 0), InputToken(1, 0), InputToken(2, 0), InputToken(3, 0)});
    auto rejected = [&](const std::vector<Session*>& group, const int* input, const std::string& label) {
        const auto bytes = fixture.context.current_allocated_bytes();
        const std::array<int, 3> lengths{first->CurrentLength(), second->CurrentLength(), full->CurrentLength()};
        bool caught = false;
        try { fixture.backend.ForwardDecodeBatch(fixture.context, group, input); }
        catch (const std::invalid_argument&) { caught = true; }
        Expect(caught, label + ": expected invalid_argument");
        Expect(first->CurrentLength() == lengths[0] && second->CurrentLength() == lengths[1] &&
                   full->CurrentLength() == lengths[2], label + ": rejected group changed Session length");
        Expect(fixture.context.current_allocated_bytes() == bytes, label + ": rejected group allocated workspace");
    };
    rejected({}, tokens, "empty");
    rejected({first.get(), second.get(), first.get(), second.get(), first.get()}, tokens, "oversized");
    rejected({first.get(), second.get()}, nullptr, "null tokens");
    rejected({first.get(), nullptr}, tokens, "null second member");
    rejected({first.get(), first.get()}, tokens, "duplicate");
    rejected({first.get(), full.get()}, tokens, "full second member");

    // Compare observable state, not merely the host length counter: a rejected
    // late row must not have written an earlier row's KV before validation.
    const auto firstActual = fixture.Forward(*first, {InputToken(0, 0)});
    const auto secondActual = fixture.Forward(*second, {InputToken(1, 0)});
    first.reset();
    second.reset();
    full.reset();
    first = fixture.Create(0, 3);
    second = fixture.Create(1, 7);
    Compare(firstActual, fixture.Forward(*first, {InputToken(0, 0)}), "rejected first member state", true);
    Compare(secondActual, fixture.Forward(*second, {InputToken(1, 0)}), "rejected second member state", true);
}
// Batched decode over prefixes that reach the exact-partial-merge threshold.
// The oracle is again an independent single-sequence Session, so a row can only
// pass if the multi-row merge reproduces the single-row kernel exactly.
template <typename Backend>
void CheckLongContextMerged(Context& context, const std::string& modelDirectory) {
    using Session = typename Backend::Session;
    constexpr int LongCapacity = 2176;
    constexpr std::array<int, 4> Prefixes{2100, 2105, 2110, 2115};
    constexpr int LongSteps = 3;
    std::cout << "[RUN] long_context capacity=" << LongCapacity << "\n";
    Fixture<Backend> fixture(context, modelDirectory, LongCapacity);
    std::array<std::array<Logits, LongSteps>, 4> references;
    {
        for (int sequence = 0; sequence < 4; ++sequence) {
            auto session = fixture.Create(sequence, Prefixes[sequence]);
            DrainOnExit drain{context};
            for (int step = 0; step < LongSteps; ++step) {
                references[sequence][step] = fixture.Forward(*session, {InputToken(sequence, step)});
            }
        }
    }
    std::array<std::unique_ptr<Session>, 4> sessions;
    std::vector<Session*> group;
    {
        DrainOnExit drain{context};
        for (int sequence = 0; sequence < 4; ++sequence) {
            sessions[sequence] = fixture.Create(sequence, Prefixes[sequence]);
            group.push_back(sessions[sequence].get());
        }
        for (int step = 0; step < LongSteps; ++step) {
            Tokens tokens;
            for (int sequence = 0; sequence < 4; ++sequence) tokens.push_back(InputToken(sequence, step));
            const auto actual = fixture.Batch(group, tokens);
            for (int sequence = 0; sequence < 4; ++sequence) {
                Compare(actual[sequence], references[sequence][step],
                        "longctx B=4 sequence=" + std::to_string(sequence) +
                        " position=" + std::to_string(Prefixes[sequence] + step), true);
            }
        }
    }
    std::cout << "[PASS] long_context all rows past the exact-merge threshold\n";
}

template <typename Backend>
void CheckSmallCapacity(Context& context, const std::string& modelDirectory) {
    using Session = typename Backend::Session;
    std::cout << "[RUN] small_capacity context=1 B=4\n";
    Fixture<Backend> fixture(context, modelDirectory, 1);
    Trace references;
    Tokens tokens;
    for (int row = 0; row < 4; ++row) {
        auto session = fixture.Create(row, 0);
        tokens.push_back(InputToken(row, 0));
        references.push_back(fixture.Forward(*session, {tokens.back()}));
    }
    std::array<std::unique_ptr<Session>, 4> sessions;
    std::vector<Session*> group;
    for (int row = 0; row < 4; ++row) {
        sessions[row] = fixture.Create(row, 0);
        Expect(sessions[row]->Capacity() == 1, "Small-capacity fixture loaded the wrong context limit");
        group.push_back(sessions[row].get());
    }
    DrainOnExit drain{context};
    const auto original = fixture.Batch(group, tokens);
    for (size_t row = 0; row < group.size(); ++row) {
        Compare(original[row], references[row], "capacity=1 B=4 row=" + std::to_string(row), false);
        Expect(sessions[row]->CurrentLength() == 1, "Small-capacity batch did not fill exactly one slot");
    }
    auto rejectUnchanged = [&](const std::string& label) {
        const auto* input = fixture.Upload(tokens);
        const size_t bytes = context.current_allocated_bytes();
        const size_t allocations = context.device_allocation_count();
        std::array<int, 4> before{};
        for (size_t row = 0; row < group.size(); ++row) before[row] = group[row]->CurrentLength();
        bool caught = false;
        try { fixture.backend.ForwardDecodeBatch(context, group, input); }
        catch (const std::invalid_argument&) { caught = true; }
        Expect(caught, label + ": capacity overflow was not rejected");
        for (size_t row = 0; row < group.size(); ++row) {
            Expect(group[row]->CurrentLength() == before[row], label + ": rejected batch partially advanced a row");
        }
        Expect(context.current_allocated_bytes() == bytes && context.device_allocation_count() == allocations,
               label + ": rejected batch changed tracked allocations");
    };
    rejectUnchanged("all rows at capacity");
    for (size_t row = 0; row + 1 < group.size(); ++row) fixture.backend.Reset(*group[row]);
    context.synchronize();
    rejectUnchanged("only final row at capacity");
    // Do not reset the earlier rows again: a rejected late member must not
    // have changed their convolution histories or recurrent matrices.
    fixture.backend.Reset(*group.back());
    context.synchronize();
    const auto replayed = fixture.Batch(group, tokens);
    for (size_t row = 0; row < group.size(); ++row) {
        Compare(replayed[row], original[row], "capacity rejection replay row=" + std::to_string(row), true);
    }
    std::cout << "[PASS] capacity=1 B=4 complete, full-final-row rejection and reset replay\n";
}

template <typename Backend>
void CheckSnapshotReplay(Fixture<Backend>& fixture) {
    using Session = typename Backend::Session;
    ScopedSnapshotStep snapshotStep(4);
    constexpr std::array<int, 2> prefixLengths{3, 7};
    std::array<std::unique_ptr<Session>, 2> sessions;
    std::vector<Session*> group;
    for (int row = 0; row < 2; ++row) {
        // One snapshot per Session keeps this case bounded and forces the
        // first batch to replace the checkpoint created by its prefix.
        sessions[row] = fixture.Create(row, prefixLengths[row], 1);
        group.push_back(sessions[row].get());
    }
    DrainOnExit drain{fixture.context};
    std::array<Trace, 3> original;
    for (int step = 0; step < 3; ++step) {
        original[step] = fixture.Batch(group, {InputToken(0, step), InputToken(1, step)});
    }
    for (size_t row = 0; row < group.size(); ++row) {
        const int peerLength = group[1 - row]->CurrentLength();
        // Lengths 5/9 must restore the newly batched checkpoints at 4/8.
        // A prefix-only checkpoint, wrong row or off-by-one publication fails.
        const int restored = fixture.backend.Truncate(*group[row], prefixLengths[row] + 2);
        fixture.context.synchronize();
        Expect(restored == prefixLengths[row] + 1 && group[row]->CurrentLength() == restored,
               "Batched snapshot restored the wrong checkpoint");
        Expect(group[1 - row]->CurrentLength() == peerLength,
               "Truncating a snapshot changed its peer's context length");
    }
    for (int step = 1; step < 3; ++step) {
        const auto replayed = fixture.Batch(group, {InputToken(0, step), InputToken(1, step)});
        for (size_t row = 0; row < group.size(); ++row) {
            Compare(replayed[row], original[step][row],
                    "snapshot replay B=2 row=" + std::to_string(row) + " step=" + std::to_string(step), true);
        }
    }
    for (size_t row = 0; row < group.size(); ++row) {
        const int peerLength = group[1 - row]->CurrentLength();
        const int restored = fixture.backend.Truncate(*group[row], prefixLengths[row]);
        fixture.context.synchronize();
        Expect(restored == 0 && group[row]->CurrentLength() == 0,
               "Snapshot quota retained an evicted prefix checkpoint");
        Expect(group[1 - row]->CurrentLength() == peerLength,
               "Evicted-checkpoint reset changed its peer's context length");
    }
    std::cout << "[PASS] snapshot replay B=2 max_snapshots=1 step=4 restored=4,8 replay=bitwise eviction=reset\n";
}

template <typename Backend>
void Run(Context& context, const std::string& modelDirectory) {
    {
        Fixture<Backend> fixture(context, modelDirectory);
        const auto references = CaptureIndependentReferences(fixture);
        for (int batchSize : {1, 2, 3, 4}) CheckFixedBatchSize(fixture, references, batchSize);
        CheckFixedBatchSize(fixture, references, 1);
        const auto dynamicReference = CaptureDynamicMembers(fixture, false);
        const auto dynamicBatch = CaptureDynamicMembers(fixture, true);
        Expect(dynamicBatch.size() == dynamicReference.size(), "Dynamic row counts differ");
        for (size_t row = 0; row < dynamicBatch.size(); ++row) {
            Compare(dynamicBatch[row], dynamicReference[row], "dynamic record=" + std::to_string(row), false);
        }
        CheckRejectedGroups(fixture);
        CheckSnapshotReplay(fixture);
    }
    CheckSmallCapacity<Backend>(context, modelDirectory);
    CheckLongContextMerged<Backend>(context, modelDirectory);
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::string backend, modelDirectory;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "AilaHybridBatchTests --backend qwen35_nf4|qwen35_bf16 --model DIR\n"
                             "Short GPU correctness fixture. Run serially under the machine GPU guard.\n";
                return 0;
            }
            Expect(i + 1 < argc, "Expected a value for " + argument);
            if (argument == "--backend" && backend.empty()) backend = argv[++i];
            else if (argument == "--model" && modelDirectory.empty()) modelDirectory = argv[++i];
            else throw std::invalid_argument("Unknown or duplicate option: " + argument);
        }
        if (backend.empty() || modelDirectory.empty()) {
            std::cout << "SKIP: specify --backend and --model under the machine GPU guard\n";
            return 77;
        }
        Expect(backend == "qwen35_nf4" || backend == "qwen35_bf16", "Unknown Hybrid backend");
        Expect(!aila::env::read_scoped_kv_quant("AILA_VLM_KV_QUANT"), "Hybrid batch fixture requires BF16 KV");
        std::cout << std::setprecision(9)
                  << "[CONTRACT] backend=" << backend << " B1=bitwise B2..4=provisional_bf16 abs=" << AbsoluteTolerance
                  << " rel=" << RelativeTolerance << " normalized_rms=" << NormalizedRmsTolerance
                  << " context=" << Capacity << " decode_steps=" << DecodeSteps
                  << " device_budget_bytes=" << DeviceBudgetBytes << " pinned_budget_bytes=" << PinnedBudgetBytes
                  << " ordinary_snapshots=0 replay_snapshots=1 replay_snapshot_step=4"
                  << " reference=independent_Forward_sessions inputs=teacher_forced\n";
        Context context;
        context.set_allocation_limits(DeviceBudgetBytes, PinnedBudgetBytes);
        const auto device = context.queue().get_device();
        Expect(device.is_gpu(), "This opt-in fixture requires a GPU device");
        std::cout << "[DEVICE] name=" << device.get_info<sycl::info::device::name>()
                  << " driver=" << device.get_info<sycl::info::device::driver_version>() << '\n';
        if (backend == "qwen35_nf4") Run<Qwen35HybridBnb4Backend>(context, modelDirectory);
        else Run<Qwen35HybridTextBackend>(context, modelDirectory);
        context.synchronize();
        context.release_execution_resources();
        std::cout << "[USM] device_bytes=" << context.current_allocated_bytes()
                  << " device_allocations=" << context.device_allocation_count()
                  << " pinned_bytes=" << context.current_pinned_host_bytes()
                  << " pinned_allocations=" << context.pinned_host_allocation_count()
                  << " peak_device_bytes=" << context.peak_allocated_bytes()
                  << " peak_pinned_bytes=" << context.peak_pinned_host_bytes() << '\n';
        Expect(context.current_allocated_bytes() == 0 && context.current_pinned_host_bytes() == 0 &&
                   context.device_allocation_count() == 0 && context.pinned_host_allocation_count() == 0,
               "Hybrid batch fixture leaked tracked device/pinned allocations");
        std::cout << "PASS: Hybrid B1 compatibility, B2/3/4 row logits, dynamic members, rejected groups, capacity<B and snapshot replay\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: Hybrid batch fixture: " << error.what() << '\n';
        return 1;
    }
}
