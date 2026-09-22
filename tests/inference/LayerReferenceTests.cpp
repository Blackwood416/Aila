#include "inference/Model.hpp"
#include "inference/Request.hpp"
#include "profile/Profiling.hpp"
#include "utils/EnvUtils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace aila::inference;
using Tokens = std::vector<int>;
constexpr int kCapacity = 256;

void Expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Persist the old executable's full float logits, not a reference produced by
// another entry point in the candidate executable. A new directory is required
// for capture, so a candidate can never silently replace its own reference.
// All lengths are compared with the known scenario before reading payloads;
// a corrupt file cannot supply an unbounded allocation or bypass a record.
class Reference final {
public:
    Reference(const std::filesystem::path& path, bool write) : writing(write) {
        if (writing) {
            Expect(std::filesystem::create_directory(path), "Refusing to overwrite reference: " + path.string());
            output.open(path / "reference.bin", std::ios::binary);
            Expect(output.is_open(), "Cannot open reference output");
        } else {
            input.open(path / "reference.bin", std::ios::binary);
            Expect(input.is_open(), "Cannot open reference input");
        }
        Text("AILA_LAYER_REFERENCE_v1_float32_exact");
    }

    void Text(const std::string& value) {
        Scalar(static_cast<uint64_t>(value.size()));
        Bytes(value.data(), value.size());
    }

    template <typename T> void Scalar(const T& value) { Bytes(&value, sizeof(value)); }

    void TokenIds(const Tokens& value) {
        Scalar(static_cast<uint64_t>(value.size()));
        for (int token : value) Scalar(static_cast<int32_t>(token));
    }

    void Logits(const std::string& label, const Tokens& tokens, int length,
                const std::vector<float>& logits) {
        currentLabel = label;
        Text(label);
        TokenIds(tokens);
        Scalar(static_cast<int32_t>(length));
        Scalar(static_cast<uint64_t>(logits.size()));
        for (float value : logits) Expect(std::isfinite(value), label + ": nonfinite logits");
        if (writing) {
            Bytes(logits.data(), logits.size() * sizeof(float));
        } else {
            std::vector<float> expected(logits.size());
            Read(expected.data(), expected.size() * sizeof(float));
            size_t count = 0, first = 0;
            float maxAbs = 0;
            for (size_t i = 0; i < logits.size(); ++i) {
                Expect(std::isfinite(expected[i]), label + ": nonfinite reference logits");
                if (std::memcmp(&expected[i], &logits[i], sizeof(float)) != 0) {
                    if (!count) first = i;
                    ++count;
                    maxAbs = std::max(maxAbs, std::abs(expected[i] - logits[i]));
                }
            }
            if (count) {
                throw std::runtime_error(label + ": " + std::to_string(count) + "/" +
                    std::to_string(logits.size()) + " float mismatches; first=" +
                    std::to_string(first) + " expected=" + std::to_string(expected[first]) +
                    " actual=" + std::to_string(logits[first]) + " max_abs=" + std::to_string(maxAbs));
            }
        }
        ++rows;
        values += logits.size();
        std::cout << "[EXACT] " << label << " context=" << length << " floats=" << logits.size() << std::endl;
    }

    void State(const std::string& label, int length, const Tokens& tokens = {}) {
        currentLabel = label;
        Text(label);
        Scalar(static_cast<int32_t>(length));
        TokenIds(tokens);
    }

    void Finish() {
        Text("COMPLETE");
        Scalar(rows);
        Scalar(values);
        if (writing) {
            output.flush();
            Expect(output.good(), "Reference flush failed");
            output.close();
            Expect(!output.fail(), "Reference close failed");
        } else {
            Expect(input.peek() == std::char_traits<char>::eof() && input.eof(), "Trailing reference payload");
        }
        std::cout << "[PASS] " << (writing ? "Captured" : "Compared") << " rows=" << rows
                  << " float_values=" << values << " tolerance=0 bitwise=true" << std::endl;
    }

private:
    void Read(void* destination, size_t bytes) {
        input.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
        Expect(input.good(), currentLabel + ": truncated or unreadable reference");
    }

    void Bytes(const void* source, size_t bytes) {
        if (!bytes) return;
        if (writing) {
            output.write(static_cast<const char*>(source), static_cast<std::streamsize>(bytes));
            Expect(output.good(), currentLabel + ": reference write failed");
        } else {
            std::vector<char> expected(bytes);
            Read(expected.data(), bytes);
            Expect(std::memcmp(source, expected.data(), bytes) == 0,
                   currentLabel + ": reference metadata/token mismatch");
        }
    }

    bool writing;
    std::ifstream input;
    std::ofstream output;
    std::string currentLabel = "header";
    uint64_t rows = 0;
    uint64_t values = 0;
};

Tokens Sequence(int count, int salt, int vocab) {
    Expect(vocab > 32, "Invalid model vocabulary");
    Tokens result;
    for (int i = 0; i < count; ++i) result.push_back(32 + (salt + i * 131) % std::min(vocab - 32, 4096));
    return result;
}

Tokens Slice(const Tokens& input, int offset, int count) {
    return Tokens(input.begin() + offset, input.begin() + offset + count);
}

void Forward(Reference& reference, const std::string& label,
             const std::shared_ptr<SessionState>& session, const Tokens& tokens) {
    const int before = session->ContextLength();
    auto logits = session->ForwardLogits(tokens);
    Expect(logits.size() == static_cast<size_t>(session->VocabSize()), label + ": incomplete logits row");
    Expect(session->ContextLength() == before + static_cast<int>(tokens.size()), label + ": wrong cache length");
    reference.Logits(label, tokens, session->ContextLength(), logits);
}

void Scenarios(Reference& reference, const std::shared_ptr<LoadedModel>& model, bool hybrid) {
    auto a = model->CreateSession();
    auto b = model->CreateSession();
    const auto sourceA = Sequence(kCapacity, 17, model->VocabSize());
    const auto sourceB = Sequence(kCapacity, 859, model->VocabSize());
    const std::array<int, 8> shapesA{1, 3, 7, 11, 16, 64, 1, 1};
    const std::array<int, 8> shapesB{64, 16, 11, 7, 3, 1, 1, 1};
    int offsetA = 0, offsetB = 0;
    for (size_t i = 0; i < shapesA.size(); ++i) {
        Forward(reference, "mixed/A/" + std::to_string(i), a, Slice(sourceA, offsetA, shapesA[i]));
        Forward(reference, "mixed/B/" + std::to_string(i), b, Slice(sourceB, offsetB, shapesB[i]));
        offsetA += shapesA[i];
        offsetB += shapesB[i];
    }

    a->Reset();
    reference.State("reset/A", a->ContextLength());
    Forward(reference, "rollback/prefill64", a, Slice(sourceA, 0, 64));
    Forward(reference, "rollback/append7", a, Slice(sourceA, 64, 7));
    Forward(reference, "rollback/decode", a, Slice(sourceA, 71, 1));
    const int restored = a->Truncate(66);
    Expect(restored == (hybrid ? 64 : 66), "Unexpected actual checkpoint length");
    reference.State("rollback/actual", a->ContextLength(), {restored});
    Forward(reference, "rollback/replay", a, Slice(sourceA, restored, 72 - restored));
    Forward(reference, "rollback/peer", b, Slice(sourceB, offsetB, 1));
    const int fallback = a->Truncate(2);
    Expect(fallback == (hybrid ? 0 : 2), "Unexpected missing-checkpoint fallback");
    reference.State("rollback/fallback", a->ContextLength(), {fallback});
    Forward(reference, "rollback/fallback_replay", a, Slice(sourceA, fallback, 11 - fallback));

    // A single decode on the snapshot stride must be replayable too.
    a->Reset();
    Forward(reference, "stride/prefill63", a, Slice(sourceA, 0, 63));
    for (int i = 63; i < 67; ++i) Forward(reference, "stride/" + std::to_string(i), a, Slice(sourceA, i, 1));
    const int stride = a->Truncate(65);
    Expect(stride == (hybrid ? 64 : 65), "Decode snapshot stride changed");
    reference.State("stride/actual", a->ContextLength(), {stride});
    for (int i = stride; i < 67; ++i) Forward(reference, "stride/replay/" + std::to_string(i), a, Slice(sourceA, i, 1));

    a->Reset();
    b->Reset();
    RequestOptions options;
    options.prefillChunkSize = 7;
    options.generation.max_new_tokens = 8;
    options.generation.do_sample = true;
    options.generation.use_fixed_seed = true;
    options.generation.sampling_seed = 37;
    options.generation.temperature = 0.8f;
    options.generation.top_k = 32;
    options.generation.top_p = 0.9f;
    options.generation.repetition_penalty = 1.15f;
    options.generation.presence_penalty = 0.2f;
    options.generation.frequency_penalty = 0.1f;
    auto requestA = RequestState::Create(a, Slice(sourceA, 0, 16), options);
    options.prefillChunkSize = 11;
    options.generation.sampling_seed = 109;
    auto requestB = RequestState::Create(b, Slice(sourceB, 0, 23), options);
    for (int steps = 0; steps < 32; ++steps) {
        if (!IsTerminal(requestA->Progress().status)) requestA->Step();
        if (!IsTerminal(requestB->Progress().status)) requestB->Step();
        if (IsTerminal(requestA->Progress().status) && IsTerminal(requestB->Progress().status)) break;
    }
    const auto resultA = requestA->Snapshot(), resultB = requestB->Snapshot();
    Expect(resultA.status == RequestStatus::Completed && resultB.status == RequestStatus::Completed,
           "Seeded requests failed: " + resultA.error + " " + resultB.error);
    Expect(resultA.generatedIds.size() == 8 && resultB.generatedIds.size() == 8, "Incomplete seeded generation");
    reference.State("seed/A", a->ContextLength(), resultA.generatedIds);
    reference.State("seed/B", b->ContextLength(), resultB.generatedIds);
    Forward(reference, "seed/continuationA", a, Slice(sourceA, 24, 3));
    Forward(reference, "seed/continuationB", b, Slice(sourceB, 31, 7));
    requestA.reset();
    a.reset();
    a = model->CreateSession();
    Forward(reference, "recreated/A", a, Slice(sourceA, 0, 11));
    Forward(reference, "recreated/peerB", b, Slice(sourceB, 38, 1));
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cout << "Opt-in GPU fixture: --backend qwen3_nf4|qwen35_nf4|qwen35_bf16 --model DIR --write NEW_DIR|--compare DIR\n";
        return 77;
    }
    try {
        std::string backend, modelPath, referencePath;
        bool writing = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            Expect(i + 1 < argc, "Missing value for " + argument);
            const std::string value = argv[++i];
            if (argument == "--backend" && backend.empty()) backend = value;
            else if (argument == "--model" && modelPath.empty()) modelPath = value;
            else if ((argument == "--write" || argument == "--compare") && referencePath.empty()) {
                writing = argument == "--write";
                referencePath = value;
            } else throw std::runtime_error("Unknown or duplicate argument: " + argument);
        }
        Expect(backend == "qwen3_nf4" || backend == "qwen35_nf4" || backend == "qwen35_bf16", "Invalid backend");
        Expect(!modelPath.empty() && !referencePath.empty(), "Model and reference are required");
        Reference reference(referencePath, writing);
        reference.Text(backend);
        reference.Scalar(static_cast<int32_t>(kCapacity));
        // Config contents identify architecture as well as vocabulary. Model
        // weight/artifact hashes and environment are recorded by the runner.
        std::ifstream config(std::filesystem::path(modelPath) / "config.json", std::ios::binary);
        Expect(config.is_open(), "Missing model config");
        reference.Text(std::string(std::istreambuf_iterator<char>(config), std::istreambuf_iterator<char>()));
        aila::set_log_level(aila::LogLevel::Warning);
        aila::env::g_kv_quant_override = false;
        aila::env::g_q35_prefill_step_override = 64;
        auto runtime = DeviceRuntime::Create();
        runtime->SetAllocationLimits(size_t{8} << 30, size_t{64} << 20);
        ModelOptions options;
        options.maxSequenceLength = kCapacity;
        options.maxSessions = 2;
        options.maxRecurrentSnapshots = 4;
        options.maxSessionHostBytes = size_t{512} << 20;
        auto model = backend == "qwen3_nf4" ? LoadedModel::LoadQwen3Nf4(runtime, modelPath, options)
                   : backend == "qwen35_nf4" ? LoadedModel::LoadQwen35Nf4(runtime, modelPath, options)
                   : LoadedModel::LoadQwen35Bf16(runtime, modelPath, options);
        reference.Scalar(static_cast<int32_t>(model->VocabSize()));
        Scenarios(reference, model, backend != "qwen3_nf4");
        Expect(model->ActiveSessions() == 0, "Session references leaked");
        model.reset();
        Expect(!runtime->FaultMessage(), "Runtime fault during model teardown");
        const auto memory = runtime->Memory();
        std::cout << "[MEMORY] live=" << memory.currentBytes << " peak=" << memory.peakBytes << std::endl;
        reference.Finish();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << std::endl;
        return 1;
    }
}
