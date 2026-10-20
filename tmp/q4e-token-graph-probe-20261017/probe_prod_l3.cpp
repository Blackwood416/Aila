// Production-Session Qwen4Exp L3 direct-vs-graph probe.
//
// The probe intentionally uses the same path as a real model load:
//   InferenceEngine::init -> real tokenizer + lazy weights
//   Qwen4ExpBnb4Backend::Session -> full 2052-token prefix
//   four ordinary count==1 forwards at positions 2052..2055
//
// Direct and graph lanes must be separate processes because
// AILA_Q4E_QSA_LAYER_GRAPH is cached in a function-local static. Each lane writes
// only the L3 cache rows it should write, the graph control table, and a small
// set of sentinel rows. It never copies the complete KV/indexer allocation.

#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "engine/Engine.hpp"
#include "models/Qwen4ExpBnb4Backend.hpp"
#include "ops/GraphControl.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr int kLayer = 3;
constexpr int kPrefixTokens = 2052;
constexpr int kDecodeSteps = 4;
constexpr int kHidden = 2560;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kIndexerDim = 128;
constexpr int kCompressRatio = 4;
// qkv, index_qk, qgkv, QSA prepare, compress, score, select, validator,
// attention, and o_proj are the ten captured production-P10 nodes.
constexpr int kExpectedGraphNodes = 10;
constexpr int kSentinelPosition = kPrefixTokens + kDecodeSteps;
constexpr int kSentinelCompressedBlock =
    (kPrefixTokens + kDecodeSteps) / kCompressRatio;
constexpr uint16_t kSentinelBits = 0x5a5a;
constexpr uint32_t kFileVersion = 1;
constexpr char kFileMagic[8] = {'A', 'I', 'L', '3', 'P', '1', '\0', '\0'};
constexpr const char* kDefaultModel =
    "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";

} // namespace

// The backend and Engine friend this exact global name.
struct Qwen4ExpGenerationTestAccess {
    static Context& Ctx(InferenceEngine& engine) { return *engine.ctx_; }

    static Qwen4ExpBnb4Backend& Backend(InferenceEngine& engine) {
        return dynamic_cast<Qwen4ExpBnb4Backend&>(*engine.backend_);
    }

    static Qwen4ExpBnb4Backend::Session& Session(InferenceEngine& engine) {
        auto& backend = Backend(engine);
        if (!backend.defaultSession_) {
            throw std::runtime_error(
                "Production Session was not created by the runnable forward");
        }
        return *backend.defaultSession_;
    }

    static std::array<int, ops::graph_control::words> ReadGraphControl(
        Qwen4ExpBnb4Backend& backend) {
        std::array<int, ops::graph_control::words> control{};
        if (!backend.qsa_layer_graph_.control.valid()) return control;
        if (backend.qsa_layer_graph_.control.size_bytes() !=
            control.size() * sizeof(int)) {
            throw std::runtime_error("Unexpected L3 graph control size");
        }
        backend.context_->memcpy_d2h(
            control.data(), backend.qsa_layer_graph_.control.data(),
            control.size() * sizeof(int));
        return control;
    }

    static bool HasGraphExecutable(Qwen4ExpBnb4Backend& backend) {
        return backend.qsa_layer_graph_.executable != nullptr;
    }
};

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    if (_putenv_s(name, value) != 0) {
        throw std::runtime_error(std::string("Cannot set environment ") + name);
    }
#else
    if (::setenv(name, value, 1) != 0) {
        throw std::runtime_error(std::string("Cannot set environment ") + name);
    }
#endif
}

uint64_t fnv1a_bytes(uint64_t hash, const void* data, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t fnv1a_ints(const std::vector<int>& values) {
    uint64_t hash = 1469598103934665603ULL;
    for (int value : values) {
        hash = fnv1a_bytes(hash, &value, sizeof(value));
    }
    return hash;
}

uint64_t fnv1a_u16(const std::vector<uint16_t>& values) {
    return fnv1a_bytes(1469598103934665603ULL, values.data(),
                       values.size() * sizeof(uint16_t));
}

std::string default_prompt_text() {
    std::string text;
    text.reserve(24000);
    for (int section = 1; section <= 48; ++section) {
        text += "Field note ";
        text += std::to_string(section);
        text +=
            ": The northern watershed team records rainfall, soil moisture, "
            "wind speed, and stream depth every five minutes. A technician "
            "checks the sensors before sunrise, compares the gauge readings "
            "with a manual sample, and writes each difference in the daily "
            "notebook. When heavy rain interrupts the route, the crew waits "
            "for the access road to clear and then verifies every timestamp. "
            "The review also considers equipment drift, battery voltage, and "
            "the location of trees that may block the telemetry signal. "
            "Afterward, the analyst summarizes the measurements, identifies "
            "missing intervals, and explains how the next observation plan "
            "will reduce uncertainty without changing the established method.\n";
    }
    return text;
}

std::string load_prompt_text(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open prompt file: " + path);
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

std::vector<uint16_t> read_bf16_bits(Context& ctx, const bf16* source,
                                     size_t count) {
    std::vector<uint16_t> bits(count);
    ctx.memcpy_d2h(bits.data(), source, count * sizeof(uint16_t));
    return bits;
}

void append_bf16_bits(Context& ctx, const bf16* source, size_t count,
                      std::vector<uint16_t>& destination) {
    const size_t offset = destination.size();
    destination.resize(offset + count);
    ctx.memcpy_d2h(destination.data() + offset, source,
                   count * sizeof(uint16_t));
}

template <size_t N>
std::array<uint16_t, N> sentinel_row() {
    std::array<uint16_t, N> row{};
    row.fill(kSentinelBits);
    return row;
}

struct SentinelState {
    std::array<uint16_t, kKvHeads * kHeadDim> kv{};
    std::array<uint16_t, kIndexerDim> indexer{};
};

SentinelState seed_sentinels(Context& ctx,
                             Qwen4ExpLayerCache& cache,
                             int capacity) {
    SentinelState expected;
    const auto kv_sentinel = sentinel_row<kKvHeads * kHeadDim>();
    const auto indexer_sentinel = sentinel_row<kIndexerDim>();
    expected.kv = kv_sentinel;
    expected.indexer = indexer_sentinel;

    bf16* k = cache.k.data_as<bf16>();
    bf16* v = cache.v.data_as<bf16>();
    for (int head = 0; head < kKvHeads; ++head) {
        const size_t target_offset =
            (static_cast<size_t>(head) * capacity + kSentinelPosition) *
            kHeadDim;
        const size_t source_offset =
            static_cast<size_t>(head) * kHeadDim;
        ctx.memcpy_h2d(k + target_offset,
                       kv_sentinel.data() + source_offset,
                       kHeadDim * sizeof(uint16_t));
        ctx.memcpy_h2d(v + target_offset,
                       kv_sentinel.data() + source_offset,
                       kHeadDim * sizeof(uint16_t));
    }
    ctx.memcpy_h2d(
        cache.indexer_k.data_as<bf16>() +
            static_cast<size_t>(kSentinelPosition) * kIndexerDim,
        indexer_sentinel.data(), kIndexerDim * sizeof(uint16_t));
    ctx.memcpy_h2d(
        cache.indexer_compressed_k.data_as<bf16>() +
            static_cast<size_t>(kSentinelCompressedBlock) * kIndexerDim,
        indexer_sentinel.data(), kIndexerDim * sizeof(uint16_t));
    ctx.synchronize();
    return expected;
}

bool verify_sentinels(Context& ctx, Qwen4ExpLayerCache& cache,
                      int capacity, const SentinelState& expected) {
    std::array<uint16_t, kKvHeads * kHeadDim> actual_kv{};
    for (int head = 0; head < kKvHeads; ++head) {
        const size_t offset =
            (static_cast<size_t>(head) * capacity + kSentinelPosition) *
            kHeadDim;
        ctx.memcpy_d2h(actual_kv.data() + static_cast<size_t>(head) * kHeadDim,
                       cache.k.data_as<bf16>() + offset,
                       kHeadDim * sizeof(uint16_t));
        std::array<uint16_t, kHeadDim> actual_v{};
        ctx.memcpy_d2h(actual_v.data(), cache.v.data_as<bf16>() + offset,
                       kHeadDim * sizeof(uint16_t));
        if (!std::equal(actual_v.begin(), actual_v.end(),
                        expected.kv.begin() +
                            static_cast<size_t>(head) * kHeadDim)) {
            return false;
        }
    }
    if (!std::equal(actual_kv.begin(), actual_kv.end(),
                    expected.kv.begin())) {
        return false;
    }
    std::array<uint16_t, kIndexerDim> actual_indexer{};
    ctx.memcpy_d2h(
        actual_indexer.data(),
        cache.indexer_k.data_as<bf16>() +
            static_cast<size_t>(kSentinelPosition) * kIndexerDim,
        kIndexerDim * sizeof(uint16_t));
    if (!std::equal(actual_indexer.begin(), actual_indexer.end(),
                    expected.indexer.begin())) {
        return false;
    }
    std::array<uint16_t, kIndexerDim> actual_compressed{};
    ctx.memcpy_d2h(
        actual_compressed.data(),
        cache.indexer_compressed_k.data_as<bf16>() +
            static_cast<size_t>(kSentinelCompressedBlock) * kIndexerDim,
        kIndexerDim * sizeof(uint16_t));
    return std::equal(actual_compressed.begin(), actual_compressed.end(),
                      expected.indexer.begin());
}

struct StepRecord {
    int position = 0;
    int visible_len = 0;
    int expected_blocks = 0;
    int session_blocks = 0;
    int graph_nodes = 0;
    int graph_last_sequence = 0;
    int graph_phase = 0;
    int graph_gate_valid = 0;
    int has_executable = 0;
    int has_compressed_row = 0;
    int sentinel_ok = 0;
    std::array<int, ops::graph_control::words> control{};
    std::vector<uint16_t> k;
    std::vector<uint16_t> v;
    std::vector<uint16_t> indexer_k;
    std::vector<uint16_t> indexer_compressed_k;
};

struct LaneDump {
    int lane = 0;
    int capacity = 0;
    int prefix_len = 0;
    int steps = 0;
    int vocab_size = 0;
    uint64_t token_hash = 0;
    uint64_t prompt_bytes = 0;
    std::vector<int> tokens;
    std::vector<StepRecord> records;
};

StepRecord capture_step(Context& ctx, Qwen4ExpBnb4Backend& backend,
                        Qwen4ExpBnb4Backend::Session& session, int position,
                        const SentinelState& sentinels) {
    StepRecord record;
    record.position = position;
    record.visible_len = position + 1;
    record.expected_blocks = (position + 1) / kCompressRatio;
    auto& cache = session.layer_caches_for_testing().at(kLayer);
    record.session_blocks = cache.num_compressed_blocks;
    record.graph_nodes =
        static_cast<int>(backend.qsa_layer_graph_node_count());
    const auto lease = backend.token_graph_lease();
    record.graph_last_sequence =
        static_cast<int>(lease.last_completed_sequence);
    record.graph_phase = static_cast<int>(lease.phase);
    record.graph_gate_valid = backend.token_graph_runtime().gate_valid() ? 1 : 0;
    record.has_executable =
        Qwen4ExpGenerationTestAccess::HasGraphExecutable(backend) ? 1 : 0;
    record.control =
        Qwen4ExpGenerationTestAccess::ReadGraphControl(backend);
    record.has_compressed_row =
        (position + 1) % kCompressRatio == 0 ? 1 : 0;
    record.sentinel_ok =
        verify_sentinels(ctx, cache, session.Capacity(), sentinels) ? 1 : 0;

    const int capacity = session.Capacity();
    for (int head = 0; head < kKvHeads; ++head) {
        const size_t offset =
            (static_cast<size_t>(head) * capacity + position) * kHeadDim;
        append_bf16_bits(ctx, cache.k.data_as<bf16>() + offset, kHeadDim,
                         record.k);
        append_bf16_bits(ctx, cache.v.data_as<bf16>() + offset, kHeadDim,
                         record.v);
    }
    append_bf16_bits(
        ctx,
        cache.indexer_k.data_as<bf16>() +
            static_cast<size_t>(position) * kIndexerDim,
        kIndexerDim, record.indexer_k);

    record.indexer_compressed_k.assign(kIndexerDim, 0);
    if (record.has_compressed_row) {
        const int completed_block =
            record.expected_blocks - 1;
        record.indexer_compressed_k =
            read_bf16_bits(ctx,
                           cache.indexer_compressed_k.data_as<bf16>() +
                               static_cast<size_t>(completed_block) *
                                   kIndexerDim,
                           kIndexerDim);
    }
    return record;
}

void require_lane_state(const LaneDump& dump, const StepRecord& record,
                        int requested_lane) {
    require(record.session_blocks == record.expected_blocks,
            "Session compressed-block counter mismatch at pos=" +
                std::to_string(record.position));
    require(record.sentinel_ok == 1,
            "A sentinel row changed at pos=" +
                std::to_string(record.position));
    if (requested_lane == 0) {
        require(record.graph_nodes == 0 && record.graph_last_sequence == 0 &&
                    record.graph_phase ==
                        static_cast<int>(aila::TokenGraphPhase::Empty) &&
                    record.has_executable == 0,
                "Direct lane unexpectedly used the L3 graph");
        return;
    }

    const int blocks_slot =
        ops::graph_control_layer_blocks_slot(0);
    require(record.graph_nodes == kExpectedGraphNodes,
            "L3 graph node count mismatch at pos=" +
                std::to_string(record.position));
    require(record.graph_last_sequence ==
                static_cast<int>(record.position - kPrefixTokens + 1),
            "L3 graph completion sequence mismatch at pos=" +
                std::to_string(record.position));
    require(record.graph_phase ==
                static_cast<int>(aila::TokenGraphPhase::Ready) &&
                record.graph_gate_valid == 1 && record.has_executable == 1,
            "L3 graph did not finish in Ready state at pos=" +
                std::to_string(record.position));
    require(record.control[ops::graph_control::start_pos] ==
                record.position,
            "L3 graph control start position mismatch");
    require(record.control[blocks_slot] == record.expected_blocks,
            "L3 graph control block count mismatch at pos=" +
                std::to_string(record.position));
    require(record.control[ops::graph_control::abort] == 0,
            "L3 graph control abort flag is set at pos=" +
                std::to_string(record.position));
    (void)dump;
}

void write_bytes(std::ostream& out, const void* data, size_t bytes) {
    out.write(static_cast<const char*>(data),
              static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("Cannot write probe output");
}

void write_u32(std::ostream& out, uint32_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_i32(std::ostream& out, int32_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_u64(std::ostream& out, uint64_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_u16_vector(std::ostream& out,
                      const std::vector<uint16_t>& values) {
    write_bytes(out, values.data(), values.size() * sizeof(uint16_t));
}

void write_dump(const std::string& path, const LaneDump& dump) {
    const std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot create " + path);
    write_bytes(out, kFileMagic, sizeof(kFileMagic));
    write_u32(out, kFileVersion);
    write_u32(out, static_cast<uint32_t>(dump.lane));
    write_u32(out, static_cast<uint32_t>(dump.capacity));
    write_u32(out, static_cast<uint32_t>(dump.prefix_len));
    write_u32(out, static_cast<uint32_t>(dump.steps));
    write_u32(out, static_cast<uint32_t>(kLayer));
    write_u32(out, static_cast<uint32_t>(kKvHeads));
    write_u32(out, static_cast<uint32_t>(kHeadDim));
    write_u32(out, static_cast<uint32_t>(kIndexerDim));
    write_u32(out, static_cast<uint32_t>(dump.vocab_size));
    write_u64(out, dump.token_hash);
    write_u64(out, dump.prompt_bytes);
    write_u32(out, static_cast<uint32_t>(dump.tokens.size()));
    write_u32(out, 0);
    for (int token : dump.tokens) write_i32(out, token);

    for (const auto& record : dump.records) {
        write_i32(out, record.position);
        write_i32(out, record.visible_len);
        write_i32(out, record.expected_blocks);
        write_i32(out, record.session_blocks);
        write_i32(out, record.graph_nodes);
        write_i32(out, record.graph_last_sequence);
        write_i32(out, record.graph_phase);
        write_i32(out, record.graph_gate_valid);
        write_i32(out, record.has_executable);
        write_i32(out, record.has_compressed_row);
        write_i32(out, record.sentinel_ok);
        write_i32(out, 0);
        for (int value : record.control) write_i32(out, value);
        write_u16_vector(out, record.k);
        write_u16_vector(out, record.v);
        write_u16_vector(out, record.indexer_k);
        write_u16_vector(out, record.indexer_compressed_k);
    }
    if (!out) throw std::runtime_error("Failed writing " + path);
}

void write_summary(const std::string& path, const LaneDump& dump,
                   const std::string& model, const std::string& prompt_file) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot create " + path);
    out << "[PROD L3] lane=" << (dump.lane == 0 ? "direct" : "graph")
        << " model=" << model << " capacity=" << dump.capacity
        << " prefix=" << dump.prefix_len << " steps=" << dump.steps
        << " layer=" << kLayer << " token_hash=0x" << std::hex
        << dump.token_hash << std::dec << " prompt_bytes="
        << dump.prompt_bytes;
    if (!prompt_file.empty()) out << " prompt_file=" << prompt_file;
    out << '\n';
    for (const auto& record : dump.records) {
        out << "[PROD L3 STEP] pos=" << record.position
            << " blocks=" << record.session_blocks
            << " expected_blocks=" << record.expected_blocks
            << " graph_nodes=" << record.graph_nodes
            << " graph_seq=" << record.graph_last_sequence
            << " graph_phase=" << record.graph_phase
            << " k_hash=0x" << std::hex << fnv1a_u16(record.k)
            << " v_hash=0x" << fnv1a_u16(record.v)
            << " idx_hash=0x" << fnv1a_u16(record.indexer_k)
            << " compressed_hash=0x"
            << fnv1a_u16(record.indexer_compressed_k) << std::dec
            << " sentinel_ok=" << record.sentinel_ok << '\n';
    }
}

void expect_read(std::istream& in, void* data, size_t bytes,
                 const std::string& path) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("Truncated probe file: " + path);
}

uint32_t read_u32(std::istream& in, const std::string& path) {
    uint32_t value = 0;
    expect_read(in, &value, sizeof(value), path);
    return value;
}

int32_t read_i32(std::istream& in, const std::string& path) {
    int32_t value = 0;
    expect_read(in, &value, sizeof(value), path);
    return value;
}

uint64_t read_u64(std::istream& in, const std::string& path) {
    uint64_t value = 0;
    expect_read(in, &value, sizeof(value), path);
    return value;
}

std::vector<uint16_t> read_u16_vector(std::istream& in, size_t count,
                                      const std::string& path) {
    std::vector<uint16_t> values(count);
    expect_read(in, values.data(), count * sizeof(uint16_t), path);
    return values;
}

LaneDump read_dump(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    char magic[8]{};
    expect_read(in, magic, sizeof(magic), path);
    require(std::memcmp(magic, kFileMagic, sizeof(magic)) == 0,
            "Bad probe magic in " + path);
    const uint32_t version = read_u32(in, path);
    require(version == kFileVersion, "Unsupported probe version in " + path);
    LaneDump dump;
    dump.lane = static_cast<int>(read_u32(in, path));
    dump.capacity = static_cast<int>(read_u32(in, path));
    dump.prefix_len = static_cast<int>(read_u32(in, path));
    dump.steps = static_cast<int>(read_u32(in, path));
    require(static_cast<int>(read_u32(in, path)) == kLayer,
            "Unexpected layer in " + path);
    require(static_cast<int>(read_u32(in, path)) == kKvHeads,
            "Unexpected KV head count in " + path);
    require(static_cast<int>(read_u32(in, path)) == kHeadDim,
            "Unexpected head dimension in " + path);
    require(static_cast<int>(read_u32(in, path)) == kIndexerDim,
            "Unexpected indexer dimension in " + path);
    dump.vocab_size = static_cast<int>(read_u32(in, path));
    dump.token_hash = read_u64(in, path);
    dump.prompt_bytes = read_u64(in, path);
    const int token_count = static_cast<int>(read_u32(in, path));
    (void)read_u32(in, path);
    require(token_count == dump.prefix_len + dump.steps,
            "Unexpected token count in " + path);
    dump.tokens.resize(token_count);
    for (int& token : dump.tokens) token = read_i32(in, path);
    dump.records.reserve(dump.steps);
    for (int i = 0; i < dump.steps; ++i) {
        StepRecord record;
        record.position = read_i32(in, path);
        record.visible_len = read_i32(in, path);
        record.expected_blocks = read_i32(in, path);
        record.session_blocks = read_i32(in, path);
        record.graph_nodes = read_i32(in, path);
        record.graph_last_sequence = read_i32(in, path);
        record.graph_phase = read_i32(in, path);
        record.graph_gate_valid = read_i32(in, path);
        record.has_executable = read_i32(in, path);
        record.has_compressed_row = read_i32(in, path);
        record.sentinel_ok = read_i32(in, path);
        (void)read_i32(in, path);
        for (int& value : record.control) value = read_i32(in, path);
        record.k = read_u16_vector(in, kKvHeads * kHeadDim, path);
        record.v = read_u16_vector(in, kKvHeads * kHeadDim, path);
        record.indexer_k = read_u16_vector(in, kIndexerDim, path);
        record.indexer_compressed_k =
            read_u16_vector(in, kIndexerDim, path);
        dump.records.push_back(std::move(record));
    }
    return dump;
}

template <typename T>
void expect_equal(const T& direct, const T& graph, const char* name,
                  int position) {
    require(direct == graph,
            std::string("L3 direct/graph mismatch: ") + name +
                " at pos=" + std::to_string(position));
}

int compare_dumps(const std::string& direct_path,
                  const std::string& graph_path) {
    const LaneDump direct = read_dump(direct_path);
    const LaneDump graph = read_dump(graph_path);
    require(direct.lane == 0 && graph.lane == 1,
            "Compare expects direct then graph lane files");
    require(direct.capacity == graph.capacity,
            "Direct/graph capacity mismatch");
    require(direct.prefix_len == kPrefixTokens &&
                graph.prefix_len == kPrefixTokens,
            "Direct/graph prefix length mismatch");
    require(direct.steps == kDecodeSteps && graph.steps == kDecodeSteps,
            "Direct/graph step count mismatch");
    require(direct.vocab_size == graph.vocab_size,
            "Direct/graph vocab mismatch");
    require(direct.token_hash == graph.token_hash &&
                direct.tokens == graph.tokens,
            "Direct/graph tokenizer prefix mismatch");
    require(direct.records.size() == graph.records.size(),
            "Direct/graph record count mismatch");

    const int blocks_slot = ops::graph_control_layer_blocks_slot(0);
    for (size_t i = 0; i < direct.records.size(); ++i) {
        const auto& d = direct.records[i];
        const auto& g = graph.records[i];
        const int position = kPrefixTokens + static_cast<int>(i);
        expect_equal(d.position, g.position, "position", position);
        expect_equal(d.visible_len, g.visible_len, "visible_len", position);
        expect_equal(d.expected_blocks, g.expected_blocks,
                     "expected_blocks", position);
        expect_equal(d.session_blocks, g.session_blocks,
                     "session_blocks", position);
        require(d.sentinel_ok == 1 && g.sentinel_ok == 1,
                "Sentinel verification failed at pos=" +
                    std::to_string(position));
        require(d.graph_nodes == 0 && d.graph_last_sequence == 0 &&
                    d.graph_phase ==
                        static_cast<int>(aila::TokenGraphPhase::Empty) &&
                    d.has_executable == 0,
                "Direct lane graph state is not empty");
        require(g.graph_nodes == kExpectedGraphNodes &&
                    g.graph_last_sequence == static_cast<int>(i + 1) &&
                    g.graph_phase ==
                        static_cast<int>(aila::TokenGraphPhase::Ready) &&
                    g.graph_gate_valid == 1 && g.has_executable == 1,
                "Graph lane did not execute the L3 executable at pos=" +
                    std::to_string(position));
        require(g.control[ops::graph_control::start_pos] == position &&
                    g.control[blocks_slot] == g.expected_blocks &&
                    g.control[ops::graph_control::abort] == 0,
                "Graph control counters mismatch at pos=" +
                    std::to_string(position));
        expect_equal(d.k, g.k, "K row", position);
        expect_equal(d.v, g.v, "V row", position);
        expect_equal(d.indexer_k, g.indexer_k, "indexer K row", position);
        expect_equal(d.has_compressed_row, g.has_compressed_row,
                     "compressed-row flag", position);
        if (d.has_compressed_row) {
            expect_equal(d.indexer_compressed_k,
                         g.indexer_compressed_k,
                         "compressed indexer row", position);
        }
    }
    std::cout
        << "[PROD L3 COMPARE] bit_exact=1 rows=K,V,indexer,compressed "
           "capacity="
        << direct.capacity << " prefix=" << direct.prefix_len
        << " positions=" << kPrefixTokens << ".."
        << kPrefixTokens + kDecodeSteps - 1
        << " graph_nodes=" << kExpectedGraphNodes
        << " control_gate=1 sentinel=1\n";
    return 0;
}

struct Arguments {
    enum class Mode { Lane, Compare, Help };
    Mode mode = Mode::Help;
    int lane = -1;
    int capacity = 4096;
    std::string model = kDefaultModel;
    std::string output;
    std::string prompt_file;
    std::string direct_file;
    std::string graph_file;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    if (argc <= 1) return args;
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        return args;
    }
    if (std::string(argv[1]) == "--compare") {
        if (argc != 4) {
            throw std::invalid_argument(
                "--compare requires DIRECT.bin GRAPH.bin");
        }
        args.mode = Arguments::Mode::Compare;
        args.direct_file = argv[2];
        args.graph_file = argv[3];
        return args;
    }

    args.mode = Arguments::Mode::Lane;
    bool saw_lane = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) +
                                            " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--lane") {
            const std::string lane = next("--lane");
            if (lane == "direct") {
                args.lane = 0;
            } else if (lane == "graph") {
                args.lane = 1;
            } else {
                throw std::invalid_argument("--lane must be direct or graph");
            }
            saw_lane = true;
        } else if (arg == "--model") {
            args.model = next("--model");
        } else if (arg == "--capacity") {
            args.capacity = std::stoi(next("--capacity"));
        } else if (arg == "--output") {
            args.output = next("--output");
        } else if (arg == "--prompt-file") {
            args.prompt_file = next("--prompt-file");
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    require(saw_lane, "--lane is required");
    require(!args.output.empty(), "--output is required");
    require(args.capacity == 4096 || args.capacity == 8192,
            "--capacity must be 4096 or 8192");
    require(args.capacity >= kPrefixTokens + kDecodeSteps,
            "Capacity must cover positions 2052..2055");
    return args;
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  probe_prod_l3 --lane direct --model DIR --capacity 4096 "
           "--output direct.bin [--prompt-file FILE]\n"
        << "  probe_prod_l3 --lane graph  --model DIR --capacity 4096 "
           "--output graph.bin  [--prompt-file FILE]\n"
        << "  probe_prod_l3 --compare direct.bin graph.bin\n";
}

void configure_common_environment(int lane) {
    set_env("AILA_Q4E_QSA_LAYER_GRAPH", lane == 0 ? "0" : "1");
    set_env("AILA_INIT_WARMUP", "0");
    set_env("AILA_Q35_PREFILL_CHUNK", "512");
    set_env("AILA_Q4E_ATTN_NF4_Q8", "0");
    set_env("AILA_Q4E_GPU_NF4_Q8", "0");
    set_env("AILA_Q4E_LM_HEAD_Q8", "0");
    set_env("AILA_Q4E_DENSE_GEMV", "1");
    set_env("AILA_QSA_COMPRESSED_K", "1");
    set_env("AILA_QSA_BLOCK_NATIVE", "1");
    set_env("AILA_KV_QUANT", "0");
    set_env("AILA_Q4E_GDN_CHUNK_MODE", "0");
    set_env("AILA_Q4E_LAYER_MAJOR", "0");
    set_env("SYCL_CACHE_PERSISTENT", "0");
}

int run_lane(const Arguments& args) {
    require(std::filesystem::exists(args.model),
            "Model directory does not exist: " + args.model);
    configure_common_environment(args.lane);

    InferenceEngine engine;
    require(engine.init(args.model, args.capacity),
            "InferenceEngine init failed: " + engine.last_error_message());
    const auto& spec = engine.model_spec();
    require(spec.family == ModelFamily::Qwen4Exp,
            "Probe requires the Qwen4Exp production backend");
    const auto& cfg = spec.qwen4exp_text;
    require(cfg.hidden_size == kHidden && cfg.num_attention_heads == 24 &&
                cfg.num_key_value_heads == kKvHeads &&
                cfg.head_dim == kHeadDim &&
                cfg.indexer_kv_heads == 1 &&
                cfg.indexer_head_dim == kIndexerDim &&
                cfg.indexer_compress_ratio == kCompressRatio &&
                cfg.indexer_budget / cfg.indexer_compress_ratio == 512,
            "Unexpected Qwen4Exp L3 geometry");
    require(static_cast<int>(cfg.layer_types.size()) > kLayer &&
                cfg.layer_types[kLayer] == "full_attention",
            "Layer 3 is not full attention");

    const std::string prompt_text = args.prompt_file.empty()
        ? default_prompt_text()
        : load_prompt_text(args.prompt_file);
    const auto token_ids = engine.tokenizer().encode(prompt_text);
    require(static_cast<int>(token_ids.size()) >=
                kPrefixTokens + kDecodeSteps,
            "Prompt must tokenize to at least 2056 tokens");
    std::vector<int> selected_tokens(
        token_ids.begin(),
        token_ids.begin() + kPrefixTokens + kDecodeSteps);
    auto& backend = Qwen4ExpGenerationTestAccess::Backend(engine);
    const int vocab_size = backend.vocab_size();
    for (int token : selected_tokens) {
        require(token >= 0 && token < vocab_size,
                "Tokenizer produced an out-of-range token id");
    }

    ForwardOptions prompt_options;
    prompt_options.snapshot_policy = SnapshotPolicy::None;
    prompt_options.need_logits = false;
    engine.forward_prompt_tokens(token_ids, 0, kPrefixTokens, 0, false,
                                 prompt_options, true);

    Context& ctx = Qwen4ExpGenerationTestAccess::Ctx(engine);
    auto& session = Qwen4ExpGenerationTestAccess::Session(engine);
    ctx.synchronize();
    require(session.CurrentLength() == kPrefixTokens,
            "Production Session did not reach position 2052");
    auto& cache = session.layer_caches_for_testing().at(kLayer);
    require(cache.num_compressed_blocks ==
                kPrefixTokens / kCompressRatio,
            "L3 compressed-block prefix count mismatch");

    const SentinelState sentinels =
        seed_sentinels(ctx, cache, session.Capacity());

    LaneDump dump;
    dump.lane = args.lane;
    dump.capacity = session.Capacity();
    dump.prefix_len = kPrefixTokens;
    dump.steps = kDecodeSteps;
    dump.vocab_size = vocab_size;
    dump.token_hash = fnv1a_ints(selected_tokens);
    dump.prompt_bytes = prompt_text.size();
    dump.tokens = selected_tokens;
    dump.records.reserve(kDecodeSteps);

    Tensor token_device =
        Tensor::allocate(ctx, {1}, dnnl::memory::data_type::s32);
    for (int step = 0; step < kDecodeSteps; ++step) {
        const int position = kPrefixTokens + step;
        require(session.CurrentLength() == position,
                "Production Session position drifted before decode step");
        const int token = selected_tokens[position];
        ctx.memcpy_h2d(token_device.data(), &token, sizeof(token));
        ForwardOptions decode_options;
        decode_options.snapshot_policy = SnapshotPolicy::None;
        decode_options.need_logits = false;
        backend.Forward(ctx, session, token_device.data_as<int>(), 1,
                        decode_options);
        ctx.synchronize();
        require(session.CurrentLength() == position + 1,
                "Production Session did not advance by one token");
        StepRecord record =
            capture_step(ctx, backend, session, position, sentinels);
        require_lane_state(dump, record, args.lane);
        std::cout << "[PROD L3 STEP] lane="
                  << (args.lane == 0 ? "direct" : "graph")
                  << " pos=" << position
                  << " blocks=" << record.session_blocks
                  << " graph_nodes=" << record.graph_nodes
                  << " graph_seq=" << record.graph_last_sequence
                  << " sentinel_ok=" << record.sentinel_ok << '\n';
        dump.records.push_back(std::move(record));
    }

    write_dump(args.output, dump);
    write_summary(args.output + ".txt", dump, args.model, args.prompt_file);
    std::cout << "[PROD L3 RESULT] lane="
              << (args.lane == 0 ? "direct" : "graph")
              << " capacity=" << dump.capacity
              << " prefix=" << dump.prefix_len
              << " positions=" << kPrefixTokens << ".."
              << kPrefixTokens + kDecodeSteps - 1
              << " token_hash=0x" << std::hex << dump.token_hash
              << std::dec << " output=" << args.output << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        if (args.mode == Arguments::Mode::Help) {
            print_usage();
            return 0;
        }
        if (args.mode == Arguments::Mode::Compare) {
            return compare_dumps(args.direct_file, args.graph_file);
        }
        return run_lane(args);
    } catch (const std::exception& error) {
        std::cerr << "[PROD L3 ERROR] " << error.what() << '\n';
        return 1;
    }
}
