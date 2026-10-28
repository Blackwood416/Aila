#define _CRT_SECURE_NO_WARNINGS
#include "models/Qwen4ExpExpertReader.hpp"
#include "utils/SafeTensors.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

void expect(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

static std::string resolve_qwen4exp_model_dir() {
    const char* env_dir = std::getenv("AILA_QWEN4EXP_MODEL_DIR");
    if (env_dir && fs::exists(env_dir)) return env_dir;
    const std::string candidate_d = "D:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (std::filesystem::exists(candidate_d)) return candidate_d;
    const std::string candidate_c = "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_c)) return candidate_c;
    const std::string candidate_f = "F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_f)) return candidate_f;
    return "";
}

// Generates a mock safetensors shard file on disk for isolated testing
struct SyntheticSafetensorsShard {
    fs::path temp_dir;
    fs::path shard_path;
    size_t gu_packed_bytes = 16384;
    size_t gu_absmax_bytes = 512;
    size_t dn_packed_bytes = 8192;
    size_t dn_absmax_bytes = 256;
    size_t slot_bytes = 16384 + 512 + 8192 + 256;
    int num_experts = 4;

    std::vector<uint8_t> gu_packed_data;
    std::vector<uint8_t> gu_absmax_data;
    std::vector<uint8_t> dn_packed_data;
    std::vector<uint8_t> dn_absmax_data;

    SyntheticSafetensorsShard() {
        temp_dir = fs::temp_directory_path() / ("aila_reader_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
        fs::create_directories(temp_dir);
        shard_path = temp_dir / "model.safetensors";

        gu_packed_data.resize(num_experts * gu_packed_bytes);
        gu_absmax_data.resize(num_experts * gu_absmax_bytes);
        dn_packed_data.resize(num_experts * dn_packed_bytes);
        dn_absmax_data.resize(num_experts * dn_absmax_bytes);

        std::mt19937 rng(1337);
        for (auto& b : gu_packed_data) b = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& b : gu_absmax_data) b = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& b : dn_packed_data) b = static_cast<uint8_t>(rng() & 0xFF);
        for (auto& b : dn_absmax_data) b = static_cast<uint8_t>(rng() & 0xFF);

        // Build safetensors json header
        size_t o0 = 0;
        size_t o1 = o0 + gu_packed_data.size();
        size_t o2 = o1 + gu_absmax_data.size();
        size_t o3 = o2 + dn_packed_data.size();
        size_t o4 = o3 + dn_absmax_data.size();

        std::string json = "{"
            "\"model.layers.0.mlp.experts.gate_up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(num_experts * 128) + ",128],\"data_offsets\":[" + std::to_string(o0) + "," + std::to_string(o1) + "]},"
            "\"model.layers.0.mlp.experts.gate_up_proj.weight.absmax\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(gu_absmax_data.size()) + "],\"data_offsets\":[" + std::to_string(o1) + "," + std::to_string(o2) + "]},"
            "\"model.layers.0.mlp.experts.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(num_experts * 256) + ",32],\"data_offsets\":[" + std::to_string(o2) + "," + std::to_string(o3) + "]},"
            "\"model.layers.0.mlp.experts.down_proj.weight.absmax\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(dn_absmax_data.size()) + "],\"data_offsets\":[" + std::to_string(o3) + "," + std::to_string(o4) + "]}"
            "}";

        uint64_t header_len = json.size();
        std::ofstream out(shard_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&header_len), 8);
        out.write(json.data(), json.size());
        out.write(reinterpret_cast<const char*>(gu_packed_data.data()), gu_packed_data.size());
        out.write(reinterpret_cast<const char*>(gu_absmax_data.data()), gu_absmax_data.size());
        out.write(reinterpret_cast<const char*>(dn_packed_data.data()), dn_packed_data.size());
        out.write(reinterpret_cast<const char*>(dn_absmax_data.data()), dn_absmax_data.size());
        out.close();
    }

    ~SyntheticSafetensorsShard() {
        try {
            fs::remove_all(temp_dir);
        } catch (...) {}
    }
};

void test_filespan_population() {
    std::cout << "[Test 1] Testing FileSpan extraction and validity...\n";
    SyntheticSafetensorsShard fixture;

    ModelWeights weights = LoadModelWeightsLazy(fixture.temp_dir.string());
    expect(weights.has_host("model.layers.0.mlp.experts.gate_up_proj.weight"), "gate_up weight not found");
    HostTensorView view = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight");

    expect(view.valid(), "view is invalid");
    expect(view.file_span.valid(), "file_span is invalid");
    expect(view.file_span.shard_id == 0, "shard_id mismatch");
    expect(view.file_span.bytes == fixture.gu_packed_data.size(), "bytes mismatch");
    expect(view.file_span.absolute_offset >= 8, "absolute_offset too small");

    std::cout << "  FileSpan verified: shard_id=" << view.file_span.shard_id
              << " offset=" << view.file_span.absolute_offset
              << " bytes=" << view.file_span.bytes << " [PASSED]\n";
}

void test_mmap_vs_overlapped_readfile_exact_match() {
    std::cout << "[Test 2] Testing exact byte-for-byte match: Mmap vs Overlapped ReadFile...\n";
    SyntheticSafetensorsShard fixture;

    ModelWeights weights = LoadModelWeightsLazy(fixture.temp_dir.string());

    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = fixture.gu_packed_bytes;
    geom.gu_absmax_count = fixture.gu_absmax_bytes;
    geom.gu_absmax_bytes = fixture.gu_absmax_bytes;
    geom.dn_packed_bytes = fixture.dn_packed_bytes;
    geom.dn_absmax_count = fixture.dn_absmax_bytes;
    geom.dn_absmax_bytes = fixture.dn_absmax_bytes;

    geom.gu_packed_offset = 0;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = fixture.slot_bytes;
    geom.compact_nested = true;

    aila::ExpertLayerSource layer_src;
    layer_src.gate_up.packed_weight = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight");
    layer_src.gate_up.absmax = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight.absmax");
    layer_src.down.packed_weight = weights.host_view("model.layers.0.mlp.experts.down_proj.weight");
    layer_src.down.absmax = weights.host_view("model.layers.0.mlp.experts.down_proj.weight.absmax");
    layer_src.valid = true;

    std::vector<aila::ExpertLayerSource> layers = { layer_src };

    auto mmap_reader = aila::CreateMmapExpertReader(geom, layers, false);
    auto readfile_reader = aila::CreateOverlappedReadFileExpertReader(geom, weights.shard_paths(), layers);

    std::vector<uint8_t> mmap_slot(geom.slot_bytes, 0);
    std::vector<uint8_t> readfile_slot(geom.slot_bytes, 0);

    for (int e = 0; e < fixture.num_experts; ++e) {
        std::fill(mmap_slot.begin(), mmap_slot.end(), 0);
        std::fill(readfile_slot.begin(), readfile_slot.end(), 0);

        mmap_reader->read_expert_slice(0, e, mmap_slot.data(), 0, 1);
        readfile_reader->read_expert_slice(0, e, readfile_slot.data(), 0, 1);

        int diff = std::memcmp(mmap_slot.data(), readfile_slot.data(), geom.slot_bytes);
        expect(diff == 0, "mmap vs readfile byte mismatch for expert " + std::to_string(e));
    }

    std::cout << "  All " << fixture.num_experts << " experts match 100% bitwise! [PASSED]\n";
}

void test_sliced_multipart_reads() {
    std::cout << "[Test 3] Testing concurrent sliced reads and exact statistics...\n";
    SyntheticSafetensorsShard fixture;

    ModelWeights weights = LoadModelWeightsLazy(fixture.temp_dir.string());

    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = fixture.gu_packed_bytes;
    geom.gu_absmax_count = fixture.gu_absmax_bytes;
    geom.gu_absmax_bytes = fixture.gu_absmax_bytes;
    geom.dn_packed_bytes = fixture.dn_packed_bytes;
    geom.dn_absmax_count = fixture.dn_absmax_bytes;
    geom.dn_absmax_bytes = fixture.dn_absmax_bytes;

    geom.gu_packed_offset = 0;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = fixture.slot_bytes;
    geom.compact_nested = true;

    aila::ExpertLayerSource layer_src;
    layer_src.gate_up.packed_weight = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight");
    layer_src.gate_up.absmax = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight.absmax");
    layer_src.down.packed_weight = weights.host_view("model.layers.0.mlp.experts.down_proj.weight");
    layer_src.down.absmax = weights.host_view("model.layers.0.mlp.experts.down_proj.weight.absmax");
    layer_src.valid = true;

    std::vector<aila::ExpertLayerSource> layers = { layer_src };
    auto readfile_reader = aila::CreateOverlappedReadFileExpertReader(geom, weights.shard_paths(), layers);

    std::vector<uint8_t> full_slot(geom.slot_bytes, 0);
    std::vector<uint8_t> sliced_slot(geom.slot_bytes, 0);

    readfile_reader->read_expert_slice(0, 3, full_slot.data(), 0, 1);
    auto mmap_reader = aila::CreateMmapExpertReader(geom, layers, false);
    constexpr int nparts = 7; // segment sizes are deliberately not divisible by 7
    constexpr int repeats = 32;
    for (auto* reader : {readfile_reader.get(), mmap_reader.get()}) {
        reader->reset_stats();
        std::vector<std::thread> threads;
        for (int p = 0; p < nparts; ++p) {
            threads.emplace_back([&, p] {
                for (int i = 0; i < repeats; ++i) {
                    reader->read_expert_slice(0, i % 4, sliced_slot.data(), p, nparts);
                }
            });
        }
        for (int i = 0; i < repeats; ++i) (void)reader->stats();
        for (auto& thread : threads) thread.join();
        expect(std::memcmp(full_slot.data(), sliced_slot.data(), geom.slot_bytes) == 0,
               "Concurrent sliced read mismatch vs full read");
        const auto stats = reader->stats();
        expect(stats.read_count == nparts * repeats, "concurrent read count lost updates");
        expect(stats.read_bytes == geom.slot_bytes * repeats, "concurrent byte count must be exact");
        expect(stats.read_ns > 0, "read duration must be positive");
        reader->reset_stats();
        expect(reader->stats().read_count == 0 && reader->stats().read_bytes == 0,
               "quiescent reset must clear statistics");
    }

    std::cout << "  Multi-part sliced reads assembled 100% bitwise identical! [PASSED]\n";
}

void test_first_vs_second_read_latency_comparison() {
    std::cout << "[Test 4] Comparing first-read vs second-read latency on real or synthetic weights...\n";
    const std::string real_dir = resolve_qwen4exp_model_dir();

    if (!real_dir.empty()) {
        std::cout << "  Using real Qwen3.8-Flash-Next model at: " << real_dir << "\n";
        ModelWeights weights = LoadModelWeightsLazy(real_dir);

        aila::ExpertSlotGeometry geom;
        geom.gu_packed_bytes = 2 * 640 * (2560 / 2); // 1638400 B
        geom.gu_absmax_count = 2 * 640 * (2560 / 64); // 51200
        geom.gu_absmax_bytes = 51200;
        geom.dn_packed_bytes = 2560 * (640 / 2); // 819200 B
        geom.dn_absmax_count = 2560 * (640 / 64); // 25600
        geom.dn_absmax_bytes = 25600;
        geom.gu_packed_offset = 0;
        geom.gu_absmax_offset = geom.gu_packed_bytes;
        geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
        geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
        geom.slot_bytes = 2534400;
        geom.compact_nested = true;

        std::vector<aila::ExpertLayerSource> layers(1);
        std::string gu_name = "model.language_model.layers.0.mlp.experts.gate_up_proj.weight";
        std::string gu_absmax = "model.language_model.layers.0.mlp.experts.gate_up_proj.weight.absmax";
        std::string dn_name = "model.language_model.layers.0.mlp.experts.down_proj.weight";
        std::string dn_absmax = "model.language_model.layers.0.mlp.experts.down_proj.weight.absmax";

        if (weights.has_host(gu_name)) {
            layers[0].gate_up.packed_weight = weights.host_view(gu_name);
            layers[0].gate_up.absmax = weights.host_view(gu_absmax);
            layers[0].down.packed_weight = weights.host_view(dn_name);
            layers[0].down.absmax = weights.host_view(dn_absmax);
            layers[0].valid = true;

            auto mmap_reader = aila::CreateMmapExpertReader(geom, layers, false);
            auto rf_reader = aila::CreateOverlappedReadFileExpertReader(geom, weights.shard_paths(), layers);

            std::vector<uint8_t> buf1(geom.slot_bytes, 0);
            std::vector<uint8_t> buf2(geom.slot_bytes, 0);

            // Cold read (expert 10)
            auto t0 = std::chrono::high_resolution_clock::now();
            rf_reader->read_expert_slice(0, 10, buf1.data(), 0, 1);
            auto t1 = std::chrono::high_resolution_clock::now();
            double rf_first_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Warm read (expert 10 again)
            t0 = std::chrono::high_resolution_clock::now();
            rf_reader->read_expert_slice(0, 10, buf2.data(), 0, 1);
            t1 = std::chrono::high_resolution_clock::now();
            double rf_second_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Mmap read (expert 10)
            std::vector<uint8_t> buf_mmap(geom.slot_bytes, 0);
            t0 = std::chrono::high_resolution_clock::now();
            mmap_reader->read_expert_slice(0, 10, buf_mmap.data(), 0, 1);
            t1 = std::chrono::high_resolution_clock::now();
            double mmap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int diff = std::memcmp(buf1.data(), buf_mmap.data(), geom.slot_bytes);
            expect(diff == 0, "Real model mmap vs readfile byte match failed");

            std::cout << "  Real model expert L0.E10 (2.53 MB):"
                      << " ReadFile 1st=" << rf_first_ms << " ms,"
                      << " ReadFile 2nd=" << rf_second_ms << " ms,"
                      << " Mmap=" << mmap_ms << " ms [PASSED]\n";
        }
    } else {
        std::cout << "  (Real model directory not found, benchmarking on synthetic fixture)\n";
        SyntheticSafetensorsShard synth_fixture;
        ModelWeights synth_weights = LoadModelWeightsLazy(synth_fixture.temp_dir.string());
        aila::ExpertSlotGeometry s_geom;
        s_geom.gu_packed_bytes = synth_fixture.gu_packed_bytes;
        s_geom.gu_absmax_count = synth_fixture.gu_absmax_bytes;
        s_geom.gu_absmax_bytes = synth_fixture.gu_absmax_bytes;
        s_geom.dn_packed_bytes = synth_fixture.dn_packed_bytes;
        s_geom.dn_absmax_count = synth_fixture.dn_absmax_bytes;
        s_geom.dn_absmax_bytes = synth_fixture.dn_absmax_bytes;
        s_geom.gu_packed_offset = 0;
        s_geom.gu_absmax_offset = s_geom.gu_packed_bytes;
        s_geom.dn_packed_offset = s_geom.gu_absmax_offset + s_geom.gu_absmax_bytes;
        s_geom.dn_absmax_offset = s_geom.dn_packed_offset + s_geom.dn_packed_bytes;
        s_geom.slot_bytes = synth_fixture.slot_bytes;
        s_geom.compact_nested = true;

        std::vector<aila::ExpertLayerSource> s_layers(1);
        s_layers[0].gate_up.packed_weight = synth_weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight");
        s_layers[0].gate_up.absmax = synth_weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight.absmax");
        s_layers[0].down.packed_weight = synth_weights.host_view("model.layers.0.mlp.experts.down_proj.weight");
        s_layers[0].down.absmax = synth_weights.host_view("model.layers.0.mlp.experts.down_proj.weight.absmax");
        s_layers[0].valid = true;

        auto s_mmap = aila::CreateMmapExpertReader(s_geom, s_layers, false);
        auto s_rf = aila::CreateOverlappedReadFileExpertReader(s_geom, synth_weights.shard_paths(), s_layers);

        std::vector<uint8_t> b1(s_geom.slot_bytes, 0);
        std::vector<uint8_t> b2(s_geom.slot_bytes, 0);
        std::vector<uint8_t> b_mmap(s_geom.slot_bytes, 0);

        auto t0 = std::chrono::high_resolution_clock::now();
        s_rf->read_expert_slice(0, 0, b1.data(), 0, 1);
        auto t1 = std::chrono::high_resolution_clock::now();
        double rf1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        s_rf->read_expert_slice(0, 0, b2.data(), 0, 1);
        t1 = std::chrono::high_resolution_clock::now();
        double rf2_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        s_mmap->read_expert_slice(0, 0, b_mmap.data(), 0, 1);
        t1 = std::chrono::high_resolution_clock::now();
        double mmap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        expect(std::memcmp(b1.data(), b_mmap.data(), s_geom.slot_bytes) == 0, "Synthetic mmap vs readfile byte match failed");
        std::cout << "  Synthetic expert (25 KB):"
                  << " ReadFile 1st=" << rf1_ms << " ms,"
                  << " ReadFile 2nd=" << rf2_ms << " ms,"
                  << " Mmap=" << mmap_ms << " ms [PASSED]\n";
    }
}

void test_cancellation_and_boundaries() {
    std::cout << "[Test 5] Testing cancellation and boundary defenses...\n";
    SyntheticSafetensorsShard fixture;

    ModelWeights weights = LoadModelWeightsLazy(fixture.temp_dir.string());
    aila::ExpertSlotGeometry geom;
    geom.slot_bytes = fixture.slot_bytes;
    std::vector<aila::ExpertLayerSource> layers;
    auto reader = aila::CreateOverlappedReadFileExpertReader(geom, weights.shard_paths(), layers);

    std::vector<uint8_t> buf(geom.slot_bytes, 0xAA);

    // Invalid layer
    reader->read_expert_slice(-1, 0, buf.data(), 0, 1);
    expect(buf[0] == 0, "Invalid layer should zero output");

    // Invalid expert
    std::fill(buf.begin(), buf.end(), 0xAA);
    reader->read_expert_slice(100, 0, buf.data(), 0, 1);
    expect(buf[0] == 0, "Unbound layer should zero output");

    // Cancel calls
    reader->cancel();
    expect(reader->stats().cancel_calls == 1, "cancel_calls count mismatch");
    expect(reader->is_cancelled(), "reader must report is_cancelled == true");

    // After cancel, reads should short-circuit and not write data
    std::fill(buf.begin(), buf.end(), 0xEE);
    reader->read_expert_slice(0, 0, buf.data(), 0, 1);
    expect(buf[0] == 0xEE, "Cancelled reader should return immediately without writing");

    reader->reset_cancelled();
    expect(!reader->is_cancelled(), "reader must report is_cancelled == false after reset");

    std::cout << "  Boundary handling and cancellation safe! [PASSED]\n";
}

void test_non_compact_unnested_float_absmax() {
    std::cout << "[Test 6] Testing non-compact unnested float32 absmax (Issue 1 defense)...\n";
    fs::path temp_dir = fs::temp_directory_path() / ("aila_unnested_test_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    fs::create_directories(temp_dir);
    fs::path shard_path = temp_dir / "model.safetensors";

    const int num_experts = 2;
    const size_t gu_packed_bytes = 8192;
    const size_t gu_absmax_count = 128;
    const size_t gu_absmax_bytes = gu_absmax_count * sizeof(float);
    const size_t dn_packed_bytes = 4096;
    const size_t dn_absmax_count = 64;
    const size_t dn_absmax_bytes = dn_absmax_count * sizeof(float);
    const size_t slot_bytes = gu_packed_bytes + gu_absmax_bytes + dn_packed_bytes + dn_absmax_bytes;

    std::vector<uint8_t> gu_packed(num_experts * gu_packed_bytes, 0x33);
    std::vector<float> gu_absmax(num_experts * gu_absmax_count, 1.5f);
    std::vector<uint8_t> dn_packed(num_experts * dn_packed_bytes, 0x55);
    std::vector<float> dn_absmax(num_experts * dn_absmax_count, 2.5f);

    size_t o0 = 0;
    size_t o1 = o0 + gu_packed.size();
    size_t o2 = o1 + gu_absmax.size() * sizeof(float);
    size_t o3 = o2 + dn_packed.size();
    size_t o4 = o3 + dn_absmax.size() * sizeof(float);

    std::string json = "{"
        "\"model.layers.0.mlp.experts.gate_up_proj.weight\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(num_experts * 64) + ",128],\"data_offsets\":[" + std::to_string(o0) + "," + std::to_string(o1) + "]},"
        "\"model.layers.0.mlp.experts.gate_up_proj.weight.absmax\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(gu_absmax.size()) + "],\"data_offsets\":[" + std::to_string(o1) + "," + std::to_string(o2) + "]},"
        "\"model.layers.0.mlp.experts.down_proj.weight\":{\"dtype\":\"U8\",\"shape\":[" + std::to_string(num_experts * 128) + ",32],\"data_offsets\":[" + std::to_string(o2) + "," + std::to_string(o3) + "]},"
        "\"model.layers.0.mlp.experts.down_proj.weight.absmax\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(dn_absmax.size()) + "],\"data_offsets\":[" + std::to_string(o3) + "," + std::to_string(o4) + "]}"
        "}";

    uint64_t header_len = json.size();
    std::ofstream out(shard_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_len), 8);
    out.write(json.data(), json.size());
    out.write(reinterpret_cast<const char*>(gu_packed.data()), gu_packed.size());
    out.write(reinterpret_cast<const char*>(gu_absmax.data()), gu_absmax.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(dn_packed.data()), dn_packed.size());
    out.write(reinterpret_cast<const char*>(dn_absmax.data()), dn_absmax.size() * sizeof(float));
    out.close();

    ModelWeights weights = LoadModelWeightsLazy(temp_dir.string());

    aila::ExpertSlotGeometry geom;
    geom.gu_packed_bytes = gu_packed_bytes;
    geom.gu_absmax_count = gu_absmax_count;
    geom.gu_absmax_bytes = gu_absmax_bytes;
    geom.dn_packed_bytes = dn_packed_bytes;
    geom.dn_absmax_count = dn_absmax_count;
    geom.dn_absmax_bytes = dn_absmax_bytes;
    geom.gu_packed_offset = 0;
    geom.gu_absmax_offset = geom.gu_packed_bytes;
    geom.dn_packed_offset = geom.gu_absmax_offset + geom.gu_absmax_bytes;
    geom.dn_absmax_offset = geom.dn_packed_offset + geom.dn_packed_bytes;
    geom.slot_bytes = slot_bytes;
    geom.compact_nested = false;

    aila::ExpertLayerSource layer_src;
    layer_src.gate_up.packed_weight = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight");
    layer_src.gate_up.absmax = weights.host_view("model.layers.0.mlp.experts.gate_up_proj.weight.absmax");
    layer_src.gate_up.quant_state.nested = false;
    layer_src.down.packed_weight = weights.host_view("model.layers.0.mlp.experts.down_proj.weight");
    layer_src.down.absmax = weights.host_view("model.layers.0.mlp.experts.down_proj.weight.absmax");
    layer_src.down.quant_state.nested = false;
    layer_src.valid = true;

    std::vector<aila::ExpertLayerSource> layers = { layer_src };
    auto mmap_reader = aila::CreateMmapExpertReader(geom, layers, false);
    auto rf_reader = aila::CreateOverlappedReadFileExpertReader(geom, weights.shard_paths(), layers);

    std::vector<uint8_t> mmap_buf(slot_bytes, 0);
    std::vector<uint8_t> rf_buf(slot_bytes, 0);

    for (int e = 0; e < num_experts; ++e) {
        mmap_reader->read_expert_slice(0, e, mmap_buf.data(), 0, 1);
        rf_reader->read_expert_slice(0, e, rf_buf.data(), 0, 1);
        expect(std::memcmp(mmap_buf.data(), rf_buf.data(), slot_bytes) == 0,
               "Unnested float absmax byte mismatch for expert " + std::to_string(e));
    }

    try { fs::remove_all(temp_dir); } catch (...) {}
    std::cout << "  Non-compact unnested float32 absmax reader verified! [PASSED]\n";
}

} // namespace

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "  Qwen4Exp ExpertReader Unit Tests (H4) \n";
        std::cout << "========================================\n";

        test_filespan_population();
        test_mmap_vs_overlapped_readfile_exact_match();
        test_sliced_multipart_reads();
        test_first_vs_second_read_latency_comparison();
        test_cancellation_and_boundaries();
        test_non_compact_unnested_float_absmax();

        std::cout << "========================================\n";
        std::cout << "  All H4 ExpertReader Tests PASSED!     \n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
