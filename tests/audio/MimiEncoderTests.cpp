#include "audio/MimiEncoder.hpp"
#include "simdjson.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

namespace {

struct TestContext {
    int passed = 0;
    int failed = 0;
};

TestContext g_ctx;

#define EXPECT_TRUE(cond) \
    do { \
        if (cond) { \
            g_ctx.passed++; \
        } else { \
            g_ctx.failed++; \
            std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_TRUE failed: " << #cond << std::endl; \
        } \
    } while (0)

#define EXPECT_EQ(a, b) \
    do { \
        if ((a) == (b)) { \
            g_ctx.passed++; \
        } else { \
            g_ctx.failed++; \
            std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_EQ failed: " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")" << std::endl; \
        } \
    } while (0)

} // anonymous namespace

void test_error_handling() {
    std::cout << "[Test] MimiEncoder: error handling..." << std::endl;
    aila::audio::MimiEncoder enc;
    TTSReferenceCodes codes;
    std::string err;

    // 1. Encode before loading weights
    std::vector<float> dummy(2048, 0.0f);
    EXPECT_TRUE(!enc.encode(dummy.data(), dummy.size(), codes, &err));

    // 2. Load from invalid path
    EXPECT_TRUE(!enc.loadWeights("invalid_path/model.safetensors", &err));

    // 3. Load actual model weights
    std::string model_safetensors = "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/model.safetensors";
    bool loaded = enc.loadWeights(model_safetensors, &err);
    if (!loaded) {
        std::cerr << "Failed to load " << model_safetensors << ": " << err << std::endl;
    }
    EXPECT_TRUE(loaded);
    EXPECT_TRUE(enc.isLoaded());

    // 4. Short audio (< 1024 samples)
    std::vector<float> short_audio(500, 0.0f);
    EXPECT_TRUE(!enc.encode(short_audio.data(), short_audio.size(), codes, &err));
}

void test_golden_alignment() {
    std::cout << "[Test] MimiEncoder: golden codes 368/368 alignment..." << std::endl;
    aila::audio::MimiEncoder enc;
    std::string err;
    std::string model_safetensors = "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/model.safetensors";
    if (!enc.loadWeights(model_safetensors, &err)) {
        std::cerr << "Failed to load weights: " << err << std::endl;
        g_ctx.failed++;
        return;
    }

    // Load golden json
    std::string golden_path = "tests/audio/data/ref_codes_golden.json";
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto json_err = parser.load(golden_path).get(doc);
    if (json_err) {
        std::cerr << "CRITICAL: Failed to load golden json " << golden_path << ": " << simdjson::error_message(json_err) << std::endl;
        g_ctx.failed++;
        return;
    }

    simdjson::dom::element root_elem;
    if (doc["ref_test.wav"].get(root_elem)) {
        std::cerr << "CRITICAL: Malformed golden json (missing ref_test.wav key)" << std::endl;
        g_ctx.failed++;
        return;
    }

    int64_t expected_frames = 0;
    int64_t expected_codebooks = 0;
    if (root_elem["frames"].get(expected_frames) || root_elem["codebooks"].get(expected_codebooks)) {
        std::cerr << "CRITICAL: Malformed golden json" << std::endl;
        g_ctx.failed++;
        return;
    }
    EXPECT_EQ(expected_frames, 23);
    EXPECT_EQ(expected_codebooks, 16);

    std::vector<int32_t> golden_codes;
    golden_codes.reserve(expected_frames * expected_codebooks);
    for (simdjson::dom::element frame_elem : root_elem["codes"]) {
        for (simdjson::dom::element code_elem : frame_elem) {
            int64_t c = 0;
            if (!code_elem.get(c)) {
                golden_codes.push_back(static_cast<int32_t>(c));
            }
        }
    }
    EXPECT_EQ(golden_codes.size(), 368);

    // 1. Test with exact 24kHz samples (PCM direct)
    std::ifstream bin_file("tmp/icl_oracle/ref_test_wav24k.bin", std::ios::binary);
    if (bin_file.is_open()) {
        std::vector<float> wav24k(43200);
        bin_file.read(reinterpret_cast<char*>(wav24k.data()), 43200 * sizeof(float));
        TTSReferenceCodes exact_codes;
        bool ok_exact = enc.encode(wav24k.data(), wav24k.size(), exact_codes, &err);
        EXPECT_TRUE(ok_exact);
        EXPECT_EQ(exact_codes.frames, 23);
        EXPECT_EQ(exact_codes.codebooks, 16);
        EXPECT_EQ(exact_codes.codes.size(), 368);
        int mismatch_count = 0;
        for (size_t i = 0; i < 368 && i < exact_codes.codes.size() && i < golden_codes.size(); ++i) {
            if (exact_codes.codes[i] != golden_codes[i]) {
                mismatch_count++;
            }
        }
        EXPECT_EQ(mismatch_count, 0);
    } else {
        std::cerr << "Could not open tmp/icl_oracle/ref_test_wav24k.bin" << std::endl;
        g_ctx.failed++;
    }

    // 2. Test encodeFromFile with 32kHz ref_test.wav
    TTSReferenceCodes file_codes;
    bool ok_file = enc.encodeFromFile("ref_test.wav", file_codes, &err);
    if (!ok_file) {
        std::cerr << "Failed to encode ref_test.wav: " << err << std::endl;
        g_ctx.failed++;
        return;
    }
    EXPECT_EQ(file_codes.frames, 23);
    EXPECT_EQ(file_codes.codebooks, 16);
    EXPECT_EQ(file_codes.codes.size(), 368);

    int file_mismatch_count = 0;
    for (size_t i = 0; i < 368 && i < file_codes.codes.size() && i < golden_codes.size(); ++i) {
        if (file_codes.codes[i] != golden_codes[i]) {
            file_mismatch_count++;
        }
    }
    std::cout << "[Test] ref_test.wav (32k -> 24k) mismatch vs golden: " << file_mismatch_count << " / 368" << std::endl;
    EXPECT_EQ(file_mismatch_count, 0);
}

int main() {
    test_error_handling();
    test_golden_alignment();

    std::cout << "AilaMimiEncoderTests: " << g_ctx.passed << " passed, " << g_ctx.failed << " failed" << std::endl;
    return g_ctx.failed == 0 ? 0 : 1;
}
