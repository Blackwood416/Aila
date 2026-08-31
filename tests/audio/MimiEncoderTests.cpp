#include "audio/MimiEncoder.hpp"
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
    std::cout << "[Test] MimiEncoder: golden codes alignment..." << std::endl;
    aila::audio::MimiEncoder enc;
    std::string err;
    std::string model_safetensors = "models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/model.safetensors";
    if (!enc.loadWeights(model_safetensors, &err)) {
        std::cerr << "Failed to load weights: " << err << std::endl;
        g_ctx.failed++;
        return;
    }

    // First test with the exact 24kHz samples
    std::ifstream bin_file("tmp/icl_oracle/ref_test_wav24k.bin", std::ios::binary);
    if (bin_file.is_open()) {
        std::vector<float> wav24k(43200);
        bin_file.read(reinterpret_cast<char*>(wav24k.data()), 43200 * sizeof(float));
        TTSReferenceCodes exact_codes;
        bool ok_exact = enc.encode(wav24k.data(), wav24k.size(), exact_codes, &err);
        EXPECT_TRUE(ok_exact);
        EXPECT_EQ(exact_codes.frames, 23);
        int32_t expected_f0[16] = {
            1995, 224, 1114, 1258, 156, 241, 800, 1701, 1857, 1919, 1381, 1716, 841, 1582, 971, 1796
        };
        for (int i = 0; i < 16; ++i) {
            EXPECT_EQ(exact_codes.codes[i], expected_f0[i]);
        }
    } else {
        std::cerr << "Could not open tmp/icl_oracle/ref_test_wav24k.bin" << std::endl;
    }

    // Now test encodeFromFile
    TTSReferenceCodes codes;
    bool ok = enc.encodeFromFile("ref_test.wav", codes, &err);
    if (!ok) {
        std::cerr << "Failed to encode ref_test.wav: " << err << std::endl;
        g_ctx.failed++;
        return;
    }

    EXPECT_EQ(codes.frames, 23);
    EXPECT_EQ(codes.codebooks, 16);
    EXPECT_EQ(codes.codes.size(), 23 * 16);

    // Test English wav with exact 24kHz samples
    std::ifstream bin_en("tmp/icl_oracle/en_test_wav24k.bin", std::ios::binary);
    if (bin_en.is_open()) {
        std::vector<float> wav24k_en(50880);
        bin_en.read(reinterpret_cast<char*>(wav24k_en.data()), 50880 * sizeof(float));
        TTSReferenceCodes exact_en_codes;
        bool ok_exact = enc.encode(wav24k_en.data(), wav24k_en.size(), exact_en_codes, &err);
        EXPECT_TRUE(ok_exact);
        EXPECT_EQ(exact_en_codes.frames, 27);
        int32_t expected_en_f0[16] = {
            1995, 802, 1114, 1151, 1352, 1953, 970, 1879, 480, 86, 743, 1790, 593, 1374, 1747, 178
        };
        for (int i = 0; i < 16; ++i) {
            EXPECT_EQ(exact_en_codes.codes[i], expected_en_f0[i]);
        }
    }

    // Test English wav via encodeFromFile
    TTSReferenceCodes en_codes;
    ok = enc.encodeFromFile("This is an English test.wav", en_codes, &err);
    if (!ok) {
        std::cerr << "Failed to encode English wav: " << err << std::endl;
        g_ctx.failed++;
        return;
    }
    EXPECT_EQ(en_codes.frames, 27);
    EXPECT_EQ(en_codes.codebooks, 16);
}

int main() {
    test_error_handling();
    test_golden_alignment();

    std::cout << "AilaMimiEncoderTests: " << g_ctx.passed << " passed, " << g_ctx.failed << " failed" << std::endl;
    return g_ctx.failed == 0 ? 0 : 1;
}
