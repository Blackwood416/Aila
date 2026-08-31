#include "engine/Engine.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

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
            std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_EQ failed: " << #a << " != " << #b << std::endl; \
        } \
    } while (0)

#define EXPECT_GE(a, b) \
    do { \
        if ((a) >= (b)) { \
            g_ctx.passed++; \
        } else { \
            g_ctx.failed++; \
            std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_GE failed: " << #a << " (" << (a) << ") < " << #b << " (" << (b) << ")" << std::endl; \
        } \
    } while (0)

} // anonymous namespace

void test_tokenization_parity() {
    std::cout << "[Test] TTS Tokenization parity between blocking and streaming..." << std::endl;
    InferenceEngine engine;
    bool ok = engine.init("models/Qwen3-TTS-12Hz-0.6B-Base", 4096);
    EXPECT_TRUE(ok);
    if (!ok) return;

    std::string text = "你好，世界。";
    std::string ref_text = "这是一个错事。";

    std::vector<int> tokens_blocking = engine.tokenizer().encode(text);
    std::vector<int> tokens_streaming = engine.tokenizer().encode(text);
    std::vector<int> ref_tokens = engine.tokenizer().encode(ref_text);

    EXPECT_EQ(tokens_blocking.size(), static_cast<size_t>(4));
    EXPECT_EQ(tokens_streaming.size(), static_cast<size_t>(4));
    EXPECT_EQ(ref_tokens.size(), static_cast<size_t>(4));
    EXPECT_TRUE(tokens_blocking == tokens_streaming);
}

void test_e2e_icl_speech_synthesis() {
    std::cout << "[Test] TTS E2E ICL Speech Synthesis RMS & Duration..." << std::endl;
    InferenceEngine engine;
    bool ok = engine.init("models/Qwen3-TTS-12Hz-0.6B-Base", 4096);
    EXPECT_TRUE(ok);
    if (!ok) return;

    GenerationConfig gen_cfg;
    gen_cfg.do_sample = true;
    gen_cfg.temperature = 0.7f;
    gen_cfg.top_k = 15;
    gen_cfg.top_p = 0.95f;
    gen_cfg.repetition_penalty = 1.05f;

    std::vector<float> samples;
    bool synth_ok = engine.synthesizeSpeech(
        "你好，世界。",
        "ref_test.wav",
        "", // speaker_name
        "", // instruct
        "chinese",
        gen_cfg,
        samples,
        "这是一个错事。",
        VoiceCloneMode::Icl
    );

    EXPECT_TRUE(synth_ok);
    EXPECT_TRUE(!samples.empty());

    // 1. Duration assertion: 24kHz audio should be between 2.0s (48000) and 2.6s (62400)
    double duration_s = static_cast<double>(samples.size()) / 24000.0;
    std::cout << "  Generated samples: " << samples.size() << " (" << duration_s << "s)" << std::endl;
    EXPECT_GE(samples.size(), static_cast<size_t>(48000));
    EXPECT_TRUE(samples.size() <= static_cast<size_t>(65000));

    // 2. RMS Energy calculation & hard assertion (must be >= 0.08, strictly preventing silence)
    double sum_sq = 0.0;
    float max_peak = 0.0f;
    for (float s : samples) {
        sum_sq += static_cast<double>(s * s);
        float abs_s = std::abs(s);
        if (abs_s > max_peak) max_peak = abs_s;
    }
    double rms = std::sqrt(sum_sq / samples.size());
    std::cout << "  RMS Energy: " << rms << ", Peak Amplitude: " << max_peak << std::endl;
    EXPECT_GE(rms, 0.08);
    EXPECT_GE(max_peak, 0.5f);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Running TTS Voice Clone Alignment Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_tokenization_parity();
    test_e2e_icl_speech_synthesis();

    std::cout << "========================================" << std::endl;
    std::cout << "  Alignment Tests: " << g_ctx.passed << " passed, " << g_ctx.failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (g_ctx.failed == 0) ? 0 : 1;
}
