#include "engine/Engine.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <mutex>

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

void test_tokenization_helper() {
    std::cout << "[Test] TTS Tokenization helper purity..." << std::endl;
    InferenceEngine engine;
    bool ok = engine.init("models/Qwen3-TTS-12Hz-0.6B-Base", 4096);
    EXPECT_TRUE(ok);
    if (!ok) return;

    std::string text = "你好，世界。";
    std::string ref_text = "这是一个错事。";

    std::vector<int> tokens = engine.tokenize_tts_text(text);
    std::vector<int> ref_tokens = engine.tokenize_tts_text(ref_text);

    EXPECT_EQ(tokens.size(), static_cast<size_t>(4));
    EXPECT_EQ(ref_tokens.size(), static_cast<size_t>(4));

    // Ensure no ChatML tags (151644, 151645) are embedded
    for (int t : tokens) {
        EXPECT_TRUE(t != 151644 && t != 151645);
    }
    for (int t : ref_tokens) {
        EXPECT_TRUE(t != 151644 && t != 151645);
    }
}

void test_blocking_streaming_e2e_parity_and_rms() {
    std::cout << "[Test] TTS E2E Blocking vs Streaming parity and RMS validation..." << std::endl;
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
    gen_cfg.use_fixed_seed = true;
    gen_cfg.sampling_seed = 42;

    // 1. Run Blocking ICL
    std::vector<float> blocking_samples;
    bool b_ok = engine.synthesizeSpeech(
        "你好，世界。",
        "ref_test.wav",
        "", // speaker_name
        "", // instruct
        "chinese",
        gen_cfg,
        blocking_samples,
        "这是一个错事。",
        VoiceCloneMode::Icl
    );
    EXPECT_TRUE(b_ok);
    EXPECT_TRUE(!blocking_samples.empty());

    // 2. Run Streaming ICL with fixed seed
    std::vector<float> streaming_samples;
    std::mutex stream_mtx;
    auto worker = engine.synthesizeSpeechStream(
        "你好，世界。",
        "ref_test.wav",
        "", // speaker_name
        "", // instruct
        "chinese",
        gen_cfg,
        [&](const float* chunk, int count) {
            if (count > 0) {
                std::lock_guard<std::mutex> lock(stream_mtx);
                streaming_samples.insert(streaming_samples.end(), chunk, chunk + count);
            }
        },
        4, // batch frames
        "这是一个错事。",
        VoiceCloneMode::Icl
    );
    worker.join();

    EXPECT_TRUE(!streaming_samples.empty());
    EXPECT_EQ(blocking_samples.size(), streaming_samples.size());

    // 3. Verify sample correlation between blocking and streaming
    double dot_prod = 0.0, norm_b = 0.0, norm_s = 0.0;
    for (size_t i = 0; i < blocking_samples.size(); ++i) {
        dot_prod += static_cast<double>(blocking_samples[i]) * static_cast<double>(streaming_samples[i]);
        norm_b += static_cast<double>(blocking_samples[i]) * static_cast<double>(blocking_samples[i]);
        norm_s += static_cast<double>(streaming_samples[i]) * static_cast<double>(streaming_samples[i]);
    }
    double correlation = dot_prod / (std::sqrt(norm_b) * std::sqrt(norm_s));
    std::cout << "  Blocking vs Streaming Cosine Similarity / Correlation: " << correlation << std::endl;
    EXPECT_GE(correlation, 0.95);

    // 4. Duration and RMS validation
    double duration_s = static_cast<double>(blocking_samples.size()) / 24000.0;
    std::cout << "  Output samples: " << blocking_samples.size() << " (" << duration_s << "s)" << std::endl;
    EXPECT_GE(blocking_samples.size(), static_cast<size_t>(48000));
    EXPECT_TRUE(blocking_samples.size() <= static_cast<size_t>(65000));

    double sum_sq = 0.0;
    float max_peak = 0.0f;
    for (float s : blocking_samples) {
        sum_sq += static_cast<double>(s * s);
        float abs_s = std::abs(s);
        if (abs_s > max_peak) max_peak = abs_s;
    }
    double rms = std::sqrt(sum_sq / blocking_samples.size());
    std::cout << "  RMS Energy: " << rms << ", Peak Amplitude: " << max_peak << std::endl;
    EXPECT_GE(rms, 0.08);
    EXPECT_GE(max_peak, 0.5f);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Running TTS Voice Clone Alignment Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_tokenization_helper();
    test_blocking_streaming_e2e_parity_and_rms();

    std::cout << "========================================" << std::endl;
    std::cout << "  Alignment Tests: " << g_ctx.passed << " passed, " << g_ctx.failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (g_ctx.failed == 0) ? 0 : 1;
}
