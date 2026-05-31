#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace aila {
namespace audio {

// ECAPA-TDNN speaker encoder for Qwen3TTS voice cloning.
// Self-contained CPU implementation — no SYCL/oneDNN dependency.
//
// Usage:
//   SpeakerEncoder enc;
//   enc.loadWeights("models/.../model.safetensors");
//   std::vector<float> embedding;
//   enc.extractEmbeddingFromFile("reference.wav", embedding);
class SpeakerEncoder {
public:
    SpeakerEncoder() = default;
    ~SpeakerEncoder() = default;

    // Load speaker_encoder.* weights from a safetensors file.
    bool loadWeights(const std::string& safetensorsPath, std::string* error = nullptr);

    // Extract speaker embedding from 24kHz mono float32 samples (normalized to [-1, 1]).
    bool extractEmbedding(const float* samples, int numSamples,
                          std::vector<float>& embedding, std::string* error = nullptr);

    // Convenience: load audio file (wav/mp3/flac), resample to 24kHz mono if needed,
    // extract speaker embedding.
    bool extractEmbeddingFromFile(const std::string& audioPath,
                                  std::vector<float>& embedding, std::string* error = nullptr);

    int embeddingDim() const { return embedding_dim_; }
    bool isLoaded() const { return loaded_; }

    // Compute mel spectrogram (public for use by GPU encoder pipeline)
    bool computeMelSpectrogram(const float* samples, int numSamples,
                               std::vector<float>& mel, int& nFrames);

private:
    static constexpr int kMelBins = 128;
    static constexpr int kHiddenDim = 512;
    static constexpr int kRes2NetScale = 8;
    static constexpr int kBranchDim = 64;  // kHiddenDim / kRes2NetScale
    static constexpr int kMfaChannels = 1536;  // 3 * kHiddenDim

    // ECAPA-TDNN forward pass (CPU).
    void forward(const float* mel, int nFrames, float* embedding);

    // --- helper ops ---
    // Reflect-pad time dim: input[C][T] -> output[C][T + 2*pad]
    static void reflectPad1d(const float* input, int C, int T, int pad, std::vector<float>& output);
    // Conv1D: out = conv(input) + bias, then optional ReLU.
    // weight layout: [outC][inC][K]
    static void conv1d(const float* input, int T, int inC,
                       const float* weight, const float* bias, int outC, int K,
                       int stride, int pad, int dilation, bool relu,
                       std::vector<float>& output);
    // Global average pool over time: [C][T] -> [C][1]
    static void globalAvgPool1d(const float* input, int C, int T, float* output);
    // Softmax over time: [C][T] computed per-channel
    static void softmax1d(float* data, int C, int T);

    // --- weights (CPU float32) ---
    // conv0: [512, 128, 5]
    std::vector<float> conv0_w_, conv0_b_;

    struct BlockWeights {
        // tdnn1: [512, 512, 1]
        std::vector<float> tdnn1_w, tdnn1_b;
        // Res2Net: 7 convs, each [64, 64, 3]
        std::vector<float> res2net_w[7], res2net_b[7];
        // tdnn2: [512, 512, 1]
        std::vector<float> tdnn2_w, tdnn2_b;
        // SE: conv1 [128, 512, 1], conv2 [512, 128, 1]
        std::vector<float> se_conv1_w, se_conv1_b;
        std::vector<float> se_conv2_w, se_conv2_b;
    };
    BlockWeights blk_[3];

    // MFA: [1536, 1536, 1]
    std::vector<float> mfa_w_, mfa_b_;

    // ASP: tdnn [128, 4608, 1], conv [1536, 128, 1]
    std::vector<float> asp_tdnn_w_, asp_tdnn_b_;
    std::vector<float> asp_conv_w_, asp_conv_b_;

    // FC: [1024, 3072, 1]
    std::vector<float> fc_w_, fc_b_;

    int embedding_dim_ = 1024;
    bool loaded_ = false;
};

} // namespace audio
} // namespace aila
