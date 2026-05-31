#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"
#include <vector>
#include <string>

namespace aila {
namespace audio {

// GPU-accelerated ECAPA-TDNN speaker encoder using SYCL kernels.
// Mel spectrogram is computed on CPU; the ECAPA-TDNN forward pass runs on GPU.
//
// Weights are kept as bf16 GPU tensors loaded from safetensors.
class GpuSpeakerEncoder {
public:
    GpuSpeakerEncoder() = default;
    ~GpuSpeakerEncoder() = default;

    // Load speaker_encoder.* weights from safetensors to GPU.
    bool loadWeights(Context& ctx, const std::string& safetensorsPath,
                     std::string* error = nullptr);

    // Extract speaker embedding from pre-computed mel spectrogram (CPU data).
    // mel: [n_mels * n_frames] row-major (same format as CPU SpeakerEncoder output).
    bool extractEmbedding(Context& ctx, const float* mel, int nFrames,
                          std::vector<float>& embedding, std::string* error = nullptr);

    int embeddingDim() const { return embedding_dim_; }
    bool isLoaded() const { return loaded_; }

private:
    static constexpr int kMelBins = 128;
    static constexpr int kHiddenDim = 512;
    static constexpr int kMfaChannels = 1536;
    static constexpr int kBranchDim = 64;

    // GPU weight tensors (bf16). Layouts match ConvOps expectations:
    //   Conv1D weight: [out_ch, in_ch, kernel_size]
    //   Conv1D bias:   [out_ch]

    // conv0: [512, 128, 5]
    Tensor conv0_w_, conv0_b_;

    struct BlockWeights {
        Tensor tdnn1_w, tdnn1_b;          // [512, 512, 1]
        Tensor res2net_w[7], res2net_b[7]; // [64, 64, 3] each
        Tensor tdnn2_w, tdnn2_b;          // [512, 512, 1]
        Tensor se_conv1_w, se_conv1_b;     // [128, 512, 1]
        Tensor se_conv2_w, se_conv2_b;     // [512, 128, 1]
    };
    BlockWeights blk_[3];

    // MFA: [1536, 1536, 1]
    Tensor mfa_w_, mfa_b_;

    // ASP: tdnn [128, 4608, 1], conv [1536, 128, 1]
    Tensor asp_tdnn_w_, asp_tdnn_b_;
    Tensor asp_conv_w_, asp_conv_b_;

    // FC: [1024, 3072, 1]
    Tensor fc_w_, fc_b_;

    int embedding_dim_ = 1024;
    bool loaded_ = false;
};

} // namespace audio
} // namespace aila
