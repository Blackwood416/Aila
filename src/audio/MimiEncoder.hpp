#pragma once

#include "engine/Types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace aila {
namespace audio {

// Qwen3-TTS 12Hz Speech Tokenizer v2 Encoder.
// Converts 24kHz mono float32 audio waveform into discrete codes [frames, 16].
// Completely native C++ implementation: SEANet causal convs + 8-layer Transformer + Downsampler + Split RVQ.
class MimiEncoder {
public:
    MimiEncoder();
    ~MimiEncoder();

    // Load encoder weights from speech_tokenizer/model.safetensors.
    // Handles weights with prefix "encoder.".
    bool loadWeights(const std::string& safetensorsPath, std::string* error = nullptr);

    bool isLoaded() const { return loaded_; }

    // Encode 24kHz mono float32 PCM samples into discrete reference codes.
    // samples: pointer to contiguous float32 samples in range [-1.0, 1.0]
    // numSamples: number of 24kHz audio samples (must be >= 1024)
    // outCodes: receives [frames, 16] frame-major codes
    bool encode(const float* samples, int numSamples,
                TTSReferenceCodes& outCodes, std::string* error = nullptr);

    // Convenience: load audio file (WAV/MP3/FLAC), resample to 24kHz mono if needed,
    // and encode into reference codes.
    bool encodeFromFile(const std::string& audioPath,
                        TTSReferenceCodes& outCodes, std::string* error = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
};

} // namespace audio
} // namespace aila
