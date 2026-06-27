#pragma once

#include "../core/Context.hpp"
#include "../core/Tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace aila::audio {

struct GpuMelSpectrogramTiming {
    double upload_ms = 0.0;
    double stft_ms = 0.0;
    double norm_ms = 0.0;
    int reused_frames = 0;
    int computed_frames = 0;
    double max_abs_diff = 0.0;
};

class AsrGpuMelCache {
public:
    void reset();

    bool constants_ready = false;
    Tensor hann;
    Tensor dft_cos;
    Tensor dft_sin;
    Tensor mel_filterbank;

    Tensor samples;
    Tensor power_spec;
    Tensor raw_log_mel;
    Tensor normalized_mel;
    Tensor mel_device;
    Tensor mel_view;

    size_t sample_capacity = 0;
    int frame_capacity = 0;
    size_t sample_count = 0;
    int n_frames = 0;
    int n_mels = 128;
};

bool compute_asr_mel_spectrogram_gpu(Context& ctx,
                                     const std::vector<float>& samples_16k,
                                     AsrGpuMelCache& cache,
                                     int& n_frames,
                                     int& actual_frames,
                                     Tensor*& mel_device,
                                     std::string* error = nullptr,
                                     GpuMelSpectrogramTiming* timing = nullptr,
                                     bool validate = false);

}  // namespace aila::audio
