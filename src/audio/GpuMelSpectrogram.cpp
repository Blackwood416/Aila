#include "GpuMelSpectrogram.hpp"

#include "AudioPreprocessor.hpp"
#include "mel_fb_data.inc"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sycl/sycl.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aila::audio {
namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kAsrFft = 400;
constexpr int kAsrHop = 160;
constexpr int kAsrMels = 128;
constexpr int kAsrFreqs = kAsrFft / 2 + 1;

double elapsed_ms(std::chrono::high_resolution_clock::time_point start,
                  std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void ensure_constants(Context& ctx, AsrGpuMelCache& cache) {
    if (cache.constants_ready) {
        return;
    }

    std::vector<float> hann(kAsrFft);
    std::vector<float> dft_cos(static_cast<size_t>(kAsrFreqs) * kAsrFft);
    std::vector<float> dft_sin(static_cast<size_t>(kAsrFreqs) * kAsrFft);
    for (int n = 0; n < kAsrFft; ++n) {
        hann[n] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * n / kAsrFft));
    }
    for (int k = 0; k < kAsrFreqs; ++k) {
        for (int n = 0; n < kAsrFft; ++n) {
            const float angle = -2.0f * static_cast<float>(M_PI) * k * n / kAsrFft;
            dft_cos[static_cast<size_t>(k) * kAsrFft + n] = std::cos(angle);
            dft_sin[static_cast<size_t>(k) * kAsrFft + n] = std::sin(angle);
        }
    }

    cache.hann = Tensor::allocate(ctx, {kAsrFft}, dnnl::memory::data_type::f32);
    cache.dft_cos = Tensor::allocate(ctx, {kAsrFreqs, kAsrFft}, dnnl::memory::data_type::f32);
    cache.dft_sin = Tensor::allocate(ctx, {kAsrFreqs, kAsrFft}, dnnl::memory::data_type::f32);
    cache.mel_filterbank = Tensor::allocate(ctx, {kAsrFreqs, kAsrMels}, dnnl::memory::data_type::f32);

    ctx.queue().memcpy(cache.hann.data(), hann.data(), hann.size() * sizeof(float));
    ctx.queue().memcpy(cache.dft_cos.data(), dft_cos.data(), dft_cos.size() * sizeof(float));
    ctx.queue().memcpy(cache.dft_sin.data(), dft_sin.data(), dft_sin.size() * sizeof(float));
    ctx.queue().memcpy(cache.mel_filterbank.data(), kMelFbData,
                       static_cast<size_t>(kMelFbFreqs) * kMelFbMels * sizeof(float));
    ctx.queue().wait_and_throw();

    cache.constants_ready = true;
}

void ensure_sample_capacity(Context& ctx, AsrGpuMelCache& cache, size_t sample_count) {
    if (cache.samples.valid() && cache.sample_capacity >= sample_count) {
        return;
    }
    cache.sample_capacity = sample_count + 16000;
    cache.samples = Tensor::allocate(ctx, {static_cast<int64_t>(cache.sample_capacity)},
                                     dnnl::memory::data_type::f32);
}

void ensure_frame_capacity(Context& ctx,
                           AsrGpuMelCache& cache,
                           int n_frames,
                           int preserve_frames) {
    if (cache.raw_log_mel.valid() && cache.frame_capacity >= n_frames) {
        return;
    }

    const int new_capacity = n_frames + 128;
    Tensor new_power = Tensor::allocate(ctx, {new_capacity, kAsrFreqs},
                                        dnnl::memory::data_type::f32);
    Tensor new_raw = Tensor::allocate(ctx, {new_capacity, kAsrMels},
                                      dnnl::memory::data_type::f32);
    Tensor new_norm = Tensor::allocate(ctx, {new_capacity, kAsrMels},
                                       dnnl::memory::data_type::f32);
    Tensor new_mel = Tensor::allocate(ctx, {1, kAsrMels, new_capacity});

    if (preserve_frames > 0 && cache.raw_log_mel.valid()) {
        const size_t bytes = static_cast<size_t>(preserve_frames) * kAsrMels * sizeof(float);
        ctx.queue().memcpy(new_raw.data(), cache.raw_log_mel.data(), bytes).wait();
    }

    cache.power_spec = std::move(new_power);
    cache.raw_log_mel = std::move(new_raw);
    cache.normalized_mel = std::move(new_norm);
    cache.mel_device = std::move(new_mel);
    cache.frame_capacity = new_capacity;
}

}  // namespace

void AsrGpuMelCache::reset() {
    sample_count = 0;
    n_frames = 0;
    n_mels = kAsrMels;
}

bool compute_asr_mel_spectrogram_gpu(Context& ctx,
                                     const std::vector<float>& samples_16k,
                                     AsrGpuMelCache& cache,
                                     int& n_frames,
                                     int& actual_frames,
                                     Tensor*& mel_device,
                                     std::string* error,
                                     GpuMelSpectrogramTiming* timing,
                                     bool validate) {
    if (timing) {
        *timing = GpuMelSpectrogramTiming{};
    }
    n_frames = 0;
    actual_frames = 0;
    mel_device = nullptr;

    const size_t sample_count = samples_16k.size();
    if (sample_count == 0) {
        if (error) {
            *error = "Input samples vector is empty";
        }
        return false;
    }

    ensure_constants(ctx, cache);

    const int frames = static_cast<int>(sample_count / kAsrHop + 1);
    bool can_reuse = cache.sample_count > 0 &&
                     cache.sample_count <= sample_count &&
                     cache.n_frames > 0 &&
                     cache.n_frames <= frames &&
                     cache.n_mels == kAsrMels &&
                     cache.raw_log_mel.valid();
    int recompute_start = can_reuse ? std::max(0, cache.n_frames - 3) : 0;
    ensure_sample_capacity(ctx, cache, sample_count);
    ensure_frame_capacity(ctx, cache, frames, recompute_start);

    auto upload_start = std::chrono::high_resolution_clock::now();
    ctx.queue().memcpy(cache.samples.data(), samples_16k.data(), sample_count * sizeof(float));
    if (timing) {
        ctx.synchronize();
        const auto upload_end = std::chrono::high_resolution_clock::now();
        timing->upload_ms += elapsed_ms(upload_start, upload_end);
    }

    auto stft_start = std::chrono::high_resolution_clock::now();
    const int compute_frames = frames - recompute_start;
    float* samples = cache.samples.data_as<float>();
    float* hann = cache.hann.data_as<float>();
    float* dft_cos = cache.dft_cos.data_as<float>();
    float* dft_sin = cache.dft_sin.data_as<float>();
    float* power = cache.power_spec.data_as<float>();
    float* raw = cache.raw_log_mel.data_as<float>();
    float* mel_fb = cache.mel_filterbank.data_as<float>();
    float* normalized = cache.normalized_mel.data_as<float>();
    bf16* mel_bf16 = cache.mel_device.data_as<bf16>();
    const int sample_count_i = static_cast<int>(sample_count);

    if (compute_frames > 0) {
        ctx.queue().parallel_for(
            sycl::range<2>(static_cast<size_t>(compute_frames), static_cast<size_t>(kAsrFreqs)),
            [=](sycl::id<2> idx) {
                const int local_frame = static_cast<int>(idx[0]);
                const int freq = static_cast<int>(idx[1]);
                const int frame = recompute_start + local_frame;
                const int center = frame * kAsrHop;
                const int start = center - kAsrFft / 2;
                float real_sum = 0.0f;
                float imag_sum = 0.0f;
                for (int n = 0; n < kAsrFft; ++n) {
                    int sample_idx = start + n;
                    if (sample_idx < 0) {
                        sample_idx = -sample_idx;
                    } else if (sample_idx >= sample_count_i) {
                        int over = sample_idx - sample_count_i;
                        sample_idx = sample_count_i - over - 2;
                    }
                    if (sample_idx < 0) {
                        sample_idx = 0;
                    } else if (sample_idx >= sample_count_i) {
                        sample_idx = sample_count_i - 1;
                    }
                    const float sample = samples[sample_idx] * hann[n];
                    const size_t table_idx = static_cast<size_t>(freq) * kAsrFft + n;
                    real_sum += sample * dft_cos[table_idx];
                    imag_sum += sample * dft_sin[table_idx];
                }
                power[static_cast<size_t>(frame) * kAsrFreqs + freq] =
                    real_sum * real_sum + imag_sum * imag_sum;
            });

        ctx.queue().parallel_for(
            sycl::range<2>(static_cast<size_t>(compute_frames), static_cast<size_t>(kAsrMels)),
            [=](sycl::id<2> idx) {
                const int local_frame = static_cast<int>(idx[0]);
                const int mel = static_cast<int>(idx[1]);
                const int frame = recompute_start + local_frame;
                float energy = 0.0f;
                for (int freq = 0; freq < kAsrFreqs; ++freq) {
                    energy += power[static_cast<size_t>(frame) * kAsrFreqs + freq] *
                              mel_fb[static_cast<size_t>(freq) * kAsrMels + mel];
                }
                energy = sycl::fmax(energy, 1e-10f);
                raw[static_cast<size_t>(frame) * kAsrMels + mel] = sycl::log10(energy);
            });
    }

    if (timing) {
        ctx.synchronize();
        const auto stft_end = std::chrono::high_resolution_clock::now();
        timing->stft_ms += elapsed_ms(stft_start, stft_end);
        timing->reused_frames = recompute_start;
        timing->computed_frames = compute_frames;
    }

    auto norm_start = std::chrono::high_resolution_clock::now();
    ctx.queue().single_task([=]() {
        const int total = frames * kAsrMels;
        float mel_max = -3.402823466e+38F;
        for (int i = 0; i < total; ++i) {
            mel_max = sycl::fmax(mel_max, raw[i]);
        }
        const float mel_min = mel_max - 8.0f;
        for (int frame = 0; frame < frames; ++frame) {
            for (int mel = 0; mel < kAsrMels; ++mel) {
                const size_t row_major = static_cast<size_t>(frame) * kAsrMels + mel;
                float v = sycl::fmax(raw[row_major], mel_min);
                v = (v + 4.0f) / 4.0f;
                normalized[row_major] = v;
                mel_bf16[static_cast<size_t>(mel) * frames + frame] = bf16(v);
            }
        }
    });
    if (timing) {
        ctx.synchronize();
        const auto norm_end = std::chrono::high_resolution_clock::now();
        timing->norm_ms += elapsed_ms(norm_start, norm_end);
    }

    if (validate) {
        MelSpectrogram cpu_mel;
        MelSpectrogramTiming cpu_timing;
        if (!compute_mel_spectrogram(samples_16k, cpu_mel, error, &cpu_timing)) {
            return false;
        }
        std::vector<float> gpu_mel(static_cast<size_t>(frames) * kAsrMels);
        ctx.memcpy_d2h(gpu_mel.data(), cache.normalized_mel.data(),
                       gpu_mel.size() * sizeof(float));
        double max_abs_diff = 0.0;
        if (cpu_mel.data.size() == gpu_mel.size()) {
            for (size_t i = 0; i < gpu_mel.size(); ++i) {
                max_abs_diff = std::max(
                    max_abs_diff,
                    static_cast<double>(std::abs(cpu_mel.data[i] - gpu_mel[i])));
            }
        } else {
            max_abs_diff = 1e30;
        }
        if (timing) {
            timing->max_abs_diff = max_abs_diff;
        }
    }

    cache.sample_count = sample_count;
    cache.n_frames = frames;
    cache.n_mels = kAsrMels;
    n_frames = frames;
    actual_frames = frames;
    cache.mel_view = Tensor::view(ctx, cache.mel_device.data(), {1, kAsrMels, frames});
    mel_device = &cache.mel_view;
    return true;
}

}  // namespace aila::audio
