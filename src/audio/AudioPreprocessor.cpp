#include "AudioPreprocessor.hpp"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Integrate dr_wav and dr_mp3
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"

// Pre-computed slaney mel filterbank (matches Python WhisperFeatureExtractor)
#include "mel_fb_data.inc"

namespace {

// Convert Hz to mel scale
inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

std::vector<float> build_mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    int n_freq = n_fft / 2 + 1;
    float mel_low = hz_to_mel(0.0f);
    float mel_high = hz_to_mel(static_cast<float>(sample_rate) / 2.0f);

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        float mel = mel_low + (mel_high - mel_low) * i / (n_mels + 1);
        mel_points[i] = static_cast<float>(std::floor(0.5f + mel_to_hz(mel) * n_fft / sample_rate));
    }

    std::vector<float> filters(n_mels * n_freq, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        for (int f = 0; f < n_freq; ++f) {
            float f_start = mel_points[m];
            float f_center = mel_points[m + 1];
            float f_end = mel_points[m + 2];

            if (f >= f_start && f <= f_center && f_center > f_start) {
                filters[m * n_freq + f] = (f - f_start) / (f_center - f_start);
            } else if (f >= f_center && f <= f_end && f_end > f_center) {
                filters[m * n_freq + f] = (f_end - f) / (f_end - f_center);
            }
        }
    }
    return filters;
}

// Local loaders via dr_libs
bool load_wav_dr(const std::string& path, AudioBuffer& audio, std::string* error) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        if (error) *error = "Failed to init WAV file";
        return false;
    }

    audio.sample_rate = wav.sampleRate;
    audio.channels = wav.channels;
    audio.samples.resize(wav.totalPCMFrameCount * wav.channels);
    
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, audio.samples.data());
    drwav_uninit(&wav);
    return true;
}

bool load_mp3_dr(const std::string& path, AudioBuffer& audio, std::string* error) {
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, path.c_str(), nullptr)) {
        if (error) *error = "Failed to init MP3 file";
        return false;
    }

    audio.sample_rate = mp3.sampleRate;
    audio.channels = mp3.channels;
    
    drmp3_uint64 total_pcm_frames = drmp3_get_pcm_frame_count(&mp3);
    if (total_pcm_frames == 0) {
        std::vector<float> temp_buffer;
        std::vector<float> chunk(4096 * mp3.channels);
        while (true) {
            drmp3_uint64 read_frames = drmp3_read_pcm_frames_f32(&mp3, 4096, chunk.data());
            if (read_frames == 0) break;
            temp_buffer.insert(temp_buffer.end(), chunk.begin(), chunk.begin() + read_frames * mp3.channels);
        }
        audio.samples = std::move(temp_buffer);
    } else {
        audio.samples.resize(total_pcm_frames * mp3.channels);
        drmp3_read_pcm_frames_f32(&mp3, total_pcm_frames, audio.samples.data());
    }

    drmp3_uninit(&mp3);
    return true;
}

bool load_flac_dr(const std::string& path, AudioBuffer& audio, std::string* error) {
    drflac* flac = drflac_open_file(path.c_str(), nullptr);
    if (!flac) {
        if (error) *error = "Failed to open FLAC file";
        return false;
    }

    audio.sample_rate = flac->sampleRate;
    audio.channels = flac->channels;
    audio.samples.resize(flac->totalPCMFrameCount * flac->channels);

    drflac_read_pcm_frames_f32(flac, flac->totalPCMFrameCount, audio.samples.data());
    drflac_close(flac);
    return true;
}

// Subprocess pipe decoding via ffmpeg
bool load_via_ffmpeg(const std::string& path, AudioBuffer& audio, std::string* error) {
    std::string cmd = "ffmpeg -v error -i \"" + path + "\" -f s16le -ac 1 -ar 16000 -";
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif

    if (!pipe) {
        if (error) *error = "Failed to open ffmpeg pipe";
        return false;
    }

    std::vector<float> samples;
    int16_t buffer[4096];
    size_t count;
    while ((count = fread(buffer, sizeof(int16_t), 4096, pipe)) > 0) {
        for (size_t i = 0; i < count; ++i) {
            samples.push_back(static_cast<float>(buffer[i]) / 32768.0f);
        }
    }

#ifdef _WIN32
    int exit_code = _pclose(pipe);
#else
    int exit_code = pclose(pipe);
#endif

    if (exit_code != 0 || samples.empty()) {
        if (error) *error = "ffmpeg failed or executable not found (exit code: " + std::to_string(exit_code) + ")";
        return false;
    }

    audio.sample_rate = 16000;
    audio.channels = 1;
    audio.samples = std::move(samples);
    return true;
}

bool load_wav_from_memory_dr(const uint8_t* data, size_t size, AudioBuffer& audio, std::string* error) {
    drwav wav;
    if (!drwav_init_memory(&wav, data, size, nullptr)) {
        if (error) *error = "Failed to init WAV memory data";
        return false;
    }
    audio.sample_rate = wav.sampleRate;
    audio.channels = wav.channels;
    audio.samples.resize(wav.totalPCMFrameCount * wav.channels);
    drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, audio.samples.data());
    drwav_uninit(&wav);
    return true;
}

bool load_mp3_from_memory_dr(const uint8_t* data, size_t size, AudioBuffer& audio, std::string* error) {
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, data, size, nullptr)) {
        if (error) *error = "Failed to init MP3 memory data";
        return false;
    }
    audio.sample_rate = mp3.sampleRate;
    audio.channels = mp3.channels;
    drmp3_uint64 total_pcm_frames = drmp3_get_pcm_frame_count(&mp3);
    if (total_pcm_frames == 0) {
        std::vector<float> temp_buffer;
        std::vector<float> chunk(4096 * mp3.channels);
        while (true) {
            drmp3_uint64 read_frames = drmp3_read_pcm_frames_f32(&mp3, 4096, chunk.data());
            if (read_frames == 0) break;
            temp_buffer.insert(temp_buffer.end(), chunk.begin(), chunk.begin() + read_frames * mp3.channels);
        }
        audio.samples = std::move(temp_buffer);
    } else {
        audio.samples.resize(total_pcm_frames * mp3.channels);
        drmp3_read_pcm_frames_f32(&mp3, total_pcm_frames, audio.samples.data());
    }
    drmp3_uninit(&mp3);
    return true;
}

bool load_flac_from_memory_dr(const uint8_t* data, size_t size, AudioBuffer& audio, std::string* error) {
    drflac* flac = drflac_open_memory(data, size, nullptr);
    if (!flac) {
        if (error) *error = "Failed to open FLAC memory data";
        return false;
    }
    audio.sample_rate = flac->sampleRate;
    audio.channels = flac->channels;
    audio.samples.resize(flac->totalPCMFrameCount * flac->channels);
    drflac_read_pcm_frames_f32(flac, flac->totalPCMFrameCount, audio.samples.data());
    drflac_close(flac);
    return true;
}

} // anonymous namespace

// ============================================================
// load_audio
// ============================================================

bool load_audio(const std::string& path, AudioBuffer& audio, std::string* error) {
    std::string ext = "";
    size_t idx = path.find_last_of('.');
    if (idx != std::string::npos) {
        ext = path.substr(idx + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    std::string local_error = "";
    if (ext == "mp3") {
        if (load_mp3_dr(path, audio, &local_error)) return true;
    } else if (ext == "wav") {
        if (load_wav_dr(path, audio, &local_error)) return true;
    } else if (ext == "flac") {
        if (load_flac_dr(path, audio, &local_error)) return true;
    }

    // Attempt native format trial if unknown ext
    if (ext != "mp3" && ext != "wav" && ext != "flac") {
        if (load_wav_dr(path, audio, nullptr)) return true;
        if (load_mp3_dr(path, audio, nullptr)) return true;
        if (load_flac_dr(path, audio, nullptr)) return true;
    }

    // Fallback to FFmpeg pipe
    std::string ffmpeg_error = "";
    if (load_via_ffmpeg(path, audio, &ffmpeg_error)) {
        return true;
    }

    if (error) {
        *error = "Local decode failed (" + local_error + ") and FFmpeg fallback failed (" + ffmpeg_error + ")";
    }
    return false;
}

// ============================================================
// load_audio_from_memory
// ============================================================

bool load_audio_from_memory(const uint8_t* data, size_t size, const std::string& format, AudioBuffer& audio, std::string* error) {
    if (!data || size == 0) {
        if (error) *error = "Memory data is empty";
        return false;
    }

    std::string fmt = format;
    std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);

    std::string local_error = "";
    if (fmt == "mp3") {
        if (load_mp3_from_memory_dr(data, size, audio, &local_error)) return true;
    } else if (fmt == "wav") {
        if (load_wav_from_memory_dr(data, size, audio, &local_error)) return true;
    } else if (fmt == "flac") {
        if (load_flac_from_memory_dr(data, size, audio, &local_error)) return true;
    }

    // fallback try all if specific failed or fmt is unknown
    if (load_wav_from_memory_dr(data, size, audio, nullptr)) return true;
    if (load_mp3_from_memory_dr(data, size, audio, nullptr)) return true;
    if (load_flac_from_memory_dr(data, size, audio, nullptr)) return true;

    if (error) {
        *error = "Failed to decode audio memory data (" + local_error + ")";
    }
    return false;
}

// ============================================================
// load_wav (Compatibility wrapper)
// ============================================================

bool load_wav(const std::string& path, WavFile& wav, std::string* error) {
    return load_audio(path, wav, error);
}

// ============================================================
// resample_to_16k (Cubic spline interpolation)
// ============================================================

void resample_to_16k(const std::vector<float>& input, int src_rate, std::vector<float>& output) {
    if (src_rate == 16000) {
        output = input;
        return;
    }

    double ratio = static_cast<double>(src_rate) / 16000.0;
    size_t input_size = input.size();
    size_t output_size = static_cast<size_t>(std::round(input_size / ratio));
    output.resize(output_size);

    auto get_sample = [&](int idx) -> float {
        if (idx < 0) return input[0];
        if (idx >= static_cast<int>(input_size)) return input[input_size - 1];
        return input[idx];
    };

    for (size_t i = 0; i < output_size; ++i) {
        double t = i * ratio;
        int idx = static_cast<int>(std::floor(t));
        double f = t - idx;

        // Cubic spline Hermite interpolation (Catmull-Rom)
        float y0 = get_sample(idx - 1);
        float y1 = get_sample(idx);
        float y2 = get_sample(idx + 1);
        float y3 = get_sample(idx + 2);

        float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
        float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        float a2 = -0.5f * y0 + 0.5f * y2;
        float a3 = y1;

        float val = static_cast<float>(((a0 * f + a1) * f + a2) * f + a3);
        output[i] = val;
    }
}

namespace {
constexpr int kMelFft = 400;
constexpr int kMelHop = 160;
constexpr int kMelBins = 128;
constexpr int kMelFreqs = kMelFft / 2 + 1;

void normalize_log_mel(const std::vector<float>& raw_log_mel,
                       int n_frames,
                       std::vector<float>& normalized) {
    normalized.resize(raw_log_mel.size());
    float mel_max = -1e30f;
    for (float v : raw_log_mel) {
        mel_max = std::max(mel_max, v);
    }
    const float mel_min = mel_max - 8.0f;
    for (size_t i = 0; i < raw_log_mel.size(); ++i) {
        float v = std::max(raw_log_mel[i], mel_min);
        normalized[i] = (v + 4.0f) / 4.0f;
    }
    (void)n_frames;
}

void compute_raw_log_mel_frames(const std::vector<float>& samples_16k,
                                int frame_start,
                                int frame_end,
                                std::vector<float>& raw_log_mel) {
    const int n_fft = 400;
    const int hop_length = 160;
    const int n_mels = 128;
    const int n_freq = n_fft / 2 + 1;  // 201
    const size_t n_samples = samples_16k.size();

    // Hann window (PyTorch default: periodic, cos(2*pi*n/N))
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));

    // Mel filterbank: kMelFbData from embedded inc file (201 freqs x 128 mels)

    // Compute mel spectrogram
    std::vector<float> power_spec(n_freq);
    // Reflect padding: mirrors PyTorch F.pad mode='reflect'
    // Left:  idx < 0        → signal[-idx]     (idx=-200→200, idx=-1→1)
    // Valid: 0 <= idx < N   → signal[idx]
    // Right: idx >= N       → signal[2N-idx-2]  (idx=N→N-2, idx=N+1→N-3,...)
    auto reflect_idx = [&](int idx) -> int {
        if (idx < 0) return -idx;
        if (idx >= static_cast<int>(n_samples)) {
            int over = idx - static_cast<int>(n_samples);
            return static_cast<int>(n_samples) - over - 2;
        }
        return idx;
    };

    for (int frm = frame_start; frm < frame_end; ++frm) {
        power_spec.assign(n_freq, 0.0f);
        // With center=True, frame center is at frm * hop
        int center = frm * hop_length;
        int start = center - n_fft / 2;

        for (int k = 0; k < n_freq; ++k) {
            float real_sum = 0.0f;
            float imag_sum = 0.0f;
            float angle_step = -2.0f * static_cast<float>(M_PI) * k / n_fft;
            for (int n = 0; n < n_fft; ++n) {
                int idx = reflect_idx(start + n);
                // 边界保护：以防反射后仍然越界（在极其短的音频上可能出现）
                if (idx < 0) idx = 0;
                else if (idx >= static_cast<int>(n_samples)) idx = static_cast<int>(n_samples) - 1;

                float sample = samples_16k[idx] * hann[n];
                float angle = angle_step * n;
                real_sum += sample * std::cos(angle);
                imag_sum += sample * std::sin(angle);
            }
            power_spec[k] = real_sum * real_sum + imag_sum * imag_sum;
        }

        for (int m = 0; m < n_mels; ++m) {
            float mel_energy = 0.0f;
            // kMelFbData is (201, 128) = (freq, mel), indexed [f * kMelFbMels + m]
            for (int f = 0; f < n_freq; ++f)
                mel_energy += power_spec[f] * kMelFbData[f * kMelFbMels + m];
            mel_energy = std::max(mel_energy, 1e-10f);
            raw_log_mel[static_cast<size_t>(frm) * n_mels + m] = std::log10(mel_energy);
        }
    }
}
}  // namespace

// ============================================================
// compute_mel_spectrogram
// ============================================================

bool compute_mel_spectrogram(const std::vector<float>& samples_16k,
                             MelSpectrogram& mel,
                             std::string* error,
                             MelSpectrogramTiming* timing) {
    if (timing) {
        *timing = MelSpectrogramTiming{};
    }
    auto stage_start = std::chrono::high_resolution_clock::now();
    auto finish_stage = [&](double& target) {
        const auto now = std::chrono::high_resolution_clock::now();
        target += std::chrono::duration<double, std::milli>(now - stage_start).count();
        stage_start = now;
    };

    const size_t n_samples = samples_16k.size();
    if (n_samples == 0) {
        if (error) *error = "Input samples vector is empty";
        return false;
    }

    // 动态计算帧数。在 PyTorch（center=True）中，帧数为 n_samples / hop_length + 1
    int n_frames = static_cast<int>(n_samples / kMelHop + 1);
    int actual_frames = n_frames;
    std::vector<float> raw_log_mel(static_cast<size_t>(kMelBins) * n_frames);
    compute_raw_log_mel_frames(samples_16k, 0, n_frames, raw_log_mel);
    if (timing) {
        finish_stage(timing->stft_ms);
        timing->computed_frames = n_frames;
    }

    // Whisper normalization: clamp to [max-8, max], then scale
    std::vector<float> mel_full;
    normalize_log_mel(raw_log_mel, n_frames, mel_full);
    if (timing) {
        finish_stage(timing->norm_ms);
    }

    mel.n_frames = n_frames;
    mel.n_mels = kMelBins;
    mel.actual_frames = actual_frames;
    mel.data = std::move(mel_full);

    return true;
}

bool compute_mel_spectrogram_cached(const std::vector<float>& samples_16k,
                                    MelSpectrogram& mel,
                                    MelSpectrogramCache& cache,
                                    std::string* error,
                                    MelSpectrogramTiming* timing,
                                    bool validate) {
    if (timing) {
        *timing = MelSpectrogramTiming{};
    }
    auto stage_start = std::chrono::high_resolution_clock::now();
    auto finish_stage = [&](double& target) {
        const auto now = std::chrono::high_resolution_clock::now();
        target += std::chrono::duration<double, std::milli>(now - stage_start).count();
        stage_start = now;
    };

    const size_t n_samples = samples_16k.size();
    if (n_samples == 0) {
        if (error) *error = "Input samples vector is empty";
        return false;
    }

    const int n_frames = static_cast<int>(n_samples / kMelHop + 1);
    bool can_reuse = cache.sample_count > 0 &&
                     cache.sample_count <= n_samples &&
                     cache.n_frames > 0 &&
                     cache.n_frames <= n_frames &&
                     cache.n_mels == kMelBins &&
                     cache.raw_log_mel.size() ==
                         static_cast<size_t>(cache.n_frames) * kMelBins;
    int recompute_start = 0;
    if (can_reuse) {
        recompute_start = std::max(0, cache.n_frames - 3);
    } else {
        cache.raw_log_mel.clear();
        recompute_start = 0;
    }

    std::vector<float> raw_log_mel(static_cast<size_t>(n_frames) * kMelBins);
    if (can_reuse && recompute_start > 0) {
        const size_t reused_values = static_cast<size_t>(recompute_start) * kMelBins;
        std::copy_n(cache.raw_log_mel.begin(), reused_values, raw_log_mel.begin());
    }

    compute_raw_log_mel_frames(samples_16k, recompute_start, n_frames, raw_log_mel);
    if (timing) {
        finish_stage(timing->stft_ms);
        timing->reused_frames = recompute_start;
        timing->computed_frames = n_frames - recompute_start;
    }

    std::vector<float> mel_full;
    normalize_log_mel(raw_log_mel, n_frames, mel_full);
    if (timing) {
        finish_stage(timing->norm_ms);
    }

    if (validate) {
        MelSpectrogram full_mel;
        MelSpectrogramTiming full_timing;
        if (!compute_mel_spectrogram(samples_16k, full_mel, error, &full_timing)) {
            return false;
        }
        double max_abs_diff = 0.0;
        if (full_mel.data.size() == mel_full.size()) {
            for (size_t i = 0; i < mel_full.size(); ++i) {
                max_abs_diff = std::max(max_abs_diff,
                                        static_cast<double>(std::abs(mel_full[i] - full_mel.data[i])));
            }
        } else {
            max_abs_diff = 1e30;
        }
        if (timing) {
            timing->max_abs_diff = max_abs_diff;
        }
    }

    cache.sample_count = n_samples;
    cache.n_frames = n_frames;
    cache.n_mels = kMelBins;
    cache.raw_log_mel = std::move(raw_log_mel);

    mel.n_frames = n_frames;
    mel.n_mels = kMelBins;
    mel.actual_frames = n_frames;
    mel.data = std::move(mel_full);
    return true;
}

bool save_wav(const std::string& path, const std::vector<float>& samples, unsigned int sample_rate) {
    if (samples.empty()) return false;
    drwav wav;
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = sample_rate;
    format.bitsPerSample = 32;

    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        return false;
    }

    drwav_uint64 frames_written = drwav_write_pcm_frames(&wav, samples.size(), samples.data());
    drwav_uninit(&wav);
    return frames_written == samples.size();
}
