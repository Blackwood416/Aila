#include "AudioPreprocessor.hpp"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
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
// High-quality band-limited Sinc Resampler with Kaiser Window
// ============================================================

static inline double bessel_i0(double x) {
    double ax = std::abs(x);
    if (ax < 3.75) {
        double y = x / 3.75;
        y = y * y;
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
    } else {
        double y = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
    }
}

static inline int64_t calc_gcd(int64_t a, int64_t b) {
    while (b) {
        int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

void resample_audio(const std::vector<float>& input, int src_rate, int dst_rate, std::vector<float>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }
    if (src_rate <= 0 || dst_rate <= 0 || src_rate == dst_rate) {
        output = input;
        return;
    }

    int64_t gcd_val = calc_gcd(src_rate, dst_rate);
    int64_t orig_freq = src_rate / gcd_val;
    int64_t new_freq = dst_rate / gcd_val;

    int lowpass_filter_width = 64;
    double rolloff = 0.99;
    double beta = 14.769656459379492;

    double base_freq = std::min(static_cast<double>(orig_freq), static_cast<double>(new_freq)) * rolloff;
    double i0_beta = bessel_i0(beta);
    double scale = base_freq / orig_freq;

    size_t target_len = static_cast<size_t>(std::ceil(static_cast<double>(input.size()) * dst_rate / src_rate));
    output.resize(target_len);

    double win_radius = lowpass_filter_width * orig_freq / base_freq;
    const double pi = 3.14159265358979323846;

    for (size_t j = 0; j < target_len; ++j) {
        double input_pos = static_cast<double>(j) * orig_freq / static_cast<double>(new_freq);
        int64_t k_min = static_cast<int64_t>(std::floor(input_pos - win_radius));
        int64_t k_max = static_cast<int64_t>(std::ceil(input_pos + win_radius));

        double sum_val = 0.0;
        for (int64_t k = k_min; k <= k_max; ++k) {
            double t = (static_cast<double>(k) - input_pos) * (base_freq / orig_freq);
            if (std::abs(t) >= lowpass_filter_width) continue;

            double sinc_val = (std::abs(t) < 1e-9) ? 1.0 : (std::sin(pi * t) / (pi * t));
            double t_rel = t / lowpass_filter_width;
            double kaiser_arg = beta * std::sqrt(std::max(0.0, 1.0 - t_rel * t_rel));
            double win_val = bessel_i0(kaiser_arg) / i0_beta;

            double weight = sinc_val * win_val * scale;
            if (k >= 0 && k < static_cast<int64_t>(input.size())) {
                sum_val += input[k] * weight;
            }
        }
        output[j] = static_cast<float>(sum_val);
    }
}

void resample_to_16k(const std::vector<float>& input, int src_rate, std::vector<float>& output) {
    resample_audio(input, src_rate, 16000, output);
}

void resample_to_24k(const std::vector<float>& input, int src_rate, std::vector<float>& output) {
    resample_audio(input, src_rate, 24000, output);
}

// ============================================================
// compute_mel_spectrogram
// ============================================================

bool compute_mel_spectrogram(const std::vector<float>& samples_16k,
                             MelSpectrogram& mel,
                             std::string* error) {
    const int n_fft = 400;
    const int hop_length = 160;
    const int n_mels = 128;
    const int n_freq = n_fft / 2 + 1;  // 201
    const size_t n_samples = samples_16k.size();
    if (n_samples == 0) {
        if (error) *error = "Input samples vector is empty";
        return false;
    }

    // 动态计算帧数。在 PyTorch（center=True）中，帧数为 n_samples / hop_length + 1
    int n_frames = static_cast<int>(n_samples / hop_length + 1);
    int actual_frames = n_frames;

    // Hann window (PyTorch default: periodic, cos(2*pi*n/N))
    std::vector<float> hann(n_fft);
    for (int i = 0; i < n_fft; ++i)
        hann[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));

    // Mel filterbank: kMelFbData from embedded inc file (201 freqs x 128 mels)

    // Compute mel spectrogram
    std::vector<float> power_spec(n_freq);
    std::vector<float> mel_full(static_cast<size_t>(n_mels) * n_frames);
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

    for (int frm = 0; frm < n_frames; ++frm) {
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
            mel_full[static_cast<size_t>(frm) * n_mels + m] = std::log10(mel_energy);
        }
    }

    // Whisper normalization: clamp to [max-8, max], then scale
    {
        float mel_max = -1e30f;
        for (size_t i = 0; i < mel_full.size(); ++i)
            mel_max = std::max(mel_max, mel_full[i]);
        float mel_min = mel_max - 8.0f;
        for (size_t i = 0; i < mel_full.size(); ++i) {
            mel_full[i] = std::max(mel_full[i], mel_min);
            mel_full[i] = (mel_full[i] + 4.0f) / 4.0f;
        }
    }

    mel.n_frames = n_frames;
    mel.n_mels = n_mels;
    mel.actual_frames = actual_frames;
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
