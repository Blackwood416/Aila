#pragma once

#include <vector>
#include <string>

// Audio preprocessing: WAV loading, resampling, mel spectrogram.
// All CPU-side C++ — no Python, no external FFmpeg.

struct AudioBuffer {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> samples;  // f32 samples interleaved if multi-channel
};

using WavFile = AudioBuffer; // For backward compatibility

struct MelSpectrogram {
    int n_frames = 0;       // time dimension
    int n_mels = 128;       // mel bins
    int actual_frames = 0;  // actual frames (unpadded), for audio encoder
    std::vector<float> data;  // [n_mels * n_frames] row-major
};

// Load audio file with WAV/MP3 native support and FFmpeg fallback.
bool load_audio(const std::string& path, AudioBuffer& audio, std::string* error = nullptr);

// Load audio from memory data
bool load_audio_from_memory(const uint8_t* data, size_t size, const std::string& format, AudioBuffer& audio, std::string* error = nullptr);

// Keep for legacy compatibility, calls load_audio internally.
bool load_wav(const std::string& path, WavFile& wav, std::string* error = nullptr);

// Cubic spline resampler from src_rate to 16kHz.
void resample_to_16k(const std::vector<float>& input, int src_rate, std::vector<float>& output);

// Compute Whisper-compatible mel spectrogram (128 bins, 16kHz, n_fft=400, hop=160).
// Input: mono 16kHz float samples.
bool compute_mel_spectrogram(const std::vector<float>& samples_16k,
                             MelSpectrogram& mel,
                             std::string* error = nullptr);

// Save float PCM samples to a WAV file.
bool save_wav(const std::string& path, const std::vector<float>& samples, unsigned int sample_rate);

// Convenience: full pipeline from WAV/audio path to mel spectrogram.
inline bool wav_to_mel(const std::string& wav_path, MelSpectrogram& mel,
                       std::string* error = nullptr) {
    AudioBuffer audio;
    if (!load_audio(wav_path, audio, error)) return false;

    std::vector<float> mono;
    if (audio.channels > 1) {
        // Convert to mono first by averaging interleaved channels
        mono.resize(audio.samples.size() / audio.channels);
        for (size_t i = 0; i < mono.size(); ++i) {
            float sum = 0.0f;
            for (int c = 0; c < audio.channels; ++c)
                sum += audio.samples[i * audio.channels + c];
            mono[i] = sum / static_cast<float>(audio.channels);
        }
    } else {
        mono = std::move(audio.samples);
    }

    std::vector<float> resampled;
    resample_to_16k(mono, audio.sample_rate, resampled);

    return compute_mel_spectrogram(resampled, mel, error);
}
