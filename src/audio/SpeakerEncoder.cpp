#include "SpeakerEncoder.hpp"
#include "AudioPreprocessor.hpp"
#include "SafeTensors.hpp"
#include "MemoryMappedFile.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aila {
namespace audio {

// ============================================================
// bf16 -> f32 conversion
// ============================================================
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ============================================================
// Safetensors loading helpers
// ============================================================

// Load a single tensor from safetensors mmap by name, converting bf16 -> f32.
// Returns true on success. 'data' is resized to numel floats.
static bool loadTensorFromSafetensors(
    const uint8_t* tensorDataStart,
    const std::unordered_map<std::string, TensorMeta>& metadata,
    const std::string& name,
    std::vector<float>& data,
    std::string* error)
{
    auto it = metadata.find(name);
    if (it == metadata.end()) {
        if (error) *error = "Tensor not found: " + name;
        return false;
    }
    const TensorMeta& meta = it->second;
    int64_t numel = 1;
    for (int i = 0; i < meta.shape.ndims; ++i) {
        numel *= meta.shape.dims[i];
    }
    size_t byteSize = meta.byte_offset_end - meta.byte_offset_start;
    data.resize(static_cast<size_t>(numel));

    if (meta.dtype == DT_BF16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(tensorDataStart + meta.byte_offset_start);
        for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
            data[i] = bf16_to_f32(src[i]);
        }
    } else if (meta.dtype == DT_F32) {
        std::memcpy(data.data(), tensorDataStart + meta.byte_offset_start, numel * sizeof(float));
    } else if (meta.dtype == DT_F16) {
        // Not expected for speaker encoder, but handle gracefully
        if (error) *error = "F16 not supported for speaker encoder: " + name;
        return false;
    } else {
        if (error) *error = "Unsupported dtype for: " + name;
        return false;
    }
    return true;
}

// ============================================================
// Mel spectrogram computation (TTS speaker encoder version)
// ============================================================

// Slaney mel scale (librosa-compatible)
static float hz_to_mel_slaney(float hz) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - 0.0f) / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (hz < min_log_hz) {
        return (hz - 0.0f) / f_sp;
    } else {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
}

static float mel_to_hz_slaney(float mel) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = (min_log_hz - 0.0f) / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;
    if (mel < min_log_mel) {
        return 0.0f + f_sp * mel;
    } else {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
}

static void buildSlaneyMelFilterbank(int n_mels, int n_fft, int sampleRate,
                                      float f_min, float f_max,
                                      std::vector<float>& filterbank) {
    int n_fft_bins = n_fft / 2 + 1;
    filterbank.assign(n_mels * n_fft_bins, 0.0f);

    float mel_min = hz_to_mel_slaney(f_min);
    float mel_max = hz_to_mel_slaney(f_max);

    std::vector<float> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    std::vector<float> hz_pts(n_mels + 2);
    std::vector<float> fft_freqs(n_fft_bins);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_pts[i] = mel_to_hz_slaney(mel_pts[i]);
    }
    for (int i = 0; i < n_fft_bins; ++i) {
        fft_freqs[i] = static_cast<float>(i) * sampleRate / n_fft;
    }

    for (int m = 0; m < n_mels; ++m) {
        float f_left  = hz_pts[m];
        float f_center = hz_pts[m + 1];
        float f_right  = hz_pts[m + 2];
        float enorm = 2.0f / (f_right - f_left);  // Slaney area normalization

        for (int k = 0; k < n_fft_bins; ++k) {
            float freq = fft_freqs[k];
            if (freq >= f_left && freq <= f_center && f_center > f_left) {
                filterbank[m * n_fft_bins + k] = enorm * (freq - f_left) / (f_center - f_left);
            } else if (freq > f_center && freq <= f_right && f_right > f_center) {
                filterbank[m * n_fft_bins + k] = enorm * (f_right - freq) / (f_right - f_center);
            }
        }
    }
}

// Direct DFT (not FFT) for correctness
static void computeDFT(const float* input, float* real, float* imag, int n) {
    for (int k = 0; k < n; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int t = 0; t < n; ++t) {
            float angle = -2.0f * static_cast<float>(M_PI) * k * t / n;
            real[k] += input[t] * std::cos(angle);
            imag[k] += input[t] * std::sin(angle);
        }
    }
}

bool SpeakerEncoder::computeMelSpectrogram(const float* samples, int numSamples,
                                            std::vector<float>& mel, int& nFrames) {
    constexpr int n_fft = 1024;
    constexpr int hop_length = 256;
    constexpr int win_length = 1024;
    constexpr int n_mels = 128;
    constexpr int sample_rate = 24000;
    constexpr float f_min = 0.0f;
    constexpr float f_max = 12000.0f;

    // Reflect padding: (n_fft - hop_length) / 2 = 384 on each side
    int padding = (n_fft - hop_length) / 2;
    int padded_len = numSamples + 2 * padding;

    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; ++i) {
        int src;
        if (i < padding) {
            src = padding - i;
        } else if (i >= padding + numSamples) {
            src = 2 * numSamples - (i - padding) - 2;
        } else {
            src = i - padding;
        }
        src = std::max(0, std::min(numSamples - 1, src));
        padded[i] = samples[src];
    }

    // center=False: frames start at 0, step by hop_length
    nFrames = (padded_len - n_fft) / hop_length + 1;
    if (nFrames <= 0) return false;

    // Build mel filterbank
    int n_fft_bins = n_fft / 2 + 1;  // 513
    std::vector<float> filterbank;
    buildSlaneyMelFilterbank(n_mels, n_fft, sample_rate, f_min, f_max, filterbank);

    // Hann window centered in n_fft (win_length samples of Hann, rest zeros)
    std::vector<float> window(n_fft, 0.0f);
    int win_offset = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; ++i) {
        window[win_offset + i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / win_length));
    }

    mel.resize(n_mels * nFrames);

    std::vector<float> frame(n_fft);
    std::vector<float> fft_real(n_fft);
    std::vector<float> fft_imag(n_fft);

    for (int f = 0; f < nFrames; ++f) {
        int start = f * hop_length;
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = padded[start + i] * window[i];
        }

        computeDFT(frame.data(), fft_real.data(), fft_imag.data(), n_fft);

        // Magnitude: sqrt(real^2 + imag^2 + 1e-9)
        for (int m = 0; m < n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_fft_bins; ++k) {
                float mag = std::sqrt(fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k] + 1e-9f);
                sum += filterbank[m * n_fft_bins + k] * mag;
            }
            // dynamic_range_compression: ln(clamp(x, min=1e-5))
            mel[m * nFrames + f] = std::log(std::max(sum, 1e-5f));
        }
    }

    return true;
}

// ============================================================
// ECAPA-TDNN helper ops
// ============================================================

void SpeakerEncoder::reflectPad1d(const float* input, int C, int T, int pad,
                                   std::vector<float>& output) {
    int Tp = T + 2 * pad;
    output.resize(C * Tp);
    for (int c = 0; c < C; ++c) {
        const float* src = input + c * T;
        float* dst = output.data() + c * Tp;
        // Left reflect
        for (int i = 0; i < pad; ++i) {
            dst[i] = src[pad - i];
        }
        // Center
        std::memcpy(dst + pad, src, T * sizeof(float));
        // Right reflect
        for (int i = 0; i < pad; ++i) {
            dst[pad + T + i] = src[T - 2 - i];
        }
    }
}

void SpeakerEncoder::conv1d(const float* input, int T, int inC,
                             const float* weight, const float* bias, int outC, int K,
                             int stride, int pad, int dilation, bool relu,
                             std::vector<float>& output) {
    // weight layout: [outC][inC][K]
    // input layout: [inC][T_padded] after reflect padding
    // output layout: [outC][T]

    // Apply reflect padding
    std::vector<float> padded;
    const float* work_input;
    int T_work;
    if (pad > 0) {
        reflectPad1d(input, inC, T, pad, padded);
        work_input = padded.data();
        T_work = T + 2 * pad;
    } else {
        work_input = input;
        T_work = T;
    }

    // Output time length
    int T_out = (T_work - (K - 1) * dilation - 1) / stride + 1;
    // For "same" padding with stride=1: T_out == T
    output.resize(outC * T_out);

    for (int o = 0; o < outC; ++o) {
        const float* w_oc = weight + o * inC * K;
        float* dst = output.data() + o * T_out;
        float b = bias ? bias[o] : 0.0f;
        for (int t = 0; t < T_out; ++t) {
            float sum = b;
            for (int i = 0; i < inC; ++i) {
                const float* w_ic = w_oc + i * K;
                const float* src = work_input + i * T_work;
                for (int k = 0; k < K; ++k) {
                    sum += w_ic[k] * src[t * stride + k * dilation];
                }
            }
            dst[t] = relu ? std::max(sum, 0.0f) : sum;
        }
    }
}

void SpeakerEncoder::globalAvgPool1d(const float* input, int C, int T, float* output) {
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        const float* src = input + c * T;
        for (int t = 0; t < T; ++t) {
            sum += src[t];
        }
        output[c] = sum / static_cast<float>(T);
    }
}

void SpeakerEncoder::softmax1d(float* data, int C, int T) {
    for (int c = 0; c < C; ++c) {
        float* ptr = data + c * T;
        float max_val = ptr[0];
        for (int t = 1; t < T; ++t) {
            max_val = std::max(max_val, ptr[t]);
        }
        float sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            ptr[t] = std::exp(ptr[t] - max_val);
            sum += ptr[t];
        }
        for (int t = 0; t < T; ++t) {
            ptr[t] /= sum;
        }
    }
}

// ============================================================
// ECAPA-TDNN forward pass
// ============================================================

void SpeakerEncoder::forward(const float* mel, int nFrames, float* embedding) {
    const int T = nFrames;
    const int C = kHiddenDim;       // 512
    const int MFA_C = kMfaChannels; // 1536
    const int SCALE = kRes2NetScale; // 8
    const int BR = kBranchDim;       // 64

    // Reusable buffers
    std::vector<float> buf_cur, buf_next, buf_branch_in, buf_concat, buf_se;

    // --- conv0: 128 -> 512, k=5, pad=2, dil=1, ReLU ---
    conv1d(mel, T, kMelBins,
           conv0_w_.data(), conv0_b_.data(), C, 5,
           1, 2, 1, true, buf_cur);
    // buf_cur: [512, T] = block_outputs[0]

    // MFA will concatenate block 1, 2, 3 outputs (not block 0)
    std::vector<float> mfa_parts[3];

    int dilations[3] = {2, 3, 4};

    for (int blk = 0; blk < 3; ++blk) {
        const auto& bw = blk_[blk];
        int dil = dilations[blk];

        // Save residual (input to this SE-Res2Net block)
        std::vector<float> residual = buf_cur;

        // --- tdnn1: 512 -> 512, k=1, pad=0, ReLU ---
        conv1d(buf_cur.data(), T, C,
               bw.tdnn1_w.data(), bw.tdnn1_b.data(), C, 1,
               1, 0, 1, true, buf_next);
        // buf_next: [512, T]

        // --- Res2Net: split 512 channels into 8 branches of 64 ---
        std::vector<float> branch_out[8];
        // Branch 0: identity
        branch_out[0].assign(buf_next.data(), buf_next.data() + BR * T);

        for (int b = 1; b < SCALE; ++b) {
            // Build branch input: for b==1, just the raw branch data;
            // for b>=2, raw branch data + previous branch output.
            const float* bin;
            if (b == 1) {
                bin = buf_next.data() + b * BR * T;
            } else {
                buf_branch_in.resize(BR * T);
                const float* raw_branch = buf_next.data() + b * BR * T;
                for (int i = 0; i < BR * T; ++i) {
                    buf_branch_in[i] = raw_branch[i] + branch_out[b - 1][i];
                }
                bin = buf_branch_in.data();
            }

            // Convolution: [64, T] -> [64, T], k=3, pad=dil, dilation=dil, ReLU
            conv1d(bin, T, BR,
                   bw.res2net_w[b - 1].data(), bw.res2net_b[b - 1].data(), BR, 3,
                   1, dil, dil, true, branch_out[b]);
        }

        // Concatenate all 8 branches back: [512, T]
        buf_concat.resize(C * T);
        for (int b = 0; b < SCALE; ++b) {
            std::memcpy(buf_concat.data() + b * BR * T, branch_out[b].data(), BR * T * sizeof(float));
        }

        // --- tdnn2: 512 -> 512, k=1, pad=0, ReLU ---
        conv1d(buf_concat.data(), T, C,
               bw.tdnn2_w.data(), bw.tdnn2_b.data(), C, 1,
               1, 0, 1, true, buf_cur);
        // buf_cur: [512, T]

        // --- SE (Squeeze-Excitation) ---
        // Global avg pool over time: [512, T] -> [512, 1]
        std::vector<float> se_pooled(C);
        globalAvgPool1d(buf_cur.data(), C, T, se_pooled.data());

        // FC (conv1d k=1): 512 -> 128, ReLU
        conv1d(se_pooled.data(), 1, C,
               bw.se_conv1_w.data(), bw.se_conv1_b.data(), 128, 1,
               1, 0, 1, true, buf_se);

        // FC: 128 -> 512, Sigmoid
        conv1d(buf_se.data(), 1, 128,
               bw.se_conv2_w.data(), bw.se_conv2_b.data(), C, 1,
               1, 0, 1, false, buf_se);
        for (float& v : buf_se) v = 1.0f / (1.0f + std::exp(-v));

        // Channel-wise multiply gate
        for (int c = 0; c < C; ++c) {
            float g = buf_se[c];
            for (int t = 0; t < T; ++t) {
                buf_cur[c * T + t] *= g;
            }
        }

        // Residual add
        for (size_t i = 0; i < residual.size(); ++i) {
            buf_cur[i] += residual[i];
        }

        // Save for MFA
        mfa_parts[blk].assign(buf_cur.data(), buf_cur.data() + C * T);
    }

    // --- MFA: concat blocks 1, 2, 3 -> [1536, T] ---
    std::vector<float> mfa_input(MFA_C * T);
    for (int b = 0; b < 3; ++b) {
        std::memcpy(mfa_input.data() + b * C * T, mfa_parts[b].data(), C * T * sizeof(float));
    }

    // MFA conv: [1536, T] -> [1536, T], k=1, ReLU
    std::vector<float> mfa_out;
    conv1d(mfa_input.data(), T, MFA_C,
           mfa_w_.data(), mfa_b_.data(), MFA_C, 1,
           1, 0, 1, true, mfa_out);

    // --- ASP (Attentive Statistics Pooling) ---
    // Global mean and std over time
    std::vector<float> global_mean(MFA_C);
    globalAvgPool1d(mfa_out.data(), MFA_C, T, global_mean.data());

    std::vector<float> global_std(MFA_C);
    for (int c = 0; c < MFA_C; ++c) {
        float sum_sq = 0.0f;
        const float* src = mfa_out.data() + c * T;
        for (int t = 0; t < T; ++t) sum_sq += src[t] * src[t];
        float mean_sq = sum_sq / static_cast<float>(T);
        float var = mean_sq - global_mean[c] * global_mean[c];
        global_std[c] = std::sqrt(std::max(var, 1e-12f));
    }

    // Concatenate features + mean + std: [1536*3, T] = [4608, T]
    std::vector<float> attn_input(MFA_C * 3 * T);
    std::memcpy(attn_input.data(), mfa_out.data(), MFA_C * T * sizeof(float));
    for (int c = 0; c < MFA_C; ++c) {
        for (int t = 0; t < T; ++t) {
            attn_input[(MFA_C + c) * T + t] = global_mean[c];
            attn_input[(2 * MFA_C + c) * T + t] = global_std[c];
        }
    }

    // ASP TDNN: [4608, T] -> [128, T], k=1, ReLU, then Tanh
    std::vector<float> asp_out;
    conv1d(attn_input.data(), T, MFA_C * 3,
           asp_tdnn_w_.data(), asp_tdnn_b_.data(), 128, 1,
           1, 0, 1, true, asp_out);
    for (float& v : asp_out) v = std::tanh(v);

    // ASP Conv: [128, T] -> [1536, T], k=1, then Softmax over time
    std::vector<float> asp_attn;
    conv1d(asp_out.data(), T, 128,
           asp_conv_w_.data(), asp_conv_b_.data(), MFA_C, 1,
           1, 0, 1, false, asp_attn);
    softmax1d(asp_attn.data(), MFA_C, T);

    // Weighted mean and std
    std::vector<float> weighted_mean(MFA_C, 0.0f);
    std::vector<float> weighted_std(MFA_C, 0.0f);
    for (int c = 0; c < MFA_C; ++c) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int t = 0; t < T; ++t) {
            float a = asp_attn[c * T + t];
            float v = mfa_out[c * T + t];
            sum += a * v;
        }
        weighted_mean[c] = sum;
        for (int t = 0; t < T; ++t) {
            float a = asp_attn[c * T + t];
            float diff = mfa_out[c * T + t] - weighted_mean[c];
            sum_sq += a * diff * diff;
        }
        weighted_std[c] = std::sqrt(std::max(sum_sq, 1e-12f));
    }

    // Concatenate: [3072, 1]
    std::vector<float> pooled(MFA_C * 2);
    std::memcpy(pooled.data(), weighted_mean.data(), MFA_C * sizeof(float));
    std::memcpy(pooled.data() + MFA_C, weighted_std.data(), MFA_C * sizeof(float));

    // --- FC: [3072, 1] -> [1024, 1], k=1 ---
    std::vector<float> fc_out;
    conv1d(pooled.data(), 1, MFA_C * 2,
           fc_w_.data(), fc_b_.data(), embedding_dim_, 1,
           1, 0, 1, false, fc_out);

    std::memcpy(embedding, fc_out.data(), embedding_dim_ * sizeof(float));
}

// ============================================================
// Public API
// ============================================================

bool SpeakerEncoder::loadWeights(const std::string& safetensorsPath, std::string* error) {
    try {
        MemoryMappedFile mmap(safetensorsPath);
        const uint8_t* raw = mmap.data();

        uint64_t headerSize = *reinterpret_cast<const uint64_t*>(raw);
        std::string jsonStr(reinterpret_cast<const char*>(raw + 8), headerSize);
        const uint8_t* tensorDataStart = raw + 8 + headerSize;

        std::unordered_map<std::string, TensorMeta> metadata;
        ParseHeader(jsonStr, metadata);

        // Helper to load a tensor by name
        auto load = [&](const std::string& name, std::vector<float>& dst) -> bool {
            return loadTensorFromSafetensors(tensorDataStart, metadata, name, dst, error);
        };

        // --- conv0 ---
        if (!load("speaker_encoder.blocks.0.conv.weight", conv0_w_)) return false;
        if (!load("speaker_encoder.blocks.0.conv.bias",   conv0_b_)) return false;

        // --- Blocks 1-3 ---
        for (int b = 0; b < 3; ++b) {
            int pyIdx = b + 1;  // Python indices: blocks.1, blocks.2, blocks.3
            auto& bw = blk_[b];
            std::string prefix = "speaker_encoder.blocks." + std::to_string(pyIdx);

            if (!load(prefix + ".tdnn1.conv.weight", bw.tdnn1_w)) return false;
            if (!load(prefix + ".tdnn1.conv.bias",   bw.tdnn1_b)) return false;

            if (!load(prefix + ".tdnn2.conv.weight", bw.tdnn2_w)) return false;
            if (!load(prefix + ".tdnn2.conv.bias",   bw.tdnn2_b)) return false;

            for (int r = 0; r < 7; ++r) {
                std::string rp = prefix + ".res2net_block.blocks." + std::to_string(r);
                if (!load(rp + ".conv.weight", bw.res2net_w[r])) return false;
                if (!load(rp + ".conv.bias",   bw.res2net_b[r])) return false;
            }

            if (!load(prefix + ".se_block.conv1.weight", bw.se_conv1_w)) return false;
            if (!load(prefix + ".se_block.conv1.bias",   bw.se_conv1_b)) return false;
            if (!load(prefix + ".se_block.conv2.weight", bw.se_conv2_w)) return false;
            if (!load(prefix + ".se_block.conv2.bias",   bw.se_conv2_b)) return false;
        }

        // --- MFA ---
        if (!load("speaker_encoder.mfa.conv.weight", mfa_w_)) return false;
        if (!load("speaker_encoder.mfa.conv.bias",   mfa_b_)) return false;

        // --- ASP ---
        if (!load("speaker_encoder.asp.tdnn.conv.weight", asp_tdnn_w_)) return false;
        if (!load("speaker_encoder.asp.tdnn.conv.bias",   asp_tdnn_b_)) return false;
        if (!load("speaker_encoder.asp.conv.weight", asp_conv_w_)) return false;
        if (!load("speaker_encoder.asp.conv.bias",   asp_conv_b_)) return false;

        // --- FC ---
        if (!load("speaker_encoder.fc.weight", fc_w_)) return false;
        if (!load("speaker_encoder.fc.bias",   fc_b_)) return false;

        // Determine embedding dimension from FC weight shape [enc_dim, 3072, 1]
        embedding_dim_ = static_cast<int>(fc_w_.size()) / 3072;

        loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("SpeakerEncoder::loadWeights: ") + e.what();
        loaded_ = false;
        return false;
    }
}

bool SpeakerEncoder::extractEmbedding(const float* samples, int numSamples,
                                       std::vector<float>& embedding, std::string* error) {
    if (!loaded_) {
        if (error) *error = "SpeakerEncoder not loaded";
        return false;
    }
    if (numSamples < 1024) {
        if (error) *error = "Audio too short (need at least 1024 samples at 24kHz)";
        return false;
    }

    std::vector<float> mel;
    int nFrames = 0;
    if (!computeMelSpectrogram(samples, numSamples, mel, nFrames)) {
        if (error) *error = "Mel spectrogram computation failed";
        return false;
    }

    embedding.resize(embedding_dim_);
    forward(mel.data(), nFrames, embedding.data());
    return true;
}

bool SpeakerEncoder::extractEmbeddingFromFile(const std::string& audioPath,
                                               std::vector<float>& embedding, std::string* error) {
    AudioBuffer audio;
    if (!load_audio(audioPath, audio, error)) {
        return false;
    }

    // Convert to mono if needed
    std::vector<float> mono;
    if (audio.channels > 1) {
        mono.resize(audio.samples.size() / audio.channels);
        for (size_t i = 0; i < mono.size(); ++i) {
            float sum = 0.0f;
            for (int c = 0; c < audio.channels; ++c) {
                sum += audio.samples[i * audio.channels + c];
            }
            mono[i] = sum / static_cast<float>(audio.channels);
        }
    } else {
        mono = std::move(audio.samples);
    }

    // Resample to 24kHz if needed (cubic spline, same algorithm as resample_to_16k)
    std::vector<float> resampled;
    if (audio.sample_rate != 24000) {
        double ratio = static_cast<double>(audio.sample_rate) / 24000.0;
        size_t outSize = static_cast<size_t>(std::round(mono.size() / ratio));
        resampled.resize(outSize);

        auto get_sample = [&](int idx) -> float {
            if (idx < 0) return mono[0];
            if (idx >= static_cast<int>(mono.size())) return mono[mono.size() - 1];
            return mono[idx];
        };

        for (size_t i = 0; i < outSize; ++i) {
            double t = i * ratio;
            int idx = static_cast<int>(std::floor(t));
            double f = t - idx;

            float y0 = get_sample(idx - 1);
            float y1 = get_sample(idx);
            float y2 = get_sample(idx + 1);
            float y3 = get_sample(idx + 2);

            float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
            float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
            float a2 = -0.5f * y0 + 0.5f * y2;
            float a3 = y1;

            resampled[i] = static_cast<float>(((a0 * f + a1) * f + a2) * f + a3);
        }
    } else {
        resampled = std::move(mono);
    }

    return extractEmbedding(resampled.data(), static_cast<int>(resampled.size()),
                            embedding, error);
}

} // namespace audio
} // namespace aila
