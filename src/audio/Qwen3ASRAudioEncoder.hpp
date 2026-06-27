#pragma once

#include "engine/Types.hpp"
#include "../utils/SafeTensors.hpp"
#include "../ops/Ops.hpp"
#include "../ops/ConvOps.hpp"
#include <string>
#include <vector>
#include <deque>
#include <sycl/sycl.hpp>

class Context;

namespace aila {
namespace audio {

using bf16 = sycl::ext::oneapi::bfloat16;

struct Qwen3ASRAudioEncoderTiming {
    double conv_ms = 0.0;
    double transformer_ms = 0.0;
    double proj_ms = 0.0;
};

// Qwen3-ASR Audio Encoder (thinker.audio_tower).
// Conv2D frontend + sinusoidal PE + N transformer encoder layers + output projection.
class Qwen3ASRAudioEncoder {
public:
    ~Qwen3ASRAudioEncoder();

    bool load(Context& ctx,
              ModelWeights& weights,
              const AudioEncoderConfig& cfg,
              std::string* error_message);

    bool ready() const { return loaded_; }
    int output_dim() const { return cfg_.output_dim; }

    // Encode mel spectrogram → audio features.
    // mel_input: [1, num_mel_bins=128, time_frames] bf16 on GPU
    // output_features: allocated by caller, must be at least [max_audio_len, output_dim]
    // output_len: actual number of audio feature tokens produced
    bool encode(Context& ctx,
                Tensor& mel_input, int mel_time_len,
                Tensor& output_features, int& output_len,
                std::string* error_message);

    Qwen3ASRAudioEncoderTiming last_timing() const { return last_timing_; }

private:
    AudioEncoderConfig cfg_;
    bool loaded_ = false;
    Context* ctx_ = nullptr;

    // Conv2D frontend weights
    Tensor* conv2d1_weight_ = nullptr;
    Tensor* conv2d1_bias_ = nullptr;
    Tensor* conv2d2_weight_ = nullptr;
    Tensor* conv2d2_bias_ = nullptr;
    Tensor* conv2d3_weight_ = nullptr;
    Tensor* conv2d3_bias_ = nullptr;
    Linear conv_out_;

    // Encoder layers
    struct EncoderLayer {
        Tensor* sa_ln_weight = nullptr;   // self_attn_layer_norm weight
        Tensor* sa_ln_bias = nullptr;     // self_attn_layer_norm bias
        Linear q_proj, k_proj, v_proj;
        Tensor* q_bias = nullptr;
        Tensor* k_bias = nullptr;
        Tensor* v_bias = nullptr;
        Linear out_proj;
        Tensor* out_bias = nullptr;

        Tensor* final_ln_weight = nullptr;  // final_layer_norm weight
        Tensor* final_ln_bias = nullptr;
        Linear fc1, fc2;
        Tensor* fc1_bias = nullptr;
        Tensor* fc2_bias = nullptr;
    };
    std::vector<EncoderLayer> layers_;

    // Output projection
    Tensor* ln_post_weight = nullptr;
    Tensor* ln_post_bias = nullptr;
    Linear proj1_, proj2_;
    Tensor* proj1_bias_ = nullptr;
    Tensor* proj2_bias_ = nullptr;

    // Runtime buffers (reallocated as needed)
    Tensor conv_chunk_mel_;   // [1,1,num_mel_bins,n_window*2]
    Tensor conv_c1_out_;      // fixed conv frontend intermediate
    Tensor conv_c2_out_;      // fixed conv frontend intermediate
    Tensor conv_c3_out_;      // fixed conv frontend intermediate
    Tensor conv_flat_;        // [max_chunk_out_w, conv_flat_dim]
    Tensor conv_all_out_;     // [audio_seq_len, d_model] before encoder
    Tensor enc_hidden_;   // [seq_len, d_model] main hidden state
    Tensor enc_normed_;   // [seq_len, d_model] after layer norm
    Tensor enc_q_;        // [seq_len, d_model] Q
    Tensor enc_k_;        // [seq_len, d_model] K
    Tensor enc_v_;        // [seq_len, d_model] V
    Tensor enc_attn_out_; // [seq_len, d_model] attention output
    Tensor enc_ffn_;      // [seq_len, encoder_ffn_dim] FFN intermediate
    Tensor enc_proj1_;    // [seq_len, d_model] proj1 output
    Tensor enc_scores_;   // [num_heads, seq_len, seq_len] attention scores
    int buf_capacity_ = 0;
    int conv_total_capacity_ = 0;
    bool conv_chunk_buffers_ready_ = false;

    std::deque<Tensor> owned_weights_;
    Qwen3ASRAudioEncoderTiming last_timing_;

    void ensure_conv_buffers(Context& ctx, int total_out);
    void ensure_buffers(int seq_len);
    Tensor* get_tensor(ModelWeights& weights, const std::string& name,
                       std::string* error_message, bool required = true);
};

} // namespace audio
} // namespace aila
