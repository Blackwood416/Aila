#include "Qwen3ASRAudioEncoder.hpp"
#include "../core/Context.hpp"
#include "../utils/EnvUtils.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdint>
#include <chrono>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace aila {
namespace audio {

// ---- helpers ----
static void dump_tensor_f32(Context& ctx, Tensor& t, const std::string& filename) {
    std::vector<bf16> dbg(t.numel());
    ctx.memcpy_d2h(dbg.data(), t.data(), dbg.size() * sizeof(bf16));
    ctx.synchronize();
    std::vector<float> f32(dbg.size());
    for (size_t i = 0; i < dbg.size(); ++i) {
        f32[i] = (float)dbg[i];
    }
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(f32.data()), f32.size() * sizeof(float));
}

static void nchw_to_ntch(Context& ctx, Tensor& src, Tensor& dst,
                          int C, int H, int W_padded, int W_out) {
    // src: [C*H*W_padded] NCHW row-major (N=1)
    // dst: [W_out, C*H] row-major
    bf16* s = src.data_as<bf16>();
    bf16* d = dst.data_as<bf16>();
    int ch = C * H;
    ctx.queue().parallel_for(sycl::range<1>(W_out * ch),
        [=](sycl::id<1> idx) {
            int i = static_cast<int>(idx[0]);
            int t = i / ch;        // w dimension → time
            int r = i % ch;        // c*H combined
            // NCHW: [n][c][h][w], n=0
            int c = r / H;
            int h = r % H;
            int src_idx = ((c * H) + h) * W_padded + t;
            d[i] = s[src_idx];
        });
}

Qwen3ASRAudioEncoder::~Qwen3ASRAudioEncoder() = default;

Tensor* Qwen3ASRAudioEncoder::get_tensor(ModelWeights& weights,
                                          const std::string& name,
                                          std::string* error_message,
                                          bool required) {
    if (!weights.has(name)) {
        if (required && error_message) {
            *error_message = "Missing weight: " + name;
        }
        return nullptr;
    }
    return &weights.get(name);
}

void Qwen3ASRAudioEncoder::ensure_buffers(int seq_len) {
    if (seq_len <= buf_capacity_) return;
    buf_capacity_ = seq_len + 64;  // padding

    auto* ctx = ctx_;
    int d = cfg_.d_model;
    int ffn = cfg_.encoder_ffn_dim;
    int heads = cfg_.encoder_attention_heads;
    int hd = cfg_.head_dim;

    enc_hidden_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_normed_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_q_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_k_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_v_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_attn_out_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_ffn_ = Tensor::allocate(*ctx, {buf_capacity_, ffn});
    enc_proj1_ = Tensor::allocate(*ctx, {buf_capacity_, d});
    enc_scores_ = Tensor::allocate(*ctx, {heads, buf_capacity_, buf_capacity_},
                                    dnnl::memory::data_type::f32);
}

// ---- load ----

bool Qwen3ASRAudioEncoder::load(Context& ctx,
                                 ModelWeights& weights,
                                 const AudioEncoderConfig& cfg,
                                 std::string* error_message) {
    ctx_ = &ctx;
    cfg_ = cfg;
    loaded_ = false;

    auto err = [&](const std::string& msg) {
        if (error_message) *error_message = msg;
    };

    auto G = [&](const std::string& name, bool required = true) -> Tensor* {
        return get_tensor(weights, "thinker.audio_tower." + name, error_message, required);
    };

    // Conv2D frontend
    conv2d1_weight_ = G("conv2d1.weight");
    conv2d1_bias_   = G("conv2d1.bias");
    conv2d2_weight_ = G("conv2d2.weight");
    conv2d2_bias_   = G("conv2d2.bias");
    conv2d3_weight_ = G("conv2d3.weight");
    conv2d3_bias_   = G("conv2d3.bias");
    if (!conv2d1_weight_ || !conv2d2_weight_ || !conv2d3_weight_) {
        err("Missing conv2d weights"); return false;
    }

    // conv_out Linear: in_features = downsample_hidden_size * feat_height
    // feat_height = ((mel_bins-1)//2+1-1)//2+1-1)//2+1 = 16 for 128 mel bins
    int feat_height = cfg.num_mel_bins;
    for (int i = 0; i < 3; ++i) {
        feat_height = (feat_height - 1) / 2 + 1;
    }
    int conv_in_dim = cfg.downsample_hidden_size * feat_height;  // 480 * 16 = 7680
    Tensor* conv_out_weight = G("conv_out.weight");
    if (conv_out_weight) {
        conv_out_.init(ctx, *conv_out_weight, conv_in_dim, cfg.d_model);
    } else {
        err("Missing conv_out.weight"); return false;
    }

    // Encoder layers
    layers_.resize(cfg.encoder_layers);
    for (int i = 0; i < cfg.encoder_layers; ++i) {
        auto& L = layers_[i];
        std::string p = "layers." + std::to_string(i) + ".";

        L.sa_ln_weight = G(p + "self_attn_layer_norm.weight");
        L.sa_ln_bias   = G(p + "self_attn_layer_norm.bias");
        L.final_ln_weight = G(p + "final_layer_norm.weight");
        L.final_ln_bias   = G(p + "final_layer_norm.bias");

        Tensor* qw = G(p + "self_attn.q_proj.weight");
        Tensor* kw = G(p + "self_attn.k_proj.weight");
        Tensor* vw = G(p + "self_attn.v_proj.weight");
        Tensor* ow = G(p + "self_attn.out_proj.weight");
        if (!qw || !kw || !vw || !ow) { err("Missing attn weights layer " + std::to_string(i)); return false; }

        L.q_proj.init(ctx, *qw, cfg.d_model, cfg.d_model);
        L.k_proj.init(ctx, *kw, cfg.d_model, cfg.d_model);
        L.v_proj.init(ctx, *vw, cfg.d_model, cfg.d_model);
        L.out_proj.init(ctx, *ow, cfg.d_model, cfg.d_model);

        L.q_bias = G(p + "self_attn.q_proj.bias", false);
        L.k_bias = G(p + "self_attn.k_proj.bias", false);
        L.v_bias = G(p + "self_attn.v_proj.bias", false);
        L.out_bias = G(p + "self_attn.out_proj.bias", false);

        Tensor* fc1w = G(p + "fc1.weight");
        Tensor* fc2w = G(p + "fc2.weight");
        if (!fc1w || !fc2w) { err("Missing fc weights layer " + std::to_string(i)); return false; }

        L.fc1.init(ctx, *fc1w, cfg.d_model, cfg.encoder_ffn_dim);
        L.fc2.init(ctx, *fc2w, cfg.encoder_ffn_dim, cfg.d_model);
        L.fc1_bias = G(p + "fc1.bias", false);
        L.fc2_bias = G(p + "fc2.bias", false);
    }

    // Output projection
    ln_post_weight = G("ln_post.weight");
    ln_post_bias   = G("ln_post.bias");
    if (!ln_post_weight) { err("Missing ln_post.weight"); return false; }

    Tensor* proj1_weight = G("proj1.weight");
    Tensor* proj2_weight = G("proj2.weight");
    if (!proj1_weight || !proj2_weight) { err("Missing output projection weights"); return false; }
    proj1_.init(ctx, *proj1_weight, cfg.d_model, cfg.d_model);
    proj2_.init(ctx, *proj2_weight, cfg.d_model, cfg.output_dim);

    proj1_bias_ = G("proj1.bias", false);
    proj2_bias_ = G("proj2.bias", false);

    if (cfg.output_dim <= 0) {
        err("Invalid audio output_dim");
        return false;
    }

    loaded_ = true;
    return true;
}

// ---- encode ----

bool Qwen3ASRAudioEncoder::encode(Context& ctx,
                                   Tensor& mel_input, int mel_time_len,
                                   Tensor& output_features, int& output_len,
                                   std::string* error_message) {
    if (!loaded_) {
        if (error_message) *error_message = "Audio encoder not loaded";
        return false;
    }

    bool dump_logits = aila::env::read_flag("AILA_DUMP_LOGITS", false);
    static const bool s_profile_asr = aila::env::read_flag("AILA_ASR_PROFILE", false);
    last_timing_ = Qwen3ASRAudioEncoderTiming{};
    auto stage_start = std::chrono::high_resolution_clock::now();
    auto finish_stage = [&](double& target) {
        if (!s_profile_asr) {
            return;
        }
        ctx.synchronize();
        const auto now = std::chrono::high_resolution_clock::now();
        target += std::chrono::duration<double, std::milli>(now - stage_start).count();
        stage_start = now;
    };

    int d = cfg_.d_model;
    int ffn = cfg_.encoder_ffn_dim;
    int heads = cfg_.encoder_attention_heads;
    int hd = cfg_.head_dim;
    int ds = cfg_.downsample_hidden_size;
    int mel_bins = cfg_.num_mel_bins;

    // Chunked processing: split mel into chunks of n_window*2 = 100 frames
    int chunk_size = cfg_.n_window * 2;  // 100
    int n_chunks = (mel_time_len + chunk_size - 1) / chunk_size;
    int feat_height = mel_bins;
    for (int i = 0; i < 3; ++i) feat_height = (feat_height - 1) / 2 + 1;  // 16
    int conv_flat_dim = ds * feat_height;  // 7680

    // Compute per-chunk conv output length (13 frames per 100-frame chunk)
    int chunk_out_w = 0;
    int total_out = 0;
    std::vector<int> chunk_out_lengths;
    for (int ci = 0; ci < n_chunks; ++ci) {
        int chunk_start = ci * chunk_size;
        int this_chunk_len = std::min(chunk_size, mel_time_len - chunk_start);
        int cw = this_chunk_len;
        for (int s = 0; s < 3; ++s) cw = (cw + 2 - 3) / 2 + 1;
        chunk_out_lengths.push_back(cw);
        total_out += cw;
        if (ci == 0) chunk_out_w = cw;  // first chunk determines padded width
    }

    // Allocate buffer for all chunks' conv output
    Tensor all_conv_out = Tensor::allocate(ctx, {total_out, d});
    int out_offset = 0;

    // Process each chunk through conv2d independently
    for (int ci = 0; ci < n_chunks; ++ci) {
        int chunk_start = ci * chunk_size;
        int this_chunk_len = std::min(chunk_size, mel_time_len - chunk_start);
        int cw = chunk_out_lengths[ci];

        // Copy chunk: mel_input({1,mel_bins,padded}) -> chunk_mel({1,1,mel_bins,chunk_size}) (always pad to chunk_size=100)
        int mel_row_len = mel_input.shape(2);
        int chunk_mel_w = chunk_size;
        Tensor chunk_mel = Tensor::allocate(ctx, {1, 1, mel_bins, chunk_mel_w});
        {
            bf16* src_base = mel_input.data_as<bf16>();
            bf16* dst = chunk_mel.data_as<bf16>();
            int ck = chunk_start;
            int cl_eff = this_chunk_len;
            int cl_pad = chunk_mel_w;
            int mb = mel_bins;
            int mrl = mel_row_len;
            ctx.queue().parallel_for(sycl::range<1>(mb * cl_pad), [=](sycl::id<1> idx) {
                int i = (int)idx[0];
                int h = i / cl_pad;
                int w = i % cl_pad;
                if (w < cl_eff) {
                    dst[h * cl_pad + w] = src_base[h * mrl + ck + w];
                } else {
                    dst[h * cl_pad + w] = (bf16)0.0f;
                }
            });
        }

        int c1_h = (mel_bins + 2 - 3) / 2 + 1;
        int c1_w = (chunk_mel_w + 2 - 3) / 2 + 1;
        Tensor c1_out = Tensor::allocate(ctx, {1, ds, c1_h, c1_w});
        ops::conv2d_gelu(ctx, chunk_mel, *conv2d1_weight_, *conv2d1_bias_,
                         c1_out, 1, 1, ds, mel_bins, chunk_mel_w, c1_h, c1_w);

        if (ci == 0 && dump_logits) {
            dump_tensor_f32(ctx, c1_out, "debug_cpp_c1.bin");
        }

        int c2_h = (c1_h + 2 - 3) / 2 + 1;
        int c2_w = (c1_w + 2 - 3) / 2 + 1;
        Tensor c2_out = Tensor::allocate(ctx, {1, ds, c2_h, c2_w});
        ops::conv2d_gelu(ctx, c1_out, *conv2d2_weight_, *conv2d2_bias_,
                         c2_out, 1, ds, ds, c1_h, c1_w, c2_h, c2_w);

        int c3_h = (c2_h + 2 - 3) / 2 + 1;
        int c3_w = (c2_w + 2 - 3) / 2 + 1;
        Tensor c3_out = Tensor::allocate(ctx, {1, ds, c3_h, c3_w});
        ops::conv2d_gelu(ctx, c2_out, *conv2d3_weight_, *conv2d3_bias_,
                         c3_out, 1, ds, ds, c2_h, c2_w, c3_h, c3_w);
        if (ci == 0 && dump_logits) {
            dump_tensor_f32(ctx, c3_out, "debug_cpp_c3.bin");
        }

        // Reshape + conv_out
        Tensor conv_flat = Tensor::allocate(ctx, {cw, conv_flat_dim});
        nchw_to_ntch(ctx, c3_out, conv_flat, ds, c3_h, c3_w, cw);

        // View of output buffer for this chunk
        bf16* out_ptr = all_conv_out.data_as<bf16>() + out_offset * d;
        Tensor chunk_dst = Tensor::view(ctx, out_ptr, {cw, d});
        conv_out_.forward(ctx, conv_flat, chunk_dst, cw);

        // Add sinusoidal PE (per-chunk positions 0..cw-1)
        ops::sinusoidal_position_embedding(ctx, chunk_dst, cw, d);

        out_offset += cw;
    }
    finish_stage(last_timing_.conv_ms);

    // Now run encoder layers on full sequence
    int audio_seq_len = total_out;
    ensure_buffers(audio_seq_len);

    // Copy all_conv_out to enc_hidden_
    ops::copy_tensor(ctx, all_conv_out, enc_hidden_, audio_seq_len * d);

    if (dump_logits) {
        dump_tensor_f32(ctx, enc_hidden_, "debug_cpp_hidden_states_pre_encoder.bin");
    }

    float ln_eps = 1e-5f;
    for (int i = 0; i < cfg_.encoder_layers; ++i) {
        auto& L = layers_[i];

        ops::layer_norm(ctx, enc_hidden_, *L.sa_ln_weight, *L.sa_ln_bias,
                        ln_eps, enc_normed_, audio_seq_len, d);

        L.q_proj.forward(ctx, enc_normed_, enc_q_, audio_seq_len);
        L.k_proj.forward(ctx, enc_normed_, enc_k_, audio_seq_len);
        L.v_proj.forward(ctx, enc_normed_, enc_v_, audio_seq_len);

        if (L.q_bias) ops::bias_add_inplace(ctx, enc_q_, *L.q_bias, audio_seq_len, d);
        if (L.k_bias) ops::bias_add_inplace(ctx, enc_k_, *L.k_bias, audio_seq_len, d);
        if (L.v_bias) ops::bias_add_inplace(ctx, enc_v_, *L.v_bias, audio_seq_len, d);

        if (i == 0 && dump_logits) {
            dump_tensor_f32(ctx, enc_q_, "debug_cpp_l0_q.bin");
            dump_tensor_f32(ctx, enc_k_, "debug_cpp_l0_k.bin");
            dump_tensor_f32(ctx, enc_v_, "debug_cpp_l0_v.bin");
        }

        // aftercnn_len = audio_seq_len: The actual model uses _get_feat_extract_output_lengths()
        // which accounts for chunked processing and gives the same value as total_out (e.g., 135).
        // This differs from the naive 3-layer stride-2 formula which incorrectly gives 130.
        int aftercnn_len = audio_seq_len;

        int block_size = 13 * (cfg_.n_window_infer / (cfg_.n_window * 2));
        std::vector<int> cu_seqlens = {0};
        {
            int curr = 0;
            while (curr < aftercnn_len) {
                int this_block = std::min(block_size, aftercnn_len - curr);
                curr += this_block;
                cu_seqlens.push_back(curr);
            }
        }

        for (size_t b = 0; b < cu_seqlens.size() - 1; ++b) {
            int start = cu_seqlens[b];
            int end = cu_seqlens[b + 1];
            int len = end - start;

            bf16* q_ptr = enc_q_.data_as<bf16>() + start * d;
            bf16* k_ptr = enc_k_.data_as<bf16>() + start * d;
            bf16* v_ptr = enc_v_.data_as<bf16>() + start * d;
            bf16* out_ptr = enc_attn_out_.data_as<bf16>() + start * d;

            Tensor q_block = Tensor::view(ctx, q_ptr, {len, d}, enc_q_.dtype());
            Tensor k_block = Tensor::view(ctx, k_ptr, {len, d}, enc_k_.dtype());
            Tensor v_block = Tensor::view(ctx, v_ptr, {len, d}, enc_v_.dtype());
            Tensor out_block = Tensor::view(ctx, out_ptr, {len, d}, enc_attn_out_.dtype());
            Tensor scores_block = Tensor::view(ctx, enc_scores_.data(), {heads, len, len}, enc_scores_.dtype());

            ops::attention_bidi(ctx, q_block, k_block, v_block,
                               out_block, scores_block,
                               len, heads, hd);
        }

        if (audio_seq_len > aftercnn_len) {
            bf16* out_ptr = enc_attn_out_.data_as<bf16>();
            int rem_start = aftercnn_len;
            int rem_end = audio_seq_len;

            ctx.queue().parallel_for(sycl::range<1>((rem_end - rem_start) * d), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                out_ptr[rem_start * d + i] = (bf16)0.0f;
            });
        }

        if (i == 0 && dump_logits) {
            dump_tensor_f32(ctx, enc_attn_out_, "debug_cpp_l0_attn_out.bin");
        }

        L.out_proj.forward(ctx, enc_attn_out_, enc_normed_, audio_seq_len);
        if (L.out_bias) ops::bias_add_inplace(ctx, enc_normed_, *L.out_bias, audio_seq_len, d);
        ops::residual_add(ctx, enc_hidden_, enc_normed_, audio_seq_len * d);

        ops::layer_norm(ctx, enc_hidden_, *L.final_ln_weight, *L.final_ln_bias,
                        ln_eps, enc_normed_, audio_seq_len, d);

        L.fc1.forward(ctx, enc_normed_, enc_ffn_, audio_seq_len);
        if (L.fc1_bias) ops::bias_add_inplace(ctx, enc_ffn_, *L.fc1_bias, audio_seq_len, ffn);
        ops::gelu_tanh_inplace(ctx, enc_ffn_, audio_seq_len * ffn);

        L.fc2.forward(ctx, enc_ffn_, enc_normed_, audio_seq_len);
        if (L.fc2_bias) ops::bias_add_inplace(ctx, enc_normed_, *L.fc2_bias, audio_seq_len, d);
        ops::residual_add(ctx, enc_hidden_, enc_normed_, audio_seq_len * d);

        if (dump_logits && (i == 0 || i == 1 || i == 2 || i == 11 || i == 23)) {
            std::string fname = "debug_cpp_l" + std::to_string(i) + "_out.bin";
            dump_tensor_f32(ctx, enc_hidden_, fname);
        }
    }
    finish_stage(last_timing_.transformer_ms);

    // Output projection
    ops::layer_norm(ctx, enc_hidden_, *ln_post_weight, *ln_post_bias,
                    ln_eps, enc_normed_, audio_seq_len, d);
    proj1_.forward(ctx, enc_normed_, enc_proj1_, audio_seq_len);
    if (proj1_bias_) {
        ops::bias_add_inplace(ctx, enc_proj1_, *proj1_bias_, audio_seq_len, d);
    }
    ops::gelu_tanh_inplace(ctx, enc_proj1_, audio_seq_len * d);
    proj2_.forward(ctx, enc_proj1_, output_features, audio_seq_len);
    if (proj2_bias_) {
        ops::bias_add_inplace(ctx, output_features, *proj2_bias_, audio_seq_len, cfg_.output_dim);
    }
    finish_stage(last_timing_.proj_ms);

    if (dump_logits) {
        dump_tensor_f32(ctx, output_features, "debug_cpp_af.bin");
    }

    output_len = audio_seq_len;
    return true;
}

} // namespace audio
} // namespace aila
