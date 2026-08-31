#include "MimiEncoder.hpp"
#include "AudioPreprocessor.hpp"
#include "SafeTensors.hpp"
#include "MemoryMappedFile.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <cfloat>

namespace aila {
namespace audio {

namespace {

// ============================================================
// Mathematical functions & helper ops
// ============================================================

inline float elu(float x) {
    return x > 0.0f ? x : std::expm1f(x);
}

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
}

// 1D Causal Convolution for SEANet
// input:  [inC, T_in]
// weight: [outC, inC, K]
// bias:   [outC] (nullable)
// output: [outC, T_out]
// pad_total: left causal padding
// extra: right padding
void conv1d_causal(
    const float* input, int inC, int T_in,
    const float* weight, const float* bias, int outC, int K,
    int stride, int pad_total, int extra, bool replicate_mode,
    std::vector<float>& output)
{
    int T_pad = pad_total + T_in + extra;
    std::vector<float> padded(inC * T_pad);

    for (int c = 0; c < inC; ++c) {
        const float* src = input + c * T_in;
        float* dst = padded.data() + c * T_pad;

        if (replicate_mode) {
            float first_val = src[0];
            for (int p = 0; p < pad_total; ++p) dst[p] = first_val;
            std::memcpy(dst + pad_total, src, T_in * sizeof(float));
            float last_val = src[T_in - 1];
            for (int e = 0; e < extra; ++e) dst[pad_total + T_in + e] = last_val;
        } else {
            // Zero (constant) padding
            std::memset(dst, 0, pad_total * sizeof(float));
            std::memcpy(dst + pad_total, src, T_in * sizeof(float));
            if (extra > 0) {
                std::memset(dst + pad_total + T_in, 0, extra * sizeof(float));
            }
        }
    }

    int T_out = (T_pad - K) / stride + 1;
    output.resize(outC * T_out);

    for (int oc = 0; oc < outC; ++oc) {
        const float* w_oc = weight + oc * inC * K;
        float* dst = output.data() + oc * T_out;
        float b = bias ? bias[oc] : 0.0f;

        for (int t = 0; t < T_out; ++t) {
            float sum = b;
            int in_start = t * stride;
            for (int ic = 0; ic < inC; ++ic) {
                const float* w_ic = w_oc + ic * K;
                const float* src_ic = padded.data() + ic * T_pad + in_start;
                for (int k = 0; k < K; ++k) {
                    sum += w_ic[k] * src_ic[k];
                }
            }
            dst[t] = sum;
        }
    }
}

void apply_elu_inplace(std::vector<float>& buf) {
    for (float& val : buf) {
        val = elu(val);
    }
}

void layer_norm(const float* x, const float* gamma, const float* beta,
                int hidden_size, float* out, float eps = 1e-5f)
{
    float mean = 0.0f;
    for (int i = 0; i < hidden_size; ++i) mean += x[i];
    mean /= static_cast<float>(hidden_size);

    float var = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(hidden_size);
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < hidden_size; ++i) {
        out[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Matrix multiplication: C = A * B
// A: [M, K], B: [N, K] (stored as standard linear weight [out_features, in_features])
// C: [M, N]
void matmul_linear(const float* A, const float* B, int M, int N, int K, float* C) {
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + m * K;
        float* c_row = C + m * N;
        for (int n = 0; n < N; ++n) {
            const float* b_row = B + n * K;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            c_row[n] = sum;
        }
    }
}

} // anonymous namespace

// ============================================================
// MimiEncoder::Impl
// ============================================================

struct MimiEncoder::Impl {
    // 1. SEANet Conv weights
    // L0: conv 1 -> 64, k=7, s=1, pad=6
    std::vector<float> l0_w, l0_b;
    // L1: ResnetBlock (64 -> 32 -> 64)
    std::vector<float> l1_b1_w, l1_b1_b;
    std::vector<float> l1_b3_w, l1_b3_b;
    // L3: conv 64 -> 128, k=8, s=4, pad=4
    std::vector<float> l3_w, l3_b;
    // L4: ResnetBlock (128 -> 64 -> 128)
    std::vector<float> l4_b1_w, l4_b1_b;
    std::vector<float> l4_b3_w, l4_b3_b;
    // L6: conv 128 -> 256, k=10, s=5, pad=5
    std::vector<float> l6_w, l6_b;
    // L7: ResnetBlock (256 -> 128 -> 256)
    std::vector<float> l7_b1_w, l7_b1_b;
    std::vector<float> l7_b3_w, l7_b3_b;
    // L9: conv 256 -> 512, k=12, s=6, pad=6
    std::vector<float> l9_w, l9_b;
    // L10: ResnetBlock (512 -> 256 -> 512)
    std::vector<float> l10_b1_w, l10_b1_b;
    std::vector<float> l10_b3_w, l10_b3_b;
    // L12: conv 512 -> 1024, k=16, s=8, pad=8
    std::vector<float> l12_w, l12_b;
    // L14: conv 1024 -> 512, k=3, s=1, pad=2
    std::vector<float> l14_w, l14_b;

    // 2. Transformer layers (8 layers)
    struct TransformerLayer {
        std::vector<float> input_ln_w, input_ln_b;
        std::vector<float> q_w, k_w, v_w, o_w;
        std::vector<float> self_attn_scale;
        std::vector<float> post_ln_w, post_ln_b;
        std::vector<float> mlp_fc1_w, mlp_fc2_w;
        std::vector<float> mlp_scale;
    };
    std::vector<TransformerLayer> tfm_layers;

    // 3. Downsample
    std::vector<float> down_w; // [512, 512, 4]

    // 4. Split RVQ
    // Semantic
    std::vector<float> sem_proj_w; // [256, 512, 1]
    std::vector<float> sem_cb;     // [2048, 256]
    // Acoustic
    std::vector<float> ac_proj_w;  // [256, 512, 1]
    std::vector<std::vector<float>> ac_cb; // 15 codebooks, each [2048, 256]
};

MimiEncoder::MimiEncoder() : impl_(std::make_unique<Impl>()) {}
MimiEncoder::~MimiEncoder() = default;

bool MimiEncoder::loadWeights(const std::string& safetensorsPath, std::string* error) {
    try {
        MemoryMappedFile mmap(safetensorsPath);
        const uint8_t* raw = mmap.data();
        uint64_t headerSize = *reinterpret_cast<const uint64_t*>(raw);
        std::string jsonStr(reinterpret_cast<const char*>(raw + 8), headerSize);
        const uint8_t* tensorDataStart = raw + 8 + headerSize;

        std::unordered_map<std::string, TensorMeta> metadata;
        ParseHeader(jsonStr, metadata);

        auto load_f32 = [&](const std::string& name, std::vector<float>& dst) -> bool {
            auto it = metadata.find(name);
            if (it == metadata.end()) {
                if (error) *error = "MimiEncoder missing tensor: " + name;
                return false;
            }
            const TensorMeta& meta = it->second;
            int64_t numel = 1;
            for (int i = 0; i < meta.shape.ndims; ++i) numel *= meta.shape.dims[i];
            dst.resize(static_cast<size_t>(numel));

            if (meta.dtype == DT_F32) {
                std::memcpy(dst.data(), tensorDataStart + meta.byte_offset_start, numel * sizeof(float));
            } else if (meta.dtype == DT_BF16) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(tensorDataStart + meta.byte_offset_start);
                for (size_t i = 0; i < static_cast<size_t>(numel); ++i) {
                    uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                    std::memcpy(&dst[i], &bits, sizeof(float));
                }
            } else {
                if (error) *error = "Unsupported dtype for: " + name;
                return false;
            }
            return true;
        };

        // 1. SEANet weights
        if (!load_f32("encoder.encoder.layers.0.conv.weight", impl_->l0_w)) return false;
        if (!load_f32("encoder.encoder.layers.0.conv.bias",   impl_->l0_b)) return false;

        if (!load_f32("encoder.encoder.layers.1.block.1.conv.weight", impl_->l1_b1_w)) return false;
        if (!load_f32("encoder.encoder.layers.1.block.1.conv.bias",   impl_->l1_b1_b)) return false;
        if (!load_f32("encoder.encoder.layers.1.block.3.conv.weight", impl_->l1_b3_w)) return false;
        if (!load_f32("encoder.encoder.layers.1.block.3.conv.bias",   impl_->l1_b3_b)) return false;

        if (!load_f32("encoder.encoder.layers.3.conv.weight", impl_->l3_w)) return false;
        if (!load_f32("encoder.encoder.layers.3.conv.bias",   impl_->l3_b)) return false;

        if (!load_f32("encoder.encoder.layers.4.block.1.conv.weight", impl_->l4_b1_w)) return false;
        if (!load_f32("encoder.encoder.layers.4.block.1.conv.bias",   impl_->l4_b1_b)) return false;
        if (!load_f32("encoder.encoder.layers.4.block.3.conv.weight", impl_->l4_b3_w)) return false;
        if (!load_f32("encoder.encoder.layers.4.block.3.conv.bias",   impl_->l4_b3_b)) return false;

        if (!load_f32("encoder.encoder.layers.6.conv.weight", impl_->l6_w)) return false;
        if (!load_f32("encoder.encoder.layers.6.conv.bias",   impl_->l6_b)) return false;

        if (!load_f32("encoder.encoder.layers.7.block.1.conv.weight", impl_->l7_b1_w)) return false;
        if (!load_f32("encoder.encoder.layers.7.block.1.conv.bias",   impl_->l7_b1_b)) return false;
        if (!load_f32("encoder.encoder.layers.7.block.3.conv.weight", impl_->l7_b3_w)) return false;
        if (!load_f32("encoder.encoder.layers.7.block.3.conv.bias",   impl_->l7_b3_b)) return false;

        if (!load_f32("encoder.encoder.layers.9.conv.weight", impl_->l9_w)) return false;
        if (!load_f32("encoder.encoder.layers.9.conv.bias",   impl_->l9_b)) return false;

        if (!load_f32("encoder.encoder.layers.10.block.1.conv.weight", impl_->l10_b1_w)) return false;
        if (!load_f32("encoder.encoder.layers.10.block.1.conv.bias",   impl_->l10_b1_b)) return false;
        if (!load_f32("encoder.encoder.layers.10.block.3.conv.weight", impl_->l10_b3_w)) return false;
        if (!load_f32("encoder.encoder.layers.10.block.3.conv.bias",   impl_->l10_b3_b)) return false;

        if (!load_f32("encoder.encoder.layers.12.conv.weight", impl_->l12_w)) return false;
        if (!load_f32("encoder.encoder.layers.12.conv.bias",   impl_->l12_b)) return false;

        if (!load_f32("encoder.encoder.layers.14.conv.weight", impl_->l14_w)) return false;
        if (!load_f32("encoder.encoder.layers.14.conv.bias",   impl_->l14_b)) return false;

        // 2. Transformer layers (8 layers)
        impl_->tfm_layers.resize(8);
        for (int i = 0; i < 8; ++i) {
            auto& tl = impl_->tfm_layers[i];
            std::string p = "encoder.encoder_transformer.layers." + std::to_string(i) + ".";

            if (!load_f32(p + "input_layernorm.weight", tl.input_ln_w)) return false;
            if (!load_f32(p + "input_layernorm.bias",   tl.input_ln_b)) return false;

            if (!load_f32(p + "self_attn.q_proj.weight", tl.q_w)) return false;
            if (!load_f32(p + "self_attn.k_proj.weight", tl.k_w)) return false;
            if (!load_f32(p + "self_attn.v_proj.weight", tl.v_w)) return false;
            if (!load_f32(p + "self_attn.o_proj.weight", tl.o_w)) return false;
            if (!load_f32(p + "self_attn_layer_scale.scale", tl.self_attn_scale)) return false;

            if (!load_f32(p + "post_attention_layernorm.weight", tl.post_ln_w)) return false;
            if (!load_f32(p + "post_attention_layernorm.bias",   tl.post_ln_b)) return false;

            if (!load_f32(p + "mlp.fc1.weight", tl.mlp_fc1_w)) return false;
            if (!load_f32(p + "mlp.fc2.weight", tl.mlp_fc2_w)) return false;
            if (!load_f32(p + "mlp_layer_scale.scale", tl.mlp_scale)) return false;
        }

        // 3. Downsample
        if (!load_f32("encoder.downsample.conv.weight", impl_->down_w)) return false;

        // 4. Split RVQ
        if (!load_f32("encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight", impl_->sem_proj_w)) return false;
        std::vector<float> sem_sum, sem_usage;
        if (!load_f32("encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum", sem_sum)) return false;
        if (!load_f32("encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage", sem_usage)) return false;
        impl_->sem_cb.resize(2048 * 256);
        for (int c = 0; c < 2048; ++c) {
            float u = std::max(sem_usage[c], 1e-5f);
            for (int d = 0; d < 256; ++d) {
                impl_->sem_cb[c * 256 + d] = sem_sum[c * 256 + d] / u;
            }
        }

        if (!load_f32("encoder.quantizer.acoustic_residual_vector_quantizer.input_proj.weight", impl_->ac_proj_w)) return false;
        impl_->ac_cb.resize(15);
        for (int l = 0; l < 15; ++l) {
            std::string p = "encoder.quantizer.acoustic_residual_vector_quantizer.layers." + std::to_string(l) + ".codebook.";
            std::vector<float> ac_sum, ac_usage;
            if (!load_f32(p + "embed_sum", ac_sum)) return false;
            if (!load_f32(p + "cluster_usage", ac_usage)) return false;
            impl_->ac_cb[l].resize(2048 * 256);
            for (int c = 0; c < 2048; ++c) {
                float u = std::max(ac_usage[c], 1e-5f);
                for (int d = 0; d < 256; ++d) {
                    impl_->ac_cb[l][c * 256 + d] = ac_sum[c * 256 + d] / u;
                }
            }
        }

        loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("MimiEncoder::loadWeights: ") + e.what();
        loaded_ = false;
        return false;
    }
}

bool MimiEncoder::encode(const float* samples, int numSamples,
                         TTSReferenceCodes& outCodes, std::string* error)
{
    if (!loaded_) {
        if (error) *error = "MimiEncoder not loaded";
        return false;
    }
    if (numSamples < 1024) {
        if (error) *error = "Audio too short (need >= 1024 samples at 24kHz)";
        return false;
    }

    // ==========================================
    // Stage 1: SEANet Causal Conv Frontend
    // ==========================================
    // x: [1, numSamples]
    std::vector<float> curr(samples, samples + numSamples);
    int curr_c = 1;
    int curr_t = numSamples;

    // Helper lambda to run conv1d
    auto run_conv = [&](const std::vector<float>& w, const std::vector<float>& b,
                        int out_c, int k, int s, int pad_total, int extra, bool repl) {
        std::vector<float> out;
        conv1d_causal(curr.data(), curr_c, curr_t,
                      w.data(), b.empty() ? nullptr : b.data(),
                      out_c, k, s, pad_total, extra, repl, out);
        curr = std::move(out);
        curr_c = out_c;
        curr_t = curr.size() / curr_c;
    };

    // Helper lambda for resnet block: residual + conv(elu(conv(elu(x))))
    auto run_resnet = [&](const std::vector<float>& w1, const std::vector<float>& b1, int mid_c,
                          const std::vector<float>& w3, const std::vector<float>& b3, int out_c) {
        std::vector<float> res = curr;
        apply_elu_inplace(curr);
        run_conv(w1, b1, mid_c, 3, 1, 2, 0, false);
        apply_elu_inplace(curr);
        run_conv(w3, b3, out_c, 1, 1, 0, 0, false);
        for (size_t i = 0; i < curr.size(); ++i) {
            curr[i] += res[i];
        }
    };

    // L0: conv 1 -> 64, k=7, s=1, pad=6
    run_conv(impl_->l0_w, impl_->l0_b, 64, 7, 1, 6, 0, false);

    // L1: ResnetBlock (64 -> 32 -> 64)
    run_resnet(impl_->l1_b1_w, impl_->l1_b1_b, 32, impl_->l1_b3_w, impl_->l1_b3_b, 64);
    apply_elu_inplace(curr);

    // L3: conv 64 -> 128, k=8, s=4, pad=4
    run_conv(impl_->l3_w, impl_->l3_b, 128, 8, 4, 4, 0, false);

    // L4: ResnetBlock (128 -> 64 -> 128)
    run_resnet(impl_->l4_b1_w, impl_->l4_b1_b, 64, impl_->l4_b3_w, impl_->l4_b3_b, 128);
    apply_elu_inplace(curr);

    // L6: conv 128 -> 256, k=10, s=5, pad=5
    run_conv(impl_->l6_w, impl_->l6_b, 256, 10, 5, 5, 0, false);

    // L7: ResnetBlock (256 -> 128 -> 256)
    run_resnet(impl_->l7_b1_w, impl_->l7_b1_b, 128, impl_->l7_b3_w, impl_->l7_b3_b, 256);
    apply_elu_inplace(curr);

    // L9: conv 256 -> 512, k=12, s=6, pad=6
    run_conv(impl_->l9_w, impl_->l9_b, 512, 12, 6, 6, 0, false);

    // L10: ResnetBlock (512 -> 256 -> 512)
    run_resnet(impl_->l10_b1_w, impl_->l10_b1_b, 256, impl_->l10_b3_w, impl_->l10_b3_b, 512);
    apply_elu_inplace(curr);

    // L12: conv 512 -> 1024, k=16, s=8, pad=8
    run_conv(impl_->l12_w, impl_->l12_b, 1024, 16, 8, 8, 0, false);
    apply_elu_inplace(curr);

    // L14: conv 1024 -> 512, k=3, s=1, pad=2
    run_conv(impl_->l14_w, impl_->l14_b, 512, 3, 1, 2, 0, false);

    // curr is now [512, T_tfm]
    int T_tfm = curr_t;
    // Transpose to [T_tfm, 512] for transformer
    std::vector<float> h(T_tfm * 512);
    for (int c = 0; c < 512; ++c) {
        for (int t = 0; t < T_tfm; ++t) {
            h[t * 512 + c] = curr[c * T_tfm + t];
        }
    }

    // ==========================================
    // Stage 2: Transformer (8 layers)
    // ==========================================
    // Precompute RoPE cos/sin table: [T_tfm, 64]
    std::vector<float> rope_cos(T_tfm * 64), rope_sin(T_tfm * 64);
    for (int t = 0; t < T_tfm; ++t) {
        for (int i = 0; i < 32; ++i) {
            float inv_freq = 1.0f / std::pow(10000.0f, (2.0f * i) / 64.0f);
            float angle = static_cast<float>(t) * inv_freq;
            float c = std::cos(angle);
            float s = std::sin(angle);
            rope_cos[t * 64 + i] = c;
            rope_cos[t * 64 + i + 32] = c;
            rope_sin[t * 64 + i] = s;
            rope_sin[t * 64 + i + 32] = s;
        }
    }

    std::vector<float> normed(T_tfm * 512);
    std::vector<float> q(T_tfm * 512), k(T_tfm * 512), v(T_tfm * 512);
    std::vector<float> attn_out(T_tfm * 512);
    std::vector<float> o_proj_out(T_tfm * 512);
    std::vector<float> scores(T_tfm);
    std::vector<float> mlp_fc1_out(T_tfm * 2048);
    std::vector<float> mlp_fc2_out(T_tfm * 512);

    for (int l = 0; l < 8; ++l) {
        const auto& tl = impl_->tfm_layers[l];

        // 1. input_layernorm
        for (int t = 0; t < T_tfm; ++t) {
            layer_norm(h.data() + t * 512, tl.input_ln_w.data(), tl.input_ln_b.data(), 512, normed.data() + t * 512);
        }

        // 2. Q, K, V projections
        matmul_linear(normed.data(), tl.q_w.data(), T_tfm, 512, 512, q.data());
        matmul_linear(normed.data(), tl.k_w.data(), T_tfm, 512, 512, k.data());
        matmul_linear(normed.data(), tl.v_w.data(), T_tfm, 512, 512, v.data());

        // 3. Apply RoPE to Q and K
        for (int t = 0; t < T_tfm; ++t) {
            const float* cos_t = rope_cos.data() + t * 64;
            const float* sin_t = rope_sin.data() + t * 64;

            for (int head = 0; head < 8; ++head) {
                float* q_head = q.data() + t * 512 + head * 64;
                float* k_head = k.data() + t * 512 + head * 64;

                for (int d = 0; d < 32; ++d) {
                    float q1 = q_head[d], q2 = q_head[d + 32];
                    q_head[d]      = q1 * cos_t[d]      + (-q2) * sin_t[d];
                    q_head[d + 32] = q2 * cos_t[d + 32] + ( q1) * sin_t[d + 32];

                    float k1 = k_head[d], k2 = k_head[d + 32];
                    k_head[d]      = k1 * cos_t[d]      + (-k2) * sin_t[d];
                    k_head[d + 32] = k2 * cos_t[d + 32] + ( k1) * sin_t[d + 32];
                }
            }
        }

        // 4. Multi-head Causal Self-Attention
        const float scale = 1.0f / 8.0f; // 1 / sqrt(64)
        for (int t_q = 0; t_q < T_tfm; ++t_q) {
            for (int head = 0; head < 8; ++head) {
                const float* q_ptr = q.data() + t_q * 512 + head * 64;

                float max_score = -FLT_MAX;
                for (int t_k = 0; t_k <= t_q; ++t_k) {
                    const float* k_ptr = k.data() + t_k * 512 + head * 64;
                    float dot = 0.0f;
                    for (int d = 0; d < 64; ++d) {
                        dot += q_ptr[d] * k_ptr[d];
                    }
                    float sc = dot * scale;
                    scores[t_k] = sc;
                    if (sc > max_score) max_score = sc;
                }

                float sum_exp = 0.0f;
                for (int t_k = 0; t_k <= t_q; ++t_k) {
                    scores[t_k] = std::exp(scores[t_k] - max_score);
                    sum_exp += scores[t_k];
                }
                float inv_sum = 1.0f / sum_exp;
                for (int t_k = 0; t_k <= t_q; ++t_k) {
                    scores[t_k] *= inv_sum;
                }

                float* out_ptr = attn_out.data() + t_q * 512 + head * 64;
                for (int d = 0; d < 64; ++d) out_ptr[d] = 0.0f;
                for (int t_k = 0; t_k <= t_q; ++t_k) {
                    float p = scores[t_k];
                    const float* v_ptr = v.data() + t_k * 512 + head * 64;
                    for (int d = 0; d < 64; ++d) {
                        out_ptr[d] += p * v_ptr[d];
                    }
                }
            }
        }

        // 5. Output projection + self_attn_layer_scale
        matmul_linear(attn_out.data(), tl.o_w.data(), T_tfm, 512, 512, o_proj_out.data());
        for (int t = 0; t < T_tfm; ++t) {
            float* h_row = h.data() + t * 512;
            const float* o_row = o_proj_out.data() + t * 512;
            for (int d = 0; d < 512; ++d) {
                h_row[d] += o_row[d] * tl.self_attn_scale[d];
            }
        }

        // 6. post_attention_layernorm
        for (int t = 0; t < T_tfm; ++t) {
            layer_norm(h.data() + t * 512, tl.post_ln_w.data(), tl.post_ln_b.data(), 512, normed.data() + t * 512);
        }

        // 7. MLP
        matmul_linear(normed.data(), tl.mlp_fc1_w.data(), T_tfm, 2048, 512, mlp_fc1_out.data());
        for (float& v_gelu : mlp_fc1_out) {
            v_gelu = gelu(v_gelu);
        }
        matmul_linear(mlp_fc1_out.data(), tl.mlp_fc2_w.data(), T_tfm, 512, 2048, mlp_fc2_out.data());

        for (int t = 0; t < T_tfm; ++t) {
            float* h_row = h.data() + t * 512;
            const float* f2_row = mlp_fc2_out.data() + t * 512;
            for (int d = 0; d < 512; ++d) {
                h_row[d] += f2_row[d] * tl.mlp_scale[d];
            }
        }
    }

    // ==========================================
    // Stage 3: Downsample (512 -> 512, k=4, s=2)
    // ==========================================
    // Transpose h back to [512, T_tfm]
    std::vector<float> h_down_in(512 * T_tfm);
    for (int t = 0; t < T_tfm; ++t) {
        for (int c = 0; c < 512; ++c) {
            h_down_in[c * T_tfm + t] = h[t * 512 + c];
        }
    }

    // extra padding formula: extra = (T_tfm % 2 != 0) ? 1 : 0
    int extra = (T_tfm % 2 != 0) ? 1 : 0;
    std::vector<float> down_out;
    conv1d_causal(h_down_in.data(), 512, T_tfm,
                  impl_->down_w.data(), nullptr,
                  512, 4, 2, 2, extra, true, down_out);

    int frames = down_out.size() / 512;

    // ==========================================
    // Stage 4: Split RVQ (16 codebooks)
    // ==========================================
    outCodes.frames = frames;
    outCodes.codebooks = 16;
    outCodes.codes.resize(frames * 16);

    // 1. Semantic RVQ (Codebook 0)
    std::vector<float> sem_proj;
    conv1d_causal(down_out.data(), 512, frames,
                  impl_->sem_proj_w.data(), nullptr,
                  256, 1, 1, 0, 0, false, sem_proj);

    for (int f = 0; f < frames; ++f) {
        // sem_proj feature vector at frame f: [256]
        // sem_proj layout is [256, frames]
        float min_dist = FLT_MAX;
        int best_idx = 0;

        for (int c = 0; c < 2048; ++c) {
            const float* cb_vec = impl_->sem_cb.data() + c * 256;
            float dist = 0.0f;
            for (int d = 0; d < 256; ++d) {
                float diff = sem_proj[d * frames + f] - cb_vec[d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = c;
            }
        }
        outCodes.codes[f * 16 + 0] = best_idx;
    }

    // 2. Acoustic RVQ (Codebooks 1 ~ 15)
    std::vector<float> ac_proj;
    conv1d_causal(down_out.data(), 512, frames,
                  impl_->ac_proj_w.data(), nullptr,
                  256, 1, 1, 0, 0, false, ac_proj);

    // Transpose ac_proj to [frames, 256] residual
    std::vector<float> residual(frames * 256);
    for (int f = 0; f < frames; ++f) {
        for (int d = 0; d < 256; ++d) {
            residual[f * 256 + d] = ac_proj[d * frames + f];
        }
    }

    for (int l = 0; l < 15; ++l) {
        const float* cb_l = impl_->ac_cb[l].data();

        for (int f = 0; f < frames; ++f) {
            float* res_vec = residual.data() + f * 256;
            float min_dist = FLT_MAX;
            int best_idx = 0;

            for (int c = 0; c < 2048; ++c) {
                const float* cb_vec = cb_l + c * 256;
                float dist = 0.0f;
                for (int d = 0; d < 256; ++d) {
                    float diff = res_vec[d] - cb_vec[d];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = c;
                }
            }

            outCodes.codes[f * 16 + (l + 1)] = best_idx;

            // Subtract quantized vector from residual
            const float* quantized_vec = cb_l + best_idx * 256;
            for (int d = 0; d < 256; ++d) {
                res_vec[d] -= quantized_vec[d];
            }
        }
    }

    return true;
}

bool MimiEncoder::encodeFromFile(const std::string& audioPath,
                                 TTSReferenceCodes& outCodes, std::string* error)
{
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

    // Resample to 24kHz if needed (band-limited Kaiser sinc)
    std::vector<float> resampled;
    if (audio.sample_rate != 24000) {
        resample_to_24k(mono, audio.sample_rate, resampled);
    } else {
        resampled = std::move(mono);
    }

    return encode(resampled.data(), static_cast<int>(resampled.size()), outCodes, error);
}

} // namespace audio
} // namespace aila
