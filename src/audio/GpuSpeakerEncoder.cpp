#include "GpuSpeakerEncoder.hpp"
#include "SafeTensors.hpp"
#include "MemoryMappedFile.hpp"
#include "../ops/ConvOps.hpp"
#include "../ops/Ops.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

using bf16 = sycl::ext::oneapi::bfloat16;

namespace aila {
namespace audio {

// Helper: load a single safetensors tensor by name and upload to GPU as bf16.
// The safetensors stores bf16 natively, so we upload directly without CPU f32 conversion.
static bool loadGpuTensor(Context& ctx,
                          const uint8_t* tensorDataStart,
                          const std::unordered_map<std::string, TensorMeta>& metadata,
                          const std::string& name,
                          Tensor& out,
                          std::string* error) {
    auto it = metadata.find(name);
    if (it == metadata.end()) {
        if (error) *error = "Tensor not found: " + name;
        return false;
    }
    const TensorMeta& meta = it->second;
    int64_t numel = 1;
    for (int i = 0; i < meta.shape.ndims; ++i) numel *= meta.shape.dims[i];

    std::vector<int64_t> shape;
    for (int i = 0; i < meta.shape.ndims; ++i) shape.push_back(meta.shape.dims[i]);

    dnnl::memory::data_type dt = (meta.dtype == DT_BF16)
        ? dnnl::memory::data_type::bf16
        : dnnl::memory::data_type::f32;

    out = Tensor::allocate(ctx, shape, dt);
    if (!out.valid()) {
        if (error) *error = "Failed to allocate GPU tensor: " + name;
        return false;
    }
    ctx.memcpy_h2d(out.data(), tensorDataStart + meta.byte_offset_start,
                   static_cast<size_t>(numel) * out.element_size());
    return true;
}

bool GpuSpeakerEncoder::loadWeights(Context& ctx, const std::string& safetensorsPath,
                                     std::string* error) {
    try {
        MemoryMappedFile mmap(safetensorsPath);
        const uint8_t* raw = mmap.data();
        uint64_t headerSize = *reinterpret_cast<const uint64_t*>(raw);
        std::string jsonStr(reinterpret_cast<const char*>(raw + 8), headerSize);
        const uint8_t* tensorDataStart = raw + 8 + headerSize;

        std::unordered_map<std::string, TensorMeta> metadata;
        ParseHeader(jsonStr, metadata);

        auto load = [&](const std::string& name, Tensor& dst) -> bool {
            return loadGpuTensor(ctx, tensorDataStart, metadata, name, dst, error);
        };

        // conv0
        if (!load("speaker_encoder.blocks.0.conv.weight", conv0_w_)) return false;
        if (!load("speaker_encoder.blocks.0.conv.bias",   conv0_b_)) return false;

        // Blocks 1-3 (Python indices)
        for (int b = 0; b < 3; ++b) {
            int pyIdx = b + 1;
            auto& bw = blk_[b];
            std::string p = "speaker_encoder.blocks." + std::to_string(pyIdx);

            if (!load(p + ".tdnn1.conv.weight", bw.tdnn1_w)) return false;
            if (!load(p + ".tdnn1.conv.bias",   bw.tdnn1_b)) return false;
            if (!load(p + ".tdnn2.conv.weight", bw.tdnn2_w)) return false;
            if (!load(p + ".tdnn2.conv.bias",   bw.tdnn2_b)) return false;

            for (int r = 0; r < 7; ++r) {
                std::string rp = p + ".res2net_block.blocks." + std::to_string(r);
                if (!load(rp + ".conv.weight", bw.res2net_w[r])) return false;
                if (!load(rp + ".conv.bias",   bw.res2net_b[r])) return false;
            }

            if (!load(p + ".se_block.conv1.weight", bw.se_conv1_w)) return false;
            if (!load(p + ".se_block.conv1.bias",   bw.se_conv1_b)) return false;
            if (!load(p + ".se_block.conv2.weight", bw.se_conv2_w)) return false;
            if (!load(p + ".se_block.conv2.bias",   bw.se_conv2_b)) return false;
        }

        // MFA
        if (!load("speaker_encoder.mfa.conv.weight", mfa_w_)) return false;
        if (!load("speaker_encoder.mfa.conv.bias",   mfa_b_)) return false;

        // ASP
        if (!load("speaker_encoder.asp.tdnn.conv.weight", asp_tdnn_w_)) return false;
        if (!load("speaker_encoder.asp.tdnn.conv.bias",   asp_tdnn_b_)) return false;
        if (!load("speaker_encoder.asp.conv.weight", asp_conv_w_)) return false;
        if (!load("speaker_encoder.asp.conv.bias",   asp_conv_b_)) return false;

        // FC
        if (!load("speaker_encoder.fc.weight", fc_w_)) return false;
        if (!load("speaker_encoder.fc.bias",   fc_b_)) return false;

        // Determine embedding dimension from FC weight shape [enc_dim, 3072, 1]
        embedding_dim_ = static_cast<int>(fc_w_.shape(0));

        loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        if (error) *error = std::string("GpuSpeakerEncoder::loadWeights: ") + e.what();
        return false;
    }
}

bool GpuSpeakerEncoder::extractEmbedding(Context& ctx, const float* mel, int nFrames,
                                          std::vector<float>& embedding, std::string* error) {
    if (!loaded_) {
        if (error) *error = "GpuSpeakerEncoder not loaded";
        return false;
    }

    const int T = nFrames;
    const int C = kHiddenDim;       // 512
    const int MFA_C = kMfaChannels; // 1536
    const int BR = kBranchDim;      // 64
    const int SCALE = 8;

    // Upload mel to GPU: transpose from CPU [n_mels, n_frames] to GPU [1, T, kMelBins]
    Tensor mel_gpu = Tensor::allocate(ctx, {1, T, kMelBins});
    std::vector<bf16> mel_bf16(kMelBins * T);
    for (int t = 0; t < T; ++t)
        for (int m = 0; m < kMelBins; ++m)
            mel_bf16[t * kMelBins + m] = bf16(mel[m * T + t]);
    ctx.memcpy_h2d(mel_gpu.data(), mel_bf16.data(), mel_bf16.size() * sizeof(bf16));

    // --- conv0: 128 → 512, k=5, pad=2, dil=1, ReLU ---
    Tensor cur = Tensor::allocate(ctx, {1, T, C});
    ops::reflect_conv1d(ctx, mel_gpu, conv0_w_, conv0_b_, cur,
                        1, kMelBins, C, T, 5, 1, 2, true);

    // Pre-allocate reusable buffers
    Tensor buf_a = Tensor::allocate(ctx, {1, T, C});     // 512-channel intermediate
    Tensor buf_branch_in = Tensor::allocate(ctx, {1, T, BR}); // branch input
    Tensor buf_branch_out[8];
    for (int b = 0; b < SCALE; ++b)
        buf_branch_out[b] = Tensor::allocate(ctx, {1, T, BR});

    Tensor buf_concat = Tensor::allocate(ctx, {1, T, C});
    Tensor buf_se_pool = Tensor::allocate(ctx, {1, 1, C});
    Tensor buf_se_hidden = Tensor::allocate(ctx, {1, 1, 128});
    Tensor buf_se_gate = Tensor::allocate(ctx, {1, 1, C});

    // Block outputs for MFA concatenation (stored as [1, T, 512] each)
    Tensor mfa_parts[3];

    int dilations[3] = {2, 3, 4};

    for (int blk = 0; blk < 3; ++blk) {
        auto& bw = blk_[blk];
        int dil = dilations[blk];

        // Save residual: copy cur → buf_a as residual
        Tensor residual = Tensor::allocate(ctx, {1, T, C});
        ops::copy_tensor(ctx, cur, residual, T * C);

        // --- tdnn1: k=1 (no padding needed, causal_conv1d works for k=1) ---
        ops::causal_conv1d(ctx, cur, bw.tdnn1_w, bw.tdnn1_b, buf_a,
                           1, C, C, T, 1, 1);
        ops::relu_inplace(ctx, buf_a, T * C);
        // buf_a = tdnn1 output

        // --- Res2Net: split 512 → 8×64 ---
        // buf_a layout: [1, T, 512] — channels last. Split channel dimension.
        // Channel c of buf_a is at offset c (since channels are contiguous in row-major:
        // element [0, t, c] = t*512 + c, so channels are interleaved with time)
        // We need each branch as [1, T, 64] with contiguous data.
        // Copy each branch out:
        {
            bf16* src = buf_a.data_as<bf16>();
            for (int b = 0; b < SCALE; ++b) {
                bf16* dst = buf_branch_out[b].data_as<bf16>();
                int offset = b * BR;
                ctx.queue().parallel_for(sycl::range<1>(T * BR), [=](sycl::id<1> idx) {
                    int t = idx / BR;
                    int c = idx % BR;
                    dst[t * BR + c] = src[t * C + offset + c];
                });
            }
        }
        ctx.queue().wait();

        // Branch 0: identity (already copied)
        // Branches 1-7: dilated conv + optional add of previous branch
        for (int b = 1; b < SCALE; ++b) {
            if (b >= 2) {
                // Add previous branch output to current branch input
                bf16* dst = buf_branch_in.data_as<bf16>();
                bf16* br_out = buf_branch_out[b - 1].data_as<bf16>();
                bf16* br_in = buf_branch_out[b].data_as<bf16>();
                ctx.queue().parallel_for(sycl::range<1>(T * BR), [=](sycl::id<1> idx) {
                    dst[idx] = bf16(static_cast<float>(br_in[idx]) + static_cast<float>(br_out[idx]));
                });
                ctx.queue().wait();
                ops::reflect_conv1d(ctx, buf_branch_in, bw.res2net_w[b - 1], bw.res2net_b[b - 1],
                                    buf_branch_out[b], 1, BR, BR, T, 3, dil, dil, true);
            } else {
                // b == 1: conv on raw branch input — must use different in/out tensors
                ops::reflect_conv1d(ctx, buf_branch_out[1], bw.res2net_w[0], bw.res2net_b[0],
                                    buf_branch_in, 1, BR, BR, T, 3, dil, dil, true);
                ops::copy_tensor(ctx, buf_branch_in, buf_branch_out[1], T * BR);
            }
        }

        // Concatenate branches back: 8 × [1, T, 64] → [1, T, 512]
        {
            bf16* dst = buf_concat.data_as<bf16>();
            for (int b = 0; b < SCALE; ++b) {
                bf16* src = buf_branch_out[b].data_as<bf16>();
                int offset = b * BR;
                ctx.queue().parallel_for(sycl::range<1>(T * BR), [=](sycl::id<1> idx) {
                    int t = idx / BR;
                    int c = idx % BR;
                    dst[t * C + offset + c] = src[t * BR + c];
                });
            }
        }
        ctx.queue().wait();

        // --- tdnn2: k=1 ---
        ops::causal_conv1d(ctx, buf_concat, bw.tdnn2_w, bw.tdnn2_b, cur,
                           1, C, C, T, 1, 1);
        ops::relu_inplace(ctx, cur, T * C);

        // --- SE (Squeeze-Excitation) ---
        ops::global_avg_pool_1d(ctx, cur, buf_se_pool, 1, T, C);          // [1, 1, 512]
        ops::causal_conv1d(ctx, buf_se_pool, bw.se_conv1_w, bw.se_conv1_b,
                           buf_se_hidden, 1, C, 128, 1, 1, 1);            // [1, 1, 128]
        ops::relu_inplace(ctx, buf_se_hidden, 128);
        ops::causal_conv1d(ctx, buf_se_hidden, bw.se_conv2_w, bw.se_conv2_b,
                           buf_se_gate, 1, 128, C, 1, 1, 1);              // [1, 1, 512]
        ops::sigmoid_inplace(ctx, buf_se_gate, C);

        // Channel-wise multiply: cur[c, t] *= gate[c]
        {
            bf16* cur_ptr = cur.data_as<bf16>();
            bf16* gate_ptr = buf_se_gate.data_as<bf16>();
            ctx.queue().parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> idx) {
                int t = idx[0], c = idx[1];
                float v = static_cast<float>(cur_ptr[t * C + c]);
                float g = static_cast<float>(gate_ptr[c]);
                cur_ptr[t * C + c] = bf16(v * g);
            });
        }
        ctx.queue().wait();

        // Residual add
        ops::residual_add(ctx, cur, residual, T * C);

        // Save for MFA
        mfa_parts[blk] = Tensor::allocate(ctx, {1, T, C});
        ops::copy_tensor(ctx, cur, mfa_parts[blk], T * C);
    }

    // --- MFA: concat blocks 1, 2, 3 → [1, T, 1536] ---
    Tensor mfa_input = Tensor::allocate(ctx, {1, T, MFA_C});
    {
        bf16* dst = mfa_input.data_as<bf16>();
        for (int b = 0; b < 3; ++b) {
            bf16* src = mfa_parts[b].data_as<bf16>();
            int offset = b * C;
            ctx.queue().parallel_for(sycl::range<1>(T * C), [=](sycl::id<1> idx) {
                int t = idx / C;
                int c = idx % C;
                dst[t * MFA_C + offset + c] = src[t * C + c];
            });
        }
    }
    ctx.queue().wait();

    // MFA conv: [1, T, 1536] → [1, T, 1536], k=1, ReLU
    Tensor mfa_out = Tensor::allocate(ctx, {1, T, MFA_C});
    ops::causal_conv1d(ctx, mfa_input, mfa_w_, mfa_b_, mfa_out,
                       1, MFA_C, MFA_C, T, 1, 1);
    ops::relu_inplace(ctx, mfa_out, T * MFA_C);

    // --- ASP (Attentive Statistics Pooling) ---
    // Global mean and std
    Tensor global_mean = Tensor::allocate(ctx, {1, 1, MFA_C});
    ops::global_avg_pool_1d(ctx, mfa_out, global_mean, 1, T, MFA_C);

    // Global std: compute E[x^2] - E[x]^2 on GPU
    Tensor global_std = Tensor::allocate(ctx, {1, 1, MFA_C});
    {
        bf16* out_ptr = mfa_out.data_as<bf16>();
        bf16* mean_ptr = global_mean.data_as<bf16>();
        bf16* std_ptr = global_std.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(MFA_C), [=](sycl::id<1> c) {
            float sum_sq = 0.0f;
            for (int t = 0; t < T; ++t) {
                float v = static_cast<float>(out_ptr[t * MFA_C + c]);
                sum_sq += v * v;
            }
            float mean_v = static_cast<float>(mean_ptr[c]);
            float var = sum_sq / (float)T - mean_v * mean_v;
            std_ptr[c] = bf16(sycl::sqrt(sycl::fmax(var, 1e-12f)));
        });
    }
    ctx.queue().wait();

    // Concatenate: [mfa_out, global_mean_repeated, global_std_repeated] → [1, T, 4608]
    int ASP_IN = MFA_C * 3; // 4608
    Tensor asp_input = Tensor::allocate(ctx, {1, T, ASP_IN});
    {
        bf16* dst = asp_input.data_as<bf16>();
        bf16* mfa_p = mfa_out.data_as<bf16>();
        bf16* mean_p = global_mean.data_as<bf16>();
        bf16* std_p = global_std.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<2>(T, MFA_C), [=](sycl::id<2> idx) {
            int t = idx[0], c = idx[1];
            dst[t * ASP_IN + c] = mfa_p[t * MFA_C + c];
            dst[t * ASP_IN + MFA_C + c] = mean_p[c];
            dst[t * ASP_IN + 2 * MFA_C + c] = std_p[c];
        });
    }
    ctx.queue().wait();

    // ASP TDNN: [1, T, 4608] → [1, T, 128], k=1, ReLU, then Tanh
    Tensor asp_tdnn_out = Tensor::allocate(ctx, {1, T, 128});
    ops::causal_conv1d(ctx, asp_input, asp_tdnn_w_, asp_tdnn_b_, asp_tdnn_out,
                       1, ASP_IN, 128, T, 1, 1);
    ops::relu_inplace(ctx, asp_tdnn_out, T * 128);
    ops::tanh_inplace(ctx, asp_tdnn_out, T * 128);

    // ASP Conv: [1, T, 128] → [1, T, 1536], k=1, then Softmax over time
    Tensor asp_attn = Tensor::allocate(ctx, {1, T, MFA_C}); // Note: output shape [1, T, 1536]
    ops::causal_conv1d(ctx, asp_tdnn_out, asp_conv_w_, asp_conv_b_, asp_attn,
                       1, 128, MFA_C, T, 1, 1);

    // Softmax over time: reshape to [1, MFA_C, T] for the softmax kernel
    Tensor asp_attn_softmax = Tensor::allocate(ctx, {1, MFA_C, T});
    {
        bf16* src = asp_attn.data_as<bf16>();
        bf16* dst = asp_attn_softmax.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<2>(T, MFA_C), [=](sycl::id<2> idx) {
            int t = idx[0], c = idx[1];
            dst[c * T + t] = src[t * MFA_C + c];
        });
    }
    ctx.queue().wait();
    ops::softmax_1d_time(ctx, asp_attn_softmax, 1, MFA_C, T);

    // Weighted mean and std
    Tensor weighted_mean = Tensor::allocate(ctx, {1, 1, MFA_C});
    Tensor weighted_std  = Tensor::allocate(ctx, {1, 1, MFA_C});
    {
        bf16* attn_p = asp_attn_softmax.data_as<bf16>();
        bf16* mfa_p = mfa_out.data_as<bf16>();
        bf16* mean_p = weighted_mean.data_as<bf16>();
        bf16* std_p = weighted_std.data_as<bf16>();
        ctx.queue().parallel_for(sycl::range<1>(MFA_C), [=](sycl::id<1> c) {
            float w_mean = 0.0f, w_sq = 0.0f;
            for (int t = 0; t < T; ++t) {
                float a = static_cast<float>(attn_p[c * T + t]);
                float v = static_cast<float>(mfa_p[t * MFA_C + c]);
                w_mean += a * v;
            }
            for (int t = 0; t < T; ++t) {
                float a = static_cast<float>(attn_p[c * T + t]);
                float diff = static_cast<float>(mfa_p[t * MFA_C + c]) - w_mean;
                w_sq += a * diff * diff;
            }
            mean_p[c] = bf16(w_mean);
            std_p[c] = bf16(sycl::sqrt(sycl::fmax(w_sq, 1e-12f)));
        });
    }
    ctx.queue().wait();

    // Concatenate: [mean, std] → [1, 1, 3072]
    int FC_IN = MFA_C * 2;
    Tensor pooled = Tensor::allocate(ctx, {1, 1, FC_IN});
    {
        bf16* dst = pooled.data_as<bf16>();
        bf16* mean_p = weighted_mean.data_as<bf16>();
        bf16* std_p = weighted_std.data_as<bf16>();
        ctx.queue().memcpy(dst, mean_p, MFA_C * sizeof(bf16));
        ctx.queue().memcpy(dst + MFA_C, std_p, MFA_C * sizeof(bf16));
    }
    ctx.queue().wait();

    // --- FC: [1, 1, 3072] → [1, 1, 1024], k=1 ---
    Tensor fc_out = Tensor::allocate(ctx, {1, 1, embedding_dim_});
    ops::causal_conv1d(ctx, pooled, fc_w_, fc_b_, fc_out,
                       1, FC_IN, embedding_dim_, 1, 1, 1);

    // Download result
    embedding.resize(embedding_dim_);
    std::vector<bf16> emb_bf16(embedding_dim_);
    ctx.memcpy_d2h(emb_bf16.data(), fc_out.data(), embedding_dim_ * sizeof(bf16));
    for (int i = 0; i < embedding_dim_; ++i) embedding[i] = static_cast<float>(emb_bf16[i]);

    return true;
}

} // namespace audio
} // namespace aila
