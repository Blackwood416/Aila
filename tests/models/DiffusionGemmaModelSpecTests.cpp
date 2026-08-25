#include "engine/Types.hpp"
#include "utils/ModelSpec.hpp"
#include "utils/SafeTensors.hpp"
#include "core/Context.hpp"
#include "core/Tensor.hpp"
#include "ops/Bnb4BitLinear.hpp"
#include "ops/Ops.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

void expect(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    if (!output) throw std::runtime_error("failed to write test fixture");
}

std::string diffusion_gemma_config() {
    return R"({
  "architectures": ["DiffusionGemmaForBlockDiffusion"],
  "model_type": "diffusion_gemma",
  "canvas_length": 256,
  "boi_token_id": 255999,
  "eoi_token_id": 258882,
  "image_token_id": 258880,
  "eos_token_id": [1, 106],
  "tie_word_embeddings": true,
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "bnb_4bit_compute_dtype": "bfloat16"
  },
  "text_config": {
    "attention_bias": false,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "final_logit_softcapping": 30.0,
    "global_head_dim": 512,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 2816,
    "intermediate_size": 2112,
    "max_position_embeddings": 262144,
    "moe_intermediate_size": 704,
    "num_attention_heads": 16,
    "num_experts": 128,
    "num_global_key_value_heads": 2,
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional"
      },
      "sliding_attention": {
        "rope_theta": 10000.0,
        "rope_type": "default"
      }
    },
    "sliding_window": 1024,
    "tie_word_embeddings": true,
    "top_k_experts": 8,
    "vocab_size": 262144,
    "layer_types": [
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "full_attention"
    ]
  },
  "vision_config": {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 72,
    "num_hidden_layers": 27,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 10240,
    "rms_norm_eps": 1e-06
  }
})";
}

std::string diffusion_gemma_generation_config() {
    return R"({
  "confidence_threshold": 0.005,
  "eos_token_id": [1, 106, 50],
  "max_denoising_steps": 48,
  "max_new_tokens": 256,
  "pad_token_id": 0,
  "sampler_config": {
    "_cls_name": "EntropyBoundSamplerConfig",
    "entropy_bound": 0.1
  },
  "stability_threshold": 1,
  "t_max": 0.8,
  "t_min": 0.4
})";
}

int run_model_spec_tests() {
    const fs::path root =
        fs::temp_directory_path() / "aila-diffusiongemma-modelspec-tests";
    fs::remove_all(root);
    fs::create_directories(root);

    write_text(root / "config.json", diffusion_gemma_config());
    write_text(root / "generation_config.json",
               diffusion_gemma_generation_config());

    ModelSpec spec;
    std::string error;
    expect(aila::modelspec::load_from_dir(root.string(), spec, &error), error);
    expect(spec.family == ModelFamily::DiffusionGemma,
           "diffusion_gemma family was not detected");
    expect(spec.diffusion_gemma.text.hidden_size == 2816,
           "hidden_size mismatch");
    expect(spec.diffusion_gemma.text.num_hidden_layers == 30,
           "num_hidden_layers mismatch");
    expect(spec.diffusion_gemma.text.global_head_dim == 512,
           "global_head_dim mismatch");
    expect(spec.diffusion_gemma.text.num_global_key_value_heads == 2,
           "global kv heads mismatch");
    expect(spec.diffusion_gemma.text.num_experts == 128 &&
               spec.diffusion_gemma.text.top_k_experts == 8,
           "MoE config mismatch");
    expect(spec.diffusion_gemma.text.layer_types.size() == 30 &&
               spec.diffusion_gemma.text.layer_types[5] == "full_attention" &&
               spec.diffusion_gemma.text.layer_types[29] == "full_attention",
           "layer_types mismatch");
    expect(std::abs(spec.diffusion_gemma.text.rope_full.rope_theta - 1000000.0f) <
                   1.0f &&
               std::abs(spec.diffusion_gemma.text.rope_full.partial_rotary_factor -
                        0.25f) < 1e-6f,
           "full attention rope mismatch");
    expect(std::abs(spec.diffusion_gemma.text.rope_sliding.rope_theta -
                    10000.0f) < 1.0f,
           "sliding attention rope mismatch");
    expect(spec.diffusion_gemma.canvas_length == 256,
           "canvas_length mismatch");
    expect(spec.diffusion_gemma.image_token_id == 258880,
           "image token id mismatch");
    expect(spec.diffusion_gemma.max_denoising_steps == 48 &&
               std::abs(spec.diffusion_gemma.entropy_bound - 0.1f) < 1e-6f &&
               std::abs(spec.diffusion_gemma.t_min - 0.4f) < 1e-6f &&
               std::abs(spec.diffusion_gemma.t_max - 0.8f) < 1e-6f,
           "diffusion generation defaults mismatch");
    expect(spec.diffusion_gemma.eos_token_ids ==
               std::vector<int>({1, 106, 50}),
           "eos token ids mismatch");
    expect(spec.is_bitsandbytes_4bit() &&
               spec.quantization.bnb_4bit_use_double_quant &&
               spec.quantization.bnb_4bit_compute_dtype == "bfloat16",
           "quantization config mismatch");
    expect(spec.diffusion_gemma.vision.enabled &&
               spec.diffusion_gemma.vision.hidden_size == 1152,
           "vision config mismatch");

    std::cout << "DiffusionGemmaModelSpecTests: model spec passed\n";
    return 0;
}

uint8_t nf4_at(const std::vector<uint8_t>& packed, int n, int k, int k_dim) {
    const int byte_index = n * (k_dim / 2) + (k / 2);
    const uint8_t byte = packed[static_cast<size_t>(byte_index)];
    return (k % 2 == 0) ? static_cast<uint8_t>(byte >> 4)
                        : static_cast<uint8_t>(byte & 0x0F);
}

int run_empty_quant_state_loader_test() {
    constexpr int N = 64;
    constexpr int K = 128;
    constexpr int Block = 32;
    constexpr int NestedBlock = 256;

    Context ctx;

    std::vector<float> qmap(16);
    for (int i = 0; i < 16; ++i) {
        qmap[static_cast<size_t>(i)] = (static_cast<float>(i) - 7.5f) * 0.125f;
    }
    std::vector<float> nested_qmap(256);
    for (int i = 0; i < 256; ++i) {
        nested_qmap[static_cast<size_t>(i)] =
            (static_cast<float>(i) - 127.5f) * 0.0078125f;
    }
    const int absmax_count = N * K / Block;
    const int nested_count = (absmax_count + NestedBlock - 1) / NestedBlock;
    std::vector<float> nested_absmax(static_cast<size_t>(nested_count));
    std::vector<uint8_t> absmax_codes(static_cast<size_t>(absmax_count));
    for (int i = 0; i < absmax_count; ++i) {
        nested_absmax[static_cast<size_t>(i / NestedBlock)] =
            0.5f + 0.01f * static_cast<float>(i);
        absmax_codes[static_cast<size_t>(i)] =
            static_cast<uint8_t>(i * 7 % 256);
    }
    std::vector<float> absmax_host(static_cast<size_t>(absmax_count));
    for (int i = 0; i < absmax_count; ++i) {
        absmax_host[static_cast<size_t>(i)] =
            nested_qmap[absmax_codes[static_cast<size_t>(i)]] *
            nested_absmax[static_cast<size_t>(i / NestedBlock)];
    }

    std::vector<uint8_t> packed(static_cast<size_t>(N * K / 2));
    for (int n = 0; n < N; ++n) {
        for (int kh = 0; kh < K / 2; ++kh) {
            const uint8_t hi = static_cast<uint8_t>((n * 3 + kh * 5) & 0x0F);
            const uint8_t lo = static_cast<uint8_t>((n * 7 + kh * 2 + 1) & 0x0F);
            packed[static_cast<size_t>(n * (K / 2) + kh)] =
                static_cast<uint8_t>((hi << 4) | lo);
        }
    }

    const std::string base = "test.weight";
    ModelWeights weights;
    weights.put(base, Tensor::allocate(ctx, {N * K / 2},
                                       dnnl::memory::data_type::u8));
    weights.put(base + ".absmax",
                Tensor::allocate(ctx, {absmax_count},
                                 dnnl::memory::data_type::u8));
    weights.put(base + ".nested_absmax",
                Tensor::allocate(ctx, {nested_count},
                                 dnnl::memory::data_type::f32));
    weights.put(base + ".nested_quant_map",
                Tensor::allocate(ctx, {256}, dnnl::memory::data_type::f32));
    weights.put(base + ".quant_map",
                Tensor::allocate(ctx, {16}, dnnl::memory::data_type::f32));
    ctx.memcpy_h2d(weights.get(base).data(), packed.data(), packed.size());
    ctx.memcpy_h2d(weights.get(base + ".absmax").data(), absmax_codes.data(),
                   absmax_codes.size());
    ctx.memcpy_h2d(weights.get(base + ".nested_absmax").data(),
                   nested_absmax.data(), nested_absmax.size() * sizeof(float));
    ctx.memcpy_h2d(weights.get(base + ".nested_quant_map").data(),
                   nested_qmap.data(), nested_qmap.size() * sizeof(float));
    ctx.memcpy_h2d(weights.get(base + ".quant_map").data(), qmap.data(),
                   qmap.size() * sizeof(float));

    Bnb4BitWeightRef ref;
    std::string error;
    expect(LoadBnb4BitWeightRef(ctx, weights, base, ref, &error, K), error);
    expect(ref.quant_state.shape.size() == 2 &&
               ref.quant_state.shape[0] == N && ref.quant_state.shape[1] == K,
           "inferred shape mismatch");
    expect(ref.quant_state.blocksize == Block, "inferred blocksize mismatch");
    expect(ref.quant_state.nested && ref.quant_state.nested_blocksize == NestedBlock,
           "inferred nested blocksize mismatch");
    expect(ref.quant_state.nested_offset == 0.0f,
           "inferred nested offset mismatch");

    Bnb4BitLinear linear;
    expect(linear.init(ctx, ref, &error), error);
    Bnb4BitLinearScratch scratch;
    scratch.ensure(ctx, 1, K, N, false);

    std::vector<bf16> input(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        input[static_cast<size_t>(k)] =
            bf16(static_cast<float>((k % 17) - 8) * 0.0625f);
    }
    Tensor input_dev = Tensor::allocate(ctx, {1, K},
                                        dnnl::memory::data_type::bf16);
    Tensor output_dev = Tensor::allocate(ctx, {1, N},
                                         dnnl::memory::data_type::bf16);
    ctx.memcpy_h2d(input_dev.data(), input.data(), input.size() * sizeof(bf16));
    linear.forward(ctx, scratch, input_dev, output_dev, 1);
    ctx.synchronize();

    std::vector<bf16> output(static_cast<size_t>(N));
    ctx.memcpy_d2h(output.data(), output_dev.data(), output.size() * sizeof(bf16));

    double max_abs_diff = 0.0;
    bool exceeds_tolerance = false;
    for (int n = 0; n < N; ++n) {
        double expected = 0.0;
        for (int k = 0; k < K; ++k) {
            const uint8_t code = nf4_at(packed, n, k, K);
            expected += static_cast<float>(input[static_cast<size_t>(k)]) *
                        qmap[static_cast<size_t>(code)] *
                        absmax_host[static_cast<size_t>(n * K / Block + k / Block)];
        }
        const double actual = static_cast<float>(output[static_cast<size_t>(n)]);
        const double diff = std::abs(actual - expected);
        max_abs_diff = std::max(max_abs_diff, diff);
        const double tolerance = std::max(0.04, 0.01 * std::abs(expected));
        if (diff > tolerance) exceeds_tolerance = true;
    }
    expect(!exceeds_tolerance, "nested NF4 dequant output mismatch");

    std::cout << "DiffusionGemmaModelSpecTests: empty quant_state loader passed "
                 "(max_abs_diff="
              << max_abs_diff << ")\n";
    return 0;
}

int run_attention_masked_test() {
    constexpr int H = 16;
    constexpr int KV = 2;
    constexpr int Q = 256;
    constexpr int T = 280;
    constexpr int D = 512;
    Context ctx;

    std::vector<bf16> q(static_cast<size_t>(Q) * H * D);
    std::vector<bf16> k(static_cast<size_t>(T) * KV * D);
    std::vector<bf16> v(static_cast<size_t>(T) * KV * D);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = bf16(static_cast<float>((static_cast<int>(i) % 37) - 18) * 0.01f);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = bf16(static_cast<float>((static_cast<int>(i) % 29) - 14) * 0.01f);
        v[i] = bf16(static_cast<float>((static_cast<int>(i) % 23) - 11) * 0.01f);
    }
    std::vector<float> mask(static_cast<size_t>(Q) * T, 0.0f);
    for (int qq = 0; qq < Q; ++qq) {
        for (int t = 0; t < T; ++t) {
            if (t > qq) {
                mask[static_cast<size_t>(qq) * T + t] = -1.0e30f;
            }
        }
    }

    Tensor q_dev = Tensor::allocate(ctx, {Q, H * D}, dnnl::memory::data_type::bf16);
    Tensor k_dev = Tensor::allocate(ctx, {T, KV * D}, dnnl::memory::data_type::bf16);
    Tensor v_dev = Tensor::allocate(ctx, {T, KV * D}, dnnl::memory::data_type::bf16);
    Tensor mask_dev = Tensor::allocate(ctx, {Q, T}, dnnl::memory::data_type::f32);
    Tensor out_dev = Tensor::allocate(ctx, {Q, H * D}, dnnl::memory::data_type::bf16);
    Tensor scores_dev =
        Tensor::allocate(ctx, {H, Q, T}, dnnl::memory::data_type::f32);
    ctx.memcpy_h2d(q_dev.data(), q.data(), q.size() * sizeof(bf16));
    ctx.memcpy_h2d(k_dev.data(), k.data(), k.size() * sizeof(bf16));
    ctx.memcpy_h2d(v_dev.data(), v.data(), v.size() * sizeof(bf16));
    ctx.memcpy_h2d(mask_dev.data(), mask.data(), mask.size() * sizeof(float));

    ops::attention_prefill_masked(ctx, q_dev, k_dev, v_dev, mask_dev, out_dev,
                                  scores_dev, Q, T, H, KV, D);
    ctx.synchronize();

    std::vector<bf16> out(static_cast<size_t>(Q) * H * D);
    ctx.memcpy_d2h(out.data(), out_dev.data(), out.size() * sizeof(bf16));

    // Spot-check against an eager fp32 reference on a few rows.
    double max_diff = 0.0;
    for (int h : {0, 7, 15}) {
        for (int qq : {0, 3, 255}) {
            std::vector<float> scores(static_cast<size_t>(T));
            float max_score = -1.0e30f;
            for (int t = 0; t < T; ++t) {
                float s = 0.0f;
                const int kv_h = h / (H / KV);
                for (int d = 0; d < D; ++d) {
                    s += static_cast<float>(q[static_cast<size_t>(qq) * H * D +
                                              static_cast<size_t>(h) * D + d]) *
                         static_cast<float>(k[static_cast<size_t>(t) * KV * D +
                                              static_cast<size_t>(kv_h) * D + d]);
                }
                if (t > qq) s = -1.0e30f;
                scores[static_cast<size_t>(t)] = s;
                max_score = std::max(max_score, s);
            }
            float sum = 0.0f;
            for (int t = 0; t < T; ++t) {
                scores[static_cast<size_t>(t)] =
                    std::exp(scores[static_cast<size_t>(t)] - max_score);
                sum += scores[static_cast<size_t>(t)];
            }
            for (int d = 0; d < D; ++d) {
                float expected = 0.0f;
                const int kv_h = h / (H / KV);
                for (int t = 0; t < T; ++t) {
                    expected +=
                        scores[static_cast<size_t>(t)] / sum *
                        static_cast<float>(v[static_cast<size_t>(t) * KV * D +
                                             static_cast<size_t>(kv_h) * D + d]);
                }
                const float actual =
                    static_cast<float>(out[static_cast<size_t>(qq) * H * D +
                                           static_cast<size_t>(h) * D + d]);
                const float expected_f = static_cast<float>(expected);
                const float rel = std::abs(actual - expected_f) /
                                  std::max(1.0f, std::abs(expected_f));
                max_diff = std::max(max_diff, static_cast<double>(rel));
            }
        }
    }
    expect(max_diff < 0.1, "attention_prefill_masked head_dim=512 mismatch");
    std::cout << "DiffusionGemmaModelSpecTests: masked attention passed "
                 "(max_rel_diff="
              << max_diff << ")\n";
    return 0;
}

} // namespace

int main() {
    int failed = 0;
    try {
        failed |= run_model_spec_tests();
        failed |= run_empty_quant_state_loader_test();
        failed |= run_attention_masked_test();
    } catch (const std::exception& e) {
        std::cerr << "DiffusionGemmaModelSpecTests: failed: " << e.what() << "\n";
        return 1;
    }
    if (failed != 0) {
        std::cout << "DiffusionGemmaModelSpecTests: failed\n";
        return failed;
    }
    std::cout << "DiffusionGemmaModelSpecTests: passed\n";
    return 0;
}
