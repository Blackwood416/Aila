#include "engine/Types.hpp"
#include "utils/ModelSpec.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace fs = std::filesystem;

namespace {

void expect(bool condition, const std::string& msg) {
    if (!condition) {
        throw std::runtime_error("Test failure: " + msg);
    }
}

std::string qwen4exp_sample_config() {
    return R"({
  "architectures": [
    "Qwen4ExpForConditionalGeneration"
  ],
  "image_token_id": 248056,
  "language_model_only": false,
  "model_type": "qwen4_exp",
  "text_config": {
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 248044,
    "dtype": "bfloat16",
    "eos_token_id": 248044,
    "full_attention_interval": 4,
    "hc_count": 4,
    "hc_lowrank": 320,
    "head_dim": 256,
    "heads_per_ngram": 8,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "initializer_range": 0.02,
    "layer_types": [
      "linear_attention",
      "linear_attention",
      "linear_attention",
      "full_attention"
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "make_ngram_vocab_size_divisible_by": 128,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 262144,
    "model_type": "qwen4_exp_text",
    "moe_intermediate_size": 640,
    "ngram_size": 3,
    "ngram_vocab_size_base": 20000000,
    "num_attention_heads": 24,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_hidden_layers": 48,
    "num_key_value_heads": 2,
    "output_gate_type": "sigmoid",
    "output_router_logits": false,
    "partial_rotary_factor": 0.25,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": 2560,
    "ple_layer_ids": [
      2
    ],
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [
        11,
        11,
        10
      ],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    },
    "router_aux_loss_coef": 0.001,
    "shared_expert_intermediate_size": 640,
    "split_ngram_parts": 128,
    "tie_word_embeddings": false,
    "vocab_size": 248320
  },
  "tie_word_embeddings": false,
  "video_token_id": 248057,
  "vision_config": {
    "depth": 27,
    "hidden_size": 1152,
    "out_hidden_size": 2560,
    "intermediate_size": 4304,
    "num_heads": 16,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2
  },
  "vision_end_token_id": 248054,
  "vision_start_token_id": 248053,
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    "bnb_4bit_compute_dtype": "bfloat16"
  }
})";
}

void test_synthetic_qwen4exp_modelspec() {
    fs::path temp_dir = fs::temp_directory_path() / "aila_qwen4exp_modelspec_test";
    fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);

    {
        std::ofstream out(temp_dir / "config.json");
        out << qwen4exp_sample_config();
    }

    ModelSpec spec;
    std::string err;
    bool ok = aila::modelspec::load_from_dir(temp_dir.string(), spec, &err);
    expect(ok, "failed to load config.json: " + err);
    expect(spec.family == ModelFamily::Qwen4Exp, "model family should be Qwen4Exp");
    expect(spec.model_type == "qwen4_exp", "model_type mismatch");

    const auto& cfg = spec.qwen4exp_text;
    expect(cfg.hidden_size == 2560, "hidden_size mismatch");
    expect(cfg.num_attention_heads == 24, "num_attention_heads mismatch");
    expect(cfg.num_key_value_heads == 2, "num_key_value_heads mismatch");
    expect(cfg.head_dim == 256, "head_dim mismatch");
    expect(cfg.num_hidden_layers == 48, "num_hidden_layers mismatch");
    expect(cfg.vocab_size == 248320, "vocab_size mismatch");
    expect(cfg.bos_token_id == 248044, "bos_token_id mismatch");
    expect(cfg.eos_token_id == 248044, "eos_token_id mismatch");

    // Hyper-Connection
    expect(cfg.hc_count == 4, "hc_count mismatch");
    expect(cfg.hc_lowrank == 320, "hc_lowrank mismatch");

    // GatedDeltaNet
    expect(cfg.linear_num_key_heads == 16, "linear_num_key_heads mismatch");
    expect(cfg.linear_num_value_heads == 48, "linear_num_value_heads mismatch");
    expect(cfg.linear_key_head_dim == 128, "linear_key_head_dim mismatch");
    expect(cfg.linear_value_head_dim == 128, "linear_value_head_dim mismatch");
    expect(cfg.linear_conv_kernel_dim == 4, "linear_conv_kernel_dim mismatch");

    // MoE
    expect(cfg.num_experts == 512, "num_experts mismatch");
    expect(cfg.num_experts_per_tok == 10, "num_experts_per_tok mismatch");
    expect(cfg.moe_intermediate_size == 640, "moe_intermediate_size mismatch");
    expect(cfg.shared_expert_intermediate_size == 640, "shared_expert_intermediate_size mismatch");
    expect(cfg.output_gate_type == "sigmoid", "output_gate_type mismatch");

    // PLE
    expect(cfg.ple_layer_ids.size() == 1 && cfg.ple_layer_ids[0] == 2, "ple_layer_ids mismatch");
    expect(cfg.ple_embed_dim == 2560, "ple_embed_dim mismatch");
    expect(cfg.ple_conv_kernel_size == 4, "ple_conv_kernel_size mismatch");
    expect(cfg.ngram_size == 3, "ngram_size mismatch");
    expect(cfg.heads_per_ngram == 8, "heads_per_ngram mismatch");
    expect(cfg.ngram_vocab_size_base == 20000000, "ngram_vocab_size_base mismatch");
    expect(cfg.make_ngram_vocab_size_divisible_by == 128, "make_ngram_vocab_size_divisible_by mismatch");
    expect(cfg.split_ngram_parts == 128, "split_ngram_parts mismatch");

    // Indexer
    expect(cfg.indexer_n_heads == 4, "indexer_n_heads mismatch");
    expect(cfg.indexer_kv_heads == 1, "indexer_kv_heads mismatch");
    expect(cfg.indexer_head_dim == 128, "indexer_head_dim mismatch");
    expect(cfg.indexer_budget == 2048, "indexer_budget mismatch");
    expect(cfg.indexer_compress_ratio == 4, "indexer_compress_ratio mismatch");

    // RoPE
    expect(cfg.rope.mrope_interleaved == true, "mrope_interleaved mismatch");
    expect(cfg.rope.mrope_section[0] == 11 && cfg.rope.mrope_section[1] == 11 && cfg.rope.mrope_section[2] == 10, "mrope_section mismatch");
    expect(std::abs(cfg.rope.rope_theta - 10000000.0f) < 1.0f, "rope_theta mismatch");
    expect(std::abs(cfg.rope.partial_rotary_factor - 0.25f) < 1e-6f, "partial_rotary_factor mismatch");

    // Vision
    expect(spec.vision.enabled == true, "vision should be enabled");
    expect(spec.vision.depth == 27, "vision depth mismatch");
    expect(spec.vision.hidden_size == 1152, "vision hidden_size mismatch");
    expect(spec.vision.out_hidden_size == 2560, "vision out_hidden_size mismatch");
    expect(spec.vision.intermediate_size == 4304, "vision intermediate_size mismatch");
    expect(spec.vision.num_heads == 16, "vision num_heads mismatch");

    // Quantization
    expect(spec.quantization.load_in_4bit == true, "load_in_4bit mismatch");
    expect(spec.quantization.bnb_4bit_quant_type == "nf4", "quant_type mismatch");
    expect(spec.quantization.bnb_4bit_compute_dtype == "bfloat16", "compute_dtype mismatch");
    expect(spec.quantization.bnb_4bit_use_double_quant == true, "use_double_quant mismatch");

    expect(is_exact_qwen4exp_text_spec(cfg), "is_exact_qwen4exp_text_spec should return true");

    fs::remove_all(temp_dir);
    std::cout << "[Qwen4ExpModelSpecTests] Synthetic fixture passed.\n";
}

static std::string resolve_qwen4exp_model_dir() {
    const char* env_dir = std::getenv("AILA_QWEN4EXP_MODEL_DIR");
    if (env_dir && fs::exists(env_dir)) return env_dir;
    const std::string candidate_c = "C:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_c)) return candidate_c;
    const std::string candidate_f = "F:/QwenModels/Qwen3.8-Flash-Next-BNB-NF4-v2";
    if (fs::exists(candidate_f)) return candidate_f;
    return candidate_c;
}

void test_real_qwen4exp_modelspec_if_present() {
    const std::string model_dir = resolve_qwen4exp_model_dir();
    if (!fs::exists(model_dir) || !fs::exists(model_dir + "/config.json")) {
        std::cout << "[Qwen4ExpModelSpecTests] Real model directory not found, skipping real model spec test.\n";
        return;
    }

    ModelSpec spec;
    std::string err;
    bool ok = aila::modelspec::load_from_dir(model_dir, spec, &err);
    expect(ok, "failed to load real config.json: " + err);
    expect(spec.family == ModelFamily::Qwen4Exp, "real model family should be Qwen4Exp");
    expect(spec.model_type == "qwen4_exp", "real model_type mismatch");

    const auto& cfg = spec.qwen4exp_text;
    expect(cfg.hidden_size == 2560, "real hidden_size mismatch");
    expect(cfg.num_attention_heads == 24, "real num_attention_heads mismatch");
    expect(cfg.num_key_value_heads == 2, "real num_key_value_heads mismatch");
    expect(cfg.head_dim == 256, "real head_dim mismatch");
    expect(cfg.num_hidden_layers == 48, "real num_hidden_layers mismatch");
    expect(cfg.hc_count == 4, "real hc_count mismatch");
    expect(cfg.hc_lowrank == 320, "real hc_lowrank mismatch");
    expect(cfg.num_experts == 512, "real num_experts mismatch");
    expect(cfg.num_experts_per_tok == 10, "real num_experts_per_tok mismatch");
    expect(cfg.layer_types.size() == 48, "real layer_types size mismatch");
    expect(cfg.output_gate_type == "sigmoid", "real output_gate_type mismatch");
    expect(is_exact_qwen4exp_text_spec(cfg), "real is_exact_qwen4exp_text_spec should be true");

    std::cout << "[Qwen4ExpModelSpecTests] Real Qwen4Exp config.json parsed successfully.\n";
}

} // namespace

int main() {
    try {
        test_synthetic_qwen4exp_modelspec();
        test_real_qwen4exp_modelspec_if_present();
        std::cout << "AilaQwen4ExpModelSpecTests: ALL PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AilaQwen4ExpModelSpecTests FAILED: " << e.what() << "\n";
        return 1;
    }
}
