#include "engine/Types.hpp"
#include "models/Qwen4ExpTokenGraph.hpp"
#include "models/Qwen4ExpTokenGraphRuntime.hpp"
#include "utils/ModelSpec.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <type_traits>

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

void test_token_graph_pod_contract() {
    static_assert(std::is_trivially_copyable<aila::TokenGraphKey>::value,
                  "TokenGraphKey must remain trivially copyable");
    static_assert(std::is_standard_layout<aila::TokenGraphKey>::value,
                  "TokenGraphKey must remain standard-layout");
    static_assert(std::is_trivially_copyable<aila::TokenGraphLease>::value,
                  "TokenGraphLease must remain trivially copyable");
    static_assert(std::is_standard_layout<aila::TokenGraphLease>::value,
                  "TokenGraphLease must remain standard-layout");

    const aila::TokenGraphKey key{};
    expect(key.backend == nullptr && key.session == nullptr,
           "token graph key must not infer object identity");
    expect(key.session_instance_id == 0 && key.session_generation == 0 &&
               key.session_state_generation == 0 &&
               key.runtime_generation == 0 && key.runtime_capacity == 0 &&
               key.residency_generation == 0 && key.descriptor_generation == 0 &&
               key.ple_workspace_generation == 0 &&
               key.ple_graph_view_generation == 0 &&
               key.topology_generation == 0 && key.topology_flags == 0,
           "token graph key defaults must be zero");
    expect(key.body == aila::TokenGraphBody::QsaLayerProof,
           "token graph body default changed");
    expect(key.attention == aila::TokenGraphAttention::QsaDenseContiguous,
           "token graph attention default changed");

    const aila::TokenGraphLease lease{};
    expect(lease.phase == aila::TokenGraphPhase::Empty &&
               lease.next_sequence == 0 && lease.last_completed_sequence == 0 &&
               !lease.executable_ready && !lease.abort_requested,
           "token graph lease defaults changed");
}

void test_token_graph_sequence_contract() {
    using aila::TokenGraphSequenceStatus;

    expect(aila::token_graph_sequence_status(7, 7) ==
               TokenGraphSequenceStatus::Current,
           "matching sequence should be current");
    expect(aila::token_graph_sequence_status(7, 0) ==
               TokenGraphSequenceStatus::NotPublished,
           "zero sequence should be unpublished");
    expect(aila::token_graph_sequence_status(7, 6) ==
               TokenGraphSequenceStatus::Stale,
           "older sequence should be stale");
    expect(aila::token_graph_sequence_status(7, 8) ==
               TokenGraphSequenceStatus::Ahead,
           "newer sequence should be ahead");
    expect(aila::token_graph_sequence_status(0xfffffffeu, 1u) ==
               TokenGraphSequenceStatus::Ahead,
           "sequence comparison should preserve wrap ordering");
    expect(aila::token_graph_sequence_is_stale(1u, 0xfffffffeu),
           "wrapped older sequence should be stale");
    expect(!aila::token_graph_sequence_is_stale(7, 7),
           "current sequence must not be stale");
}

void test_token_graph_ring_pod_contract() {
    static_assert(alignof(aila::TokenGraphControlRing) == 64);
    static_assert(alignof(aila::TokenGraphHostRing) == 64);
    static_assert(alignof(aila::TokenGraphCpuPacketRing) == 64);
    static_assert(alignof(aila::TokenGraphDeviceEpoch) == 64);
    static_assert(sizeof(aila::TokenGraphHostRing) == 576);
    static_assert(sizeof(aila::TokenGraphCpuPacketRing) == 169088);
    static_assert(sizeof(aila::TokenGraphDeviceEpoch) == 64);

    const aila::TokenGraphHostRing host{};
    expect(host.control.route_sequence == 0 &&
               host.control.cpu_flag == 0 &&
               host.control.result_sequence == 0 &&
               host.control.done_sequence == 0,
           "host ring must zero initialize");

    const aila::TokenGraphCpuPacketRing packet{};
    expect(packet.ready_sequence == 0 && packet.rows == 0 &&
               packet.rank_to_row[0] == 0,
           "CPU packet must zero initialize");

    aila::TokenGraphDeviceEpoch epoch{};
    expect(!aila::token_graph_gate_is_valid(epoch),
           "pending epoch must not be valid");
    epoch.gate_state =
        static_cast<std::uint32_t>(aila::TokenGraphGateState::Valid);
    expect(aila::token_graph_gate_is_valid(epoch),
           "valid epoch rejected");
    epoch.sticky_abort =
        static_cast<std::uint32_t>(aila::TokenGraphGateState::Stale);
    expect(!aila::token_graph_gate_is_valid(epoch),
           "sticky abort must invalidate epoch");
}

void test_token_graph_runtime_state_machine() {
    static_assert(!std::is_copy_constructible<aila::TokenGraphRuntime>::value,
                  "TokenGraphRuntime must not be copy constructible");
    static_assert(!std::is_copy_assignable<aila::TokenGraphRuntime>::value,
                  "TokenGraphRuntime must not be copy assignable");
    static_assert(!std::is_move_constructible<aila::TokenGraphRuntime>::value,
                  "TokenGraphRuntime must not be move constructible");
    static_assert(!std::is_move_assignable<aila::TokenGraphRuntime>::value,
                  "TokenGraphRuntime must not be move assignable");
    static_assert(
        aila::TokenGraphEventStatus::NoEvent !=
            aila::TokenGraphEventStatus::Pending &&
        aila::TokenGraphEventStatus::Pending !=
            aila::TokenGraphEventStatus::Done &&
        aila::TokenGraphEventStatus::Done !=
            aila::TokenGraphEventStatus::Failed,
        "token graph event statuses must remain distinguishable");

    aila::TokenGraphRuntime runtime;
    expect(runtime.phase() == aila::TokenGraphPhase::Empty,
           "runtime must start empty");
    expect(runtime.event_status() == aila::TokenGraphEventStatus::NoEvent,
           "empty runtime must report no event");
    expect(!runtime.MarkExecutableReady(),
           "empty runtime must reject executable publication");
    expect(!runtime.Launch(1, sycl::event{}),
           "launch outside Ready must fail");

    aila::TokenGraphKey key{};
    key.session_instance_id = 41;
    expect(runtime.BeginCapture(key), "Empty -> Capturing must succeed");
    expect(runtime.phase() == aila::TokenGraphPhase::Capturing,
           "runtime did not enter Capturing");
    expect(!runtime.Launch(1, sycl::event{}),
           "launch from Capturing must fail");
    expect(runtime.MarkExecutableReady(),
           "Capturing -> Ready must succeed");
    expect(runtime.executable_ready(), "Ready must expose executable");
    expect(runtime.event_status() == aila::TokenGraphEventStatus::NoEvent,
           "Ready runtime must report no event");

    const aila::TokenGraphLease ready_snapshot = runtime.Snapshot();
    expect(ready_snapshot.phase == aila::TokenGraphPhase::Ready &&
               ready_snapshot.executable_ready &&
               ready_snapshot.next_sequence == 1 &&
               ready_snapshot.last_completed_sequence == 0,
           "Ready lease snapshot mismatch");
    expect(ready_snapshot.key.session_instance_id == 41,
           "lease snapshot lost graph key identity");

    expect(runtime.Launch(1, sycl::event{}),
           "Ready -> InFlight must succeed");
    expect(runtime.phase() == aila::TokenGraphPhase::InFlight,
           "runtime did not enter InFlight");
    const auto default_event_status = runtime.event_status();
    expect(default_event_status == aila::TokenGraphEventStatus::Done ||
               default_event_status == aila::TokenGraphEventStatus::Failed,
           "default event must not be reported as Pending");
    expect(!runtime.BeginCapture(key),
           "recapture while InFlight must fail");
    expect(!runtime.Launch(2, sycl::event{}),
           "second in-flight launch must fail");
    expect(runtime.Complete(1, true),
           "matching successful completion must succeed");
    expect(runtime.phase() == aila::TokenGraphPhase::Ready &&
               runtime.gate_valid() &&
               runtime.last_completed_sequence() == 1 &&
               runtime.next_sequence() == 2,
           "successful completion state mismatch");
    expect(runtime.event_status() == aila::TokenGraphEventStatus::NoEvent,
           "completed runtime must clear its event");
    expect(runtime.Snapshot().last_completed_sequence == 1,
           "completion not reflected in lease snapshot");

    expect(runtime.Launch(2, sycl::event{}),
           "second replay launch must succeed");
    expect(!runtime.Complete(2, false),
           "rejected gate must not complete");
    expect(runtime.phase() == aila::TokenGraphPhase::Aborting &&
               runtime.abort_requested() && !runtime.gate_valid() &&
               !runtime.executable_ready(),
           "rejected gate must enter Aborting");
    expect(!runtime.Launch(3, sycl::event{}),
           "launch from Aborting must fail");
    expect(runtime.Poison(), "Aborting -> Poisoned must succeed");
    expect(!runtime.Poison(), "second Poison must not change state");
    expect(!runtime.BeginCapture(key),
           "Poisoned runtime must reject recapture");
    expect(!runtime.MarkExecutableReady(),
           "Poisoned runtime must reject executable publication");
    expect(!runtime.Launch(3, sycl::event{}),
           "Poisoned runtime must reject launch forever");
    expect(runtime.phase() == aila::TokenGraphPhase::Poisoned,
           "Poison must be permanent");

    aila::TokenGraphRuntime mismatch;
    expect(mismatch.BeginCapture(key) && mismatch.MarkExecutableReady(),
           "mismatch fixture setup failed");
    expect(mismatch.Launch(1, sycl::event{}),
           "mismatch fixture launch failed");
    expect(!mismatch.Complete(2, true),
           "completion sequence mismatch must fail closed");
    expect(mismatch.phase() == aila::TokenGraphPhase::Aborting,
           "completion mismatch must enter Aborting");
    const auto aborted_event_status = mismatch.event_status();
    expect(aborted_event_status == aila::TokenGraphEventStatus::Done ||
               aborted_event_status == aila::TokenGraphEventStatus::Failed,
           "Aborting must retain a launched event for status queries");

    aila::TokenGraphRuntime aborted;
    expect(aborted.BeginCapture(key) && aborted.MarkExecutableReady(),
           "abort fixture setup failed");
    expect(aborted.Abort(), "Ready -> Aborting must succeed");
    expect(aborted.phase() == aila::TokenGraphPhase::Aborting &&
               aborted.abort_requested() &&
               !aborted.executable_ready(),
           "Abort must stop launch and mark the runtime dirty");
    expect(aborted.event_status() == aila::TokenGraphEventStatus::NoEvent,
           "Abort without a launch must report no event");
    expect(!aborted.Abort(), "repeated Abort must not change state");
    expect(aborted.Poison(), "Aborting -> Poisoned must succeed");

    aila::TokenGraphRuntime recapture;
    expect(recapture.BeginCapture(key) && recapture.MarkExecutableReady() &&
               recapture.Launch(1, sycl::event{}) &&
               recapture.Complete(1, true),
           "recapture fixture setup failed");
    aila::TokenGraphKey replacement = key;
    replacement.session_instance_id = 99;
    expect(recapture.BeginCapture(replacement),
           "Ready -> Capturing recapture must succeed");
    expect(recapture.phase() == aila::TokenGraphPhase::Capturing &&
               !recapture.executable_ready() &&
               recapture.next_sequence() == 1 &&
               recapture.last_completed_sequence() == 0,
           "recapture must reset executable and sequence state");
    expect(recapture.Snapshot().key.session_instance_id == 99,
           "recapture must replace the graph key");
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
        test_token_graph_pod_contract();
        test_token_graph_sequence_contract();
        test_token_graph_ring_pod_contract();
        test_token_graph_runtime_state_machine();
        test_real_qwen4exp_modelspec_if_present();
        std::cout << "AilaQwen4ExpModelSpecTests: ALL PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "AilaQwen4ExpModelSpecTests FAILED: " << e.what() << "\n";
        return 1;
    }
}
