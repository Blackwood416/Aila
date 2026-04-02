
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

// ============================================================
// Qwen3-0.6B Model Configuration
// ============================================================
struct Qwen3Config {
    int hidden_size           = 1024;
    int num_attention_heads   = 16;     // Q heads
    int num_key_value_heads   = 8;      // KV heads (GQA, 2:1)
    int head_dim              = 128;
    int num_hidden_layers     = 28;
    int intermediate_size     = 3072;   // FFN
    int vocab_size            = 151936;
    int max_position_embeddings = 40960;
    float rope_theta          = 1000000.0f;
    float rms_norm_eps        = 1e-6f;
    bool tie_word_embeddings  = true;

    int bos_token_id = 151643;
    int eos_token_id = 151645;
    int im_start_id  = 151644;
    int im_end_id    = 151645;

    int num_heads_per_kv_group() const { return num_attention_heads / num_key_value_heads; }
};

// ============================================================
// Generation Parameters
// ============================================================
struct GenerationConfig {
    int max_new_tokens      = 512;
    float temperature       = 0.6f;
    int top_k               = 20;
    float top_p             = 0.95f;
    float repetition_penalty = 1.0f;
    bool do_sample          = true;
    int decode_chunk_size   = 1;    // greedy mode: host sync interval (1 = per token)
};
