
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <array>

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
// Generic chat/message types (OpenAI-style content parts)
// ============================================================
enum class ContentType {
    Text,
    Image,
    Video
};

struct ContentPart {
    ContentType type = ContentType::Text;
    std::string text;
    std::string uri;
};

struct Message {
    std::string role;  // "system" | "user" | "assistant" | "tool"
    std::vector<ContentPart> content;
};

enum class ModelFamily {
    Qwen3Dense,
    Qwen35Hybrid,
    Unknown
};

struct RopeSpec {
    float rope_theta = 1000000.0f;
    float partial_rotary_factor = 1.0f;
    bool mrope_interleaved = false;
    std::array<int, 3> mrope_section = {0, 0, 0};
};

struct Qwen35TextConfig {
    int hidden_size = 1024;
    int num_attention_heads = 8;
    int num_key_value_heads = 2;
    int head_dim = 256;
    int num_hidden_layers = 24;
    int intermediate_size = 3584;
    int vocab_size = 248320;
    int max_position_embeddings = 262144;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = true;
    bool attn_output_gate = true;

    int linear_num_key_heads = 16;
    int linear_num_value_heads = 16;
    int linear_key_head_dim = 128;
    int linear_value_head_dim = 128;
    int linear_conv_kernel_dim = 4;

    int eos_token_id = 248044;
    RopeSpec rope{};
    std::vector<std::string> layer_types;
};

inline bool is_exact_qwen35_hybrid_0p8b_spec(const Qwen35TextConfig& cfg) {
    return cfg.hidden_size == 1024 &&
           cfg.num_attention_heads == 8 &&
           cfg.num_key_value_heads == 2 &&
           cfg.head_dim == 256 &&
           cfg.num_hidden_layers == 24 &&
           cfg.intermediate_size == 3584 &&
           cfg.linear_num_key_heads == 16 &&
           cfg.linear_num_value_heads == 16 &&
           cfg.linear_key_head_dim == 128 &&
           cfg.linear_value_head_dim == 128 &&
           cfg.linear_conv_kernel_dim == 4;
}

inline bool is_supported_qwen35_hybrid_0p8b_spec(const Qwen35TextConfig& cfg) {
    return is_exact_qwen35_hybrid_0p8b_spec(cfg);
}

struct VisionConfig {
    bool enabled = false;
    int image_token_id = -1;
    int video_token_id = -1;
    int vision_start_token_id = -1;
    int vision_end_token_id = -1;

    int depth = 0;
    int hidden_size = 0;
    int out_hidden_size = 0;
    int intermediate_size = 0;
    int num_heads = 0;
    int patch_size = 0;
    int temporal_patch_size = 0;
    int spatial_merge_size = 0;
};

struct ModelSpec {
    ModelFamily family = ModelFamily::Qwen3Dense;
    std::string model_type;

    // Legacy/dense path
    Qwen3Config qwen3{};

    // Qwen3.5 path
    Qwen35TextConfig qwen35_text{};
    VisionConfig vision{};

    bool has_vision() const { return vision.enabled; }
};

enum class EngineErrorCode {
    Ok = 0,
    InvalidArgument = 1,
    TemplateError = 2,
    JsonParseError = 3,
    VisionNotEnabled = 4,
    ContextOverflow = 5,
    RuntimeError = 6,
};

// ============================================================
// Generation Parameters
// ============================================================
struct GenerationConfig {
    int max_new_tokens      = 512;
    float temperature       = 0.6f;
    int top_k               = 20;
    float top_p             = 0.95f;
    bool do_sample          = true;
    uint64_t sampling_seed  = 42;   // used when use_fixed_seed=true
    bool use_fixed_seed     = false;
    int decode_chunk_size   = 12;   // greedy + non-streaming: tokens per host sync chunk
    int stream_chunk_size   = 4;    // greedy + streaming: tokens per flush chunk

    // Penalty parameters
    float repetition_penalty = 1.0f;   // > 1.0 penalizes repeated tokens multiplicatively
    float presence_penalty   = 0.0f;   // > 0.0 penalizes any token that has appeared
    float frequency_penalty  = 0.0f;   // > 0.0 penalizes based on how often token appeared

    bool has_penalties() const {
        return repetition_penalty != 1.0f || presence_penalty != 0.0f || frequency_penalty != 0.0f;
    }
};

// ============================================================
// Chat History for multi-turn conversation
// ============================================================
struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

class ChatHistory {
public:
    void add(const std::string& role, const std::string& content) {
        messages_.push_back({role, content});
    }

    void add_user(const std::string& content) { add("user", content); }
    void add_assistant(const std::string& content) { add("assistant", content); }

    void clear() { messages_.clear(); }

    bool empty() const { return messages_.empty(); }
    size_t size() const { return messages_.size(); }

    const std::vector<ChatMessage>& messages() const { return messages_; }

    // Remove oldest user+assistant pairs to fit within token budget.
    // Keeps at least the last pair. Returns number of messages removed.
    int truncate_oldest(int max_messages) {
        if (static_cast<int>(messages_.size()) <= max_messages || max_messages < 1) return 0;
        int to_remove = static_cast<int>(messages_.size()) - max_messages;
        // Always remove in pairs (user+assistant) to keep conversation coherent
        // Round up to even number
        to_remove = ((to_remove + 1) / 2) * 2;
        if (to_remove >= static_cast<int>(messages_.size())) {
            to_remove = static_cast<int>(messages_.size()) - 1; // keep at least 1
        }
        if (to_remove > 0) {
            messages_.erase(messages_.begin(), messages_.begin() + to_remove);
        }
        return to_remove;
    }

private:
    std::vector<ChatMessage> messages_;
};
