
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
    bool do_sample          = true;
    int decode_chunk_size   = 1;    // greedy + non-streaming: host sync interval
    int stream_chunk_size   = 1;    // greedy + streaming: tokens per flush chunk

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
