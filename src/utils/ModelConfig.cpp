#include "ModelConfig.hpp"
#include "simdjson.h"
#include <fstream>
#include <iterator>
#include <sstream>

namespace aila {
namespace modelcfg {

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

void set_error(std::string* err, const std::string& msg) {
    if (err) *err = msg;
}

bool read_int64(simdjson::dom::element root, const char* key, int& out) {
    simdjson::dom::element v;
    if (root.at_key(key).get(v) != simdjson::SUCCESS) return false;
    int64_t x = 0;
    if (v.get_int64().get(x) != simdjson::SUCCESS) return false;
    out = static_cast<int>(x);
    return true;
}

bool read_double(simdjson::dom::element root, const char* key, float& out) {
    simdjson::dom::element v;
    if (root.at_key(key).get(v) != simdjson::SUCCESS) return false;

    double x = 0.0;
    if (v.get_double().get(x) == simdjson::SUCCESS) {
        out = static_cast<float>(x);
        return true;
    }

    int64_t xi = 0;
    if (v.get_int64().get(xi) == simdjson::SUCCESS) {
        out = static_cast<float>(xi);
        return true;
    }
    return false;
}

bool read_bool(simdjson::dom::element root, const char* key, bool& out) {
    simdjson::dom::element v;
    if (root.at_key(key).get(v) != simdjson::SUCCESS) return false;
    bool b = false;
    if (v.get_bool().get(b) != simdjson::SUCCESS) return false;
    out = b;
    return true;
}

} // namespace

bool load_qwen3_config_from_dir(const std::string& model_dir,
                                Qwen3Config& cfg,
                                std::string* error_message) {
    std::string path = model_dir + "/config.json";
    std::string text = read_text_file(path);
    if (text.empty()) {
        set_error(error_message, "config.json not found or empty: " + path);
        return false;
    }

    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.parse(text);

        // Core architecture
        read_int64(root, "hidden_size", cfg.hidden_size);
        read_int64(root, "num_attention_heads", cfg.num_attention_heads);
        read_int64(root, "num_key_value_heads", cfg.num_key_value_heads);
        read_int64(root, "head_dim", cfg.head_dim);
        read_int64(root, "num_hidden_layers", cfg.num_hidden_layers);
        read_int64(root, "intermediate_size", cfg.intermediate_size);
        read_int64(root, "vocab_size", cfg.vocab_size);
        read_int64(root, "max_position_embeddings", cfg.max_position_embeddings);

        // Numeric hyper-parameters
        read_double(root, "rope_theta", cfg.rope_theta);
        read_double(root, "rms_norm_eps", cfg.rms_norm_eps);

        // Token IDs and flags
        read_bool(root, "tie_word_embeddings", cfg.tie_word_embeddings);
        read_int64(root, "bos_token_id", cfg.bos_token_id);
        read_int64(root, "eos_token_id", cfg.eos_token_id);

        // Fallback: if num_key_value_heads missing, default to full MHA.
        if (cfg.num_key_value_heads <= 0) {
            cfg.num_key_value_heads = cfg.num_attention_heads;
        }

        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("parse config.json failed: ") + e.what());
        return false;
    }
}

} // namespace modelcfg
} // namespace aila
