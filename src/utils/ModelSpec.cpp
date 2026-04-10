#include "ModelSpec.hpp"
#include "simdjson.h"
#include <fstream>
#include <iterator>

namespace aila {
namespace modelspec {

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

bool read_float(simdjson::dom::element root, const char* key, float& out) {
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

std::string read_string(simdjson::dom::element root, const char* key,
                        const std::string& fallback = "") {
    simdjson::dom::element v;
    if (root.at_key(key).get(v) != simdjson::SUCCESS) return fallback;
    std::string_view sv;
    if (v.get_string().get(sv) != simdjson::SUCCESS) return fallback;
    return std::string(sv);
}

void parse_qwen3_dense(simdjson::dom::element root, ModelSpec& spec) {
    spec.family = ModelFamily::Qwen3Dense;
    auto& cfg = spec.qwen3;

    read_int64(root, "hidden_size", cfg.hidden_size);
    read_int64(root, "num_attention_heads", cfg.num_attention_heads);
    read_int64(root, "num_key_value_heads", cfg.num_key_value_heads);
    read_int64(root, "head_dim", cfg.head_dim);
    read_int64(root, "num_hidden_layers", cfg.num_hidden_layers);
    read_int64(root, "intermediate_size", cfg.intermediate_size);
    read_int64(root, "vocab_size", cfg.vocab_size);
    read_int64(root, "max_position_embeddings", cfg.max_position_embeddings);
    read_float(root, "rope_theta", cfg.rope_theta);
    read_float(root, "rms_norm_eps", cfg.rms_norm_eps);
    read_bool(root, "tie_word_embeddings", cfg.tie_word_embeddings);
    read_int64(root, "bos_token_id", cfg.bos_token_id);
    read_int64(root, "eos_token_id", cfg.eos_token_id);

    if (cfg.num_key_value_heads <= 0) {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }
}

void parse_qwen35(simdjson::dom::element root, ModelSpec& spec) {
    spec.family = ModelFamily::Qwen35Hybrid;
    auto& cfg = spec.qwen35_text;
    auto& vis = spec.vision;

    read_int64(root, "image_token_id", vis.image_token_id);
    read_int64(root, "video_token_id", vis.video_token_id);
    read_int64(root, "vision_start_token_id", vis.vision_start_token_id);
    read_int64(root, "vision_end_token_id", vis.vision_end_token_id);

    simdjson::dom::element text_cfg_elem;
    if (root.at_key("text_config").get(text_cfg_elem) == simdjson::SUCCESS) {
        read_int64(text_cfg_elem, "hidden_size", cfg.hidden_size);
        read_int64(text_cfg_elem, "num_attention_heads", cfg.num_attention_heads);
        read_int64(text_cfg_elem, "num_key_value_heads", cfg.num_key_value_heads);
        read_int64(text_cfg_elem, "head_dim", cfg.head_dim);
        read_int64(text_cfg_elem, "num_hidden_layers", cfg.num_hidden_layers);
        read_int64(text_cfg_elem, "intermediate_size", cfg.intermediate_size);
        read_int64(text_cfg_elem, "vocab_size", cfg.vocab_size);
        read_int64(text_cfg_elem, "max_position_embeddings", cfg.max_position_embeddings);
        read_float(text_cfg_elem, "rms_norm_eps", cfg.rms_norm_eps);
        read_bool(text_cfg_elem, "tie_word_embeddings", cfg.tie_word_embeddings);
        read_bool(text_cfg_elem, "attn_output_gate", cfg.attn_output_gate);
        read_int64(text_cfg_elem, "linear_num_key_heads", cfg.linear_num_key_heads);
        read_int64(text_cfg_elem, "linear_num_value_heads", cfg.linear_num_value_heads);
        read_int64(text_cfg_elem, "linear_key_head_dim", cfg.linear_key_head_dim);
        read_int64(text_cfg_elem, "linear_value_head_dim", cfg.linear_value_head_dim);
        read_int64(text_cfg_elem, "linear_conv_kernel_dim", cfg.linear_conv_kernel_dim);
        read_int64(text_cfg_elem, "eos_token_id", cfg.eos_token_id);

        simdjson::dom::element rope_elem;
        if (text_cfg_elem.at_key("rope_parameters").get(rope_elem) == simdjson::SUCCESS) {
            read_float(rope_elem, "rope_theta", cfg.rope.rope_theta);
            read_float(rope_elem, "partial_rotary_factor", cfg.rope.partial_rotary_factor);
            read_bool(rope_elem, "mrope_interleaved", cfg.rope.mrope_interleaved);
            simdjson::dom::element section_elem;
            if (rope_elem.at_key("mrope_section").get(section_elem) == simdjson::SUCCESS) {
                auto arr = section_elem.get_array();
                size_t idx = 0;
                for (auto x : arr) {
                    if (idx >= cfg.rope.mrope_section.size()) break;
                    int64_t v = 0;
                    if (x.get_int64().get(v) == simdjson::SUCCESS) {
                        cfg.rope.mrope_section[idx] = static_cast<int>(v);
                    }
                    idx++;
                }
            }
        }

        cfg.layer_types.clear();
        simdjson::dom::element layer_types_elem;
        if (text_cfg_elem.at_key("layer_types").get(layer_types_elem) == simdjson::SUCCESS) {
            for (auto x : layer_types_elem.get_array()) {
                std::string_view sv;
                if (x.get_string().get(sv) == simdjson::SUCCESS) {
                    cfg.layer_types.emplace_back(sv);
                }
            }
        }
    }

    simdjson::dom::element vis_cfg_elem;
    if (root.at_key("vision_config").get(vis_cfg_elem) == simdjson::SUCCESS) {
        vis.enabled = true;
        read_int64(vis_cfg_elem, "depth", vis.depth);
        read_int64(vis_cfg_elem, "hidden_size", vis.hidden_size);
        read_int64(vis_cfg_elem, "out_hidden_size", vis.out_hidden_size);
        read_int64(vis_cfg_elem, "intermediate_size", vis.intermediate_size);
        read_int64(vis_cfg_elem, "num_heads", vis.num_heads);
        read_int64(vis_cfg_elem, "patch_size", vis.patch_size);
        read_int64(vis_cfg_elem, "temporal_patch_size", vis.temporal_patch_size);
        read_int64(vis_cfg_elem, "spatial_merge_size", vis.spatial_merge_size);
    }
}

} // namespace

bool load_from_dir(const std::string& model_dir,
                   ModelSpec& spec,
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

        spec.model_type = read_string(root, "model_type", "qwen3");
        if (spec.model_type == "qwen3_5") {
            parse_qwen35(root, spec);
        } else {
            parse_qwen3_dense(root, spec);
        }
        return true;
    } catch (const std::exception& e) {
        set_error(error_message, std::string("parse config.json failed: ") + e.what());
        return false;
    }
}

} // namespace modelspec
} // namespace aila
