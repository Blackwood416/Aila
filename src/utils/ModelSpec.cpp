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

void parse_quantization(simdjson::dom::element root, ModelSpec& spec) {
    spec.quantization = {};

    simdjson::dom::element quant_elem;
    if (root.at_key("quantization_config").get(quant_elem) != simdjson::SUCCESS) {
        return;
    }

    auto& quant = spec.quantization;
    quant.quant_method = read_string(quant_elem, "quant_method");
    if (!read_bool(quant_elem, "load_in_4bit", quant.load_in_4bit)) {
        read_bool(quant_elem, "_load_in_4bit", quant.load_in_4bit);
    }
    if (!read_bool(quant_elem, "load_in_8bit", quant.load_in_8bit)) {
        read_bool(quant_elem, "_load_in_8bit", quant.load_in_8bit);
    }
    quant.bnb_4bit_quant_type = read_string(quant_elem, "bnb_4bit_quant_type");
    quant.bnb_4bit_compute_dtype = read_string(quant_elem, "bnb_4bit_compute_dtype");
    quant.bnb_4bit_quant_storage = read_string(quant_elem, "bnb_4bit_quant_storage");
    read_bool(quant_elem, "bnb_4bit_use_double_quant", quant.bnb_4bit_use_double_quant);
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

void parse_qwen35_text_config(simdjson::dom::element text_cfg_elem, Qwen35TextConfig& cfg) {
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

void parse_qwen3_asr(simdjson::dom::element root, ModelSpec& spec) {
    spec.family = ModelFamily::Qwen3ASR;

    // Navigate into thinker_config
    simdjson::dom::element thinker;
    if (root.at_key("thinker_config").get(thinker) != simdjson::SUCCESS) {
        thinker = root; // fallback: try reading from root
    }

    // --- Audio config ---
    simdjson::dom::element audio_cfg_elem;
    if (thinker.at_key("audio_config").get(audio_cfg_elem) == simdjson::SUCCESS) {
        auto& ac = spec.audio;
        read_int64(audio_cfg_elem, "d_model", ac.d_model);
        read_int64(audio_cfg_elem, "encoder_attention_heads", ac.encoder_attention_heads);
        read_int64(audio_cfg_elem, "encoder_ffn_dim", ac.encoder_ffn_dim);
        read_int64(audio_cfg_elem, "encoder_layers", ac.encoder_layers);
        // encoder_layers might be stored as "num_hidden_layers"
        if (ac.encoder_layers <= 0) {
            read_int64(audio_cfg_elem, "num_hidden_layers", ac.encoder_layers);
        }
        read_int64(audio_cfg_elem, "output_dim", ac.output_dim);
        read_int64(audio_cfg_elem, "num_mel_bins", ac.num_mel_bins);
        read_int64(audio_cfg_elem, "downsample_hidden_size", ac.downsample_hidden_size);
        read_int64(audio_cfg_elem, "n_window", ac.n_window);
        read_int64(audio_cfg_elem, "n_window_infer", ac.n_window_infer);
        read_int64(audio_cfg_elem, "conv_chunksize", ac.conv_chunksize);
        read_int64(audio_cfg_elem, "max_source_positions", ac.max_source_positions);
        ac.head_dim = ac.d_model / ac.encoder_attention_heads;
    }

    // --- Audio token IDs ---
    read_int64(thinker, "audio_token_id", spec.audio_token_id);
    read_int64(thinker, "audio_start_token_id", spec.audio_start_token_id);
    read_int64(thinker, "audio_end_token_id", spec.audio_end_token_id);

    // --- Detect forced_aligner subtype ---
    {
        std::string thinker_type = read_string(thinker, "model_type");
        if (thinker_type == "qwen3_forced_aligner") {
            spec.family = ModelFamily::Qwen3ForceAligner;
            read_int64(thinker, "classify_num", spec.classify_num);
        }
    }

    // --- Timestamp config (top-level, for forced_aligner) ---
    read_int64(root, "timestamp_token_id", spec.timestamp_token_id);
    read_int64(root, "timestamp_segment_time", spec.timestamp_segment_time);

    // --- Text config ---
    simdjson::dom::element text_cfg_elem;
    bool has_text_config = (thinker.at_key("text_config").get(text_cfg_elem) == simdjson::SUCCESS);
    auto& tc = spec.qwen3;

    // Read from text_config sub-object, or from thinker_config root if flat
    auto read_text_int = [&](const char* key, int& out) {
        if (has_text_config && read_int64(text_cfg_elem, key, out)) return;
        read_int64(thinker, key, out);
    };
    auto read_text_float = [&](const char* key, float& out) {
        if (has_text_config && read_float(text_cfg_elem, key, out)) return;
        read_float(thinker, key, out);
    };
    auto read_text_bool = [&](const char* key, bool& out) {
        if (has_text_config && read_bool(text_cfg_elem, key, out)) return;
        read_bool(thinker, key, out);
    };

    read_text_int("hidden_size", tc.hidden_size);
    read_text_int("num_attention_heads", tc.num_attention_heads);
    read_text_int("num_key_value_heads", tc.num_key_value_heads);
    read_text_int("head_dim", tc.head_dim);
    read_text_int("num_hidden_layers", tc.num_hidden_layers);
    read_text_int("intermediate_size", tc.intermediate_size);
    read_text_int("vocab_size", tc.vocab_size);
    read_text_int("max_position_embeddings", tc.max_position_embeddings);
    read_text_float("rope_theta", tc.rope_theta);
    read_text_float("rms_norm_eps", tc.rms_norm_eps);
    read_text_bool("tie_word_embeddings", tc.tie_word_embeddings);

    read_text_int("bos_token_id", tc.bos_token_id);
    read_text_int("eos_token_id", tc.eos_token_id);
    if (tc.eos_token_id <= 0) tc.eos_token_id = 151643;
    if (tc.bos_token_id <= 0) tc.bos_token_id = 151643;
    tc.im_start_id = 151644;
    tc.im_end_id = tc.eos_token_id;

    if (tc.num_key_value_heads <= 0) {
        tc.num_key_value_heads = tc.num_attention_heads;
    }

    // --- MRoPE config from text_config.rope_scaling ---
    {
        simdjson::dom::element rope_scaling_elem;
        simdjson::dom::element* rope_parent = &text_cfg_elem;
        if (!has_text_config) rope_parent = &thinker;
        if (rope_parent->at_key("rope_scaling").get(rope_scaling_elem) == simdjson::SUCCESS) {
            read_bool(rope_scaling_elem, "mrope_interleaved", tc.rope.mrope_interleaved);
            simdjson::dom::element section_elem;
            if (rope_scaling_elem.at_key("mrope_section").get(section_elem) == simdjson::SUCCESS) {
                auto arr = section_elem.get_array();
                size_t idx = 0;
                for (auto x : arr) {
                    if (idx >= tc.rope.mrope_section.size()) break;
                    int64_t v = 0;
                    if (x.get_int64().get(v) == simdjson::SUCCESS) {
                        tc.rope.mrope_section[idx] = static_cast<int>(v);
                    }
                    idx++;
                }
            }
        }
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
        parse_qwen35_text_config(text_cfg_elem, cfg);
    } else {
        parse_qwen35_text_config(root, cfg);
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

void parse_qwen3_tts(simdjson::dom::element root, ModelSpec& spec) {
    spec.family = ModelFamily::Qwen3TTS;

    // Parse TTS model type from top-level config
    {
        std::string_view ttm_type;
        if (root.at_key("tts_model_type").get(ttm_type) == simdjson::SUCCESS) {
            if (ttm_type == "custom_voice") {
                spec.qwen3.tts_model_type = Qwen3TTSModelType::CustomVoice;
            } else if (ttm_type == "voice_design") {
                spec.qwen3.tts_model_type = Qwen3TTSModelType::VoiceDesign;
            }
            // "base" is already the default
        }
    }

    simdjson::dom::element talker;
    if (root.at_key("talker_config").get(talker) != simdjson::SUCCESS) {
        talker = root;
    }

    auto& tc = spec.qwen3;
    read_int64(talker, "hidden_size", tc.hidden_size);
    read_int64(talker, "num_attention_heads", tc.num_attention_heads);
    read_int64(talker, "num_key_value_heads", tc.num_key_value_heads);
    read_int64(talker, "head_dim", tc.head_dim);
    read_int64(talker, "num_hidden_layers", tc.num_hidden_layers);
    read_int64(talker, "intermediate_size", tc.intermediate_size);
    read_int64(talker, "vocab_size", tc.vocab_size);
    read_int64(talker, "max_position_embeddings", tc.max_position_embeddings);
    read_float(talker, "rope_theta", tc.rope_theta);
    read_float(talker, "rms_norm_eps", tc.rms_norm_eps);
    read_bool(talker, "tie_word_embeddings", tc.tie_word_embeddings);
    read_int64(talker, "num_code_groups", tc.num_code_groups);
    read_int64(talker, "codec_eos_token_id", tc.codec_eos_token_id);
    read_int64(talker, "codec_think_id", tc.codec_think_id);
    read_int64(talker, "codec_nothink_id", tc.codec_nothink_id);
    read_int64(talker, "codec_think_bos_id", tc.codec_think_bos_id);
    read_int64(talker, "codec_think_eos_id", tc.codec_think_eos_id);
    read_int64(talker, "codec_pad_id", tc.codec_pad_id);
    read_int64(talker, "codec_bos_id", tc.codec_bos_id);
    read_int64(talker, "text_hidden_size", tc.text_hidden_size);

    // Parse spk_id dictionary from talker_config
    {
        simdjson::dom::element spk_id_obj;
        if (talker.at_key("spk_id").get(spk_id_obj) == simdjson::SUCCESS) {
            auto obj = spk_id_obj.get_object();
            for (auto field : obj) {
                int64_t val = 0;
                if (field.value.get_int64().get(val) == simdjson::SUCCESS) {
                    spec.qwen3.spk_id[std::string(field.key.data(), field.key.size())] = static_cast<int>(val);
                }
            }
        }
    }

    // Parse spk_is_dialect dictionary from talker_config
    {
        simdjson::dom::element dialect_obj;
        if (talker.at_key("spk_is_dialect").get(dialect_obj) == simdjson::SUCCESS) {
            auto obj = dialect_obj.get_object();
            for (auto field : obj) {
                std::string_view val_sv;
                if (field.value.get_string().get(val_sv) == simdjson::SUCCESS) {
                    spec.qwen3.spk_is_dialect[std::string(field.key.data(), field.key.size())]
                        = std::string(val_sv.data(), val_sv.size());
                }
            }
        }
    }

    // Parse codec_language_id dictionary from talker_config
    {
        simdjson::dom::element lang_obj;
        if (talker.at_key("codec_language_id").get(lang_obj) == simdjson::SUCCESS) {
            auto obj = lang_obj.get_object();
            for (auto field : obj) {
                int64_t val = 0;
                if (field.value.get_int64().get(val) == simdjson::SUCCESS) {
                    spec.qwen3.codec_language_id[std::string(field.key.data(), field.key.size())] = static_cast<int>(val);
                }
            }
        }
    }

    if (tc.num_key_value_heads <= 0) {
        tc.num_key_value_heads = tc.num_attention_heads;
    }

    // Parse predictor config
    simdjson::dom::element predictor;
    if (talker.at_key("code_predictor_config").get(predictor) == simdjson::SUCCESS) {
        auto& pc = spec.code_predictor;
        read_int64(predictor, "hidden_size", pc.hidden_size);
        read_int64(predictor, "num_attention_heads", pc.num_attention_heads);
        read_int64(predictor, "num_key_value_heads", pc.num_key_value_heads);
        read_int64(predictor, "head_dim", pc.head_dim);
        read_int64(predictor, "num_hidden_layers", pc.num_hidden_layers);
        read_int64(predictor, "intermediate_size", pc.intermediate_size);
        read_int64(predictor, "vocab_size", pc.vocab_size);
        read_float(predictor, "rope_theta", pc.rope_theta);
        read_float(predictor, "rms_norm_eps", pc.rms_norm_eps);
        read_bool(predictor, "tie_word_embeddings", pc.tie_word_embeddings);
        read_int64(predictor, "num_code_groups", pc.num_code_groups);
        read_int64(predictor, "text_hidden_size", pc.text_hidden_size);
        if (pc.num_key_value_heads <= 0) {
            pc.num_key_value_heads = pc.num_attention_heads;
        }
    }

    // Parse MRoPE configuration
    {
        simdjson::dom::element rope_scaling_elem;
        if (talker.at_key("rope_scaling").get(rope_scaling_elem) == simdjson::SUCCESS) {
            read_bool(rope_scaling_elem, "mrope_interleaved", tc.rope.mrope_interleaved);
            simdjson::dom::element section_elem;
            if (rope_scaling_elem.at_key("mrope_section").get(section_elem) == simdjson::SUCCESS) {
                auto arr = section_elem.get_array();
                size_t idx = 0;
                for (auto x : arr) {
                    if (idx >= tc.rope.mrope_section.size()) break;
                    int64_t v = 0;
                    if (x.get_int64().get(v) == simdjson::SUCCESS) {
                        tc.rope.mrope_section[idx] = static_cast<int>(v);
                    }
                    idx++;
                }
            }
        }
    }

    // Parse TTS special token IDs from top-level config
    read_int64(root, "tts_pad_token_id", spec.qwen3.tts_pad_token_id);
    read_int64(root, "tts_bos_token_id", spec.qwen3.tts_bos_token_id);
    read_int64(root, "tts_eos_token_id", spec.qwen3.tts_eos_token_id);
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
        parse_quantization(root, spec);
        if (spec.model_type == "qwen3_asr") {
            parse_qwen3_asr(root, spec);
        } else if (spec.model_type == "qwen3_tts") {
            parse_qwen3_tts(root, spec);
        } else if (spec.model_type == "qwen3_5" || spec.model_type == "qwen3_5_text") {
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
