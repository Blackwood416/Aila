#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/IModelBackend.hpp"
#include "../src/models/Qwen3DenseBackend.hpp"
#include "../src/models/Qwen3Bnb4Backend.hpp"
#include "../src/models/Qwen35HybridBnb4Backend.hpp"
#include "../src/models/Qwen35HybridTextBackend.hpp"
#include "../src/models/Qwen3ASRBackend.hpp"
#include "../src/models/Qwen3ASRBnb4Backend.hpp"
#include "../src/models/Qwen3TTSBackend.hpp"
#include "../src/models/Qwen3ForceAlignerBackend.hpp"
#include "../src/models/Qwen3ForceAlignerBnb4Backend.hpp"
#include "../src/vision/Qwen35VisionEncoder.hpp"
#include "../src/vision/Yolo26Detector.hpp"
#include "../src/audio/Qwen3ASRAudioEncoder.hpp"
#include "../src/audio/AudioPreprocessor.hpp"
#include "../src/audio/SpeakerEncoder.hpp"
#include "../src/audio/MimiEncoder.hpp"
#include "../src/utils/ForceAlignerPostProcess.hpp"
#include "../src/audio/GpuSpeakerEncoder.hpp"
#include "../src/lora/LoraLoader.hpp"
#include "../src/lora/LoraConfig.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/ModelConfig.hpp"
#include "../src/utils/ModelSpec.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "../src/chat/AssistantOutputParser.hpp"
#include "../src/chat/ChatFormatter.hpp"
#include "../src/chat/ChatJson.hpp"
#include "../src/chat/ChatSessionState.hpp"
#include "../src/chat/ChatStreamJson.hpp"
#include "../src/chat/StructuredStreamParser.hpp"
#include "../src/chat/ThinkingBudgetController.hpp"
#include "../src/chat/ToolPolicy.hpp"
#include "../src/profile/Profiling.hpp"
#include "../src/utils/EnvUtils.hpp"
#include "../src/utils/JsonParser.hpp"
#include "Types.hpp"
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cctype>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <cstddef>
#include <utility>

namespace aila_asr {

inline std::vector<uint32_t> utf8_to_utf32(const std::string& text) {
    std::vector<uint32_t> out;
    size_t offset = 0;
    while (offset < text.size()) {
        unsigned char c0 = static_cast<unsigned char>(text[offset]);
        uint32_t cp = c0;
        size_t bytes = 1;
        if (c0 >= 0x80) {
            auto cont = [&](size_t idx) -> uint32_t {
                if (offset + idx < text.size()) {
                    return static_cast<unsigned char>(text[offset + idx]) & 0x3Fu;
                }
                return 0;
            };
            if ((c0 & 0xE0) == 0xC0 && offset + 1 < text.size()) {
                cp = ((c0 & 0x1Fu) << 6) | cont(1);
                bytes = 2;
            } else if ((c0 & 0xF0) == 0xE0 && offset + 2 < text.size()) {
                cp = ((c0 & 0x0Fu) << 12) | (cont(1) << 6) | cont(2);
                bytes = 3;
            } else if ((c0 & 0xF8) == 0xF0 && offset + 3 < text.size()) {
                cp = ((c0 & 0x07u) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3);
                bytes = 4;
            }
        }
        out.push_back(cp);
        offset += bytes;
    }
    return out;
}

inline std::string utf32_to_utf8(const std::vector<uint32_t>& u32) {
    std::string out;
    for (uint32_t cp : u32) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

inline std::vector<uint32_t> fix_char_repeats(const std::vector<uint32_t>& s, int thresh) {
    std::vector<uint32_t> res;
    int i = 0;
    int n = static_cast<int>(s.size());
    while (i < n) {
        int count = 1;
        while (i + count < n && s[i + count] == s[i]) {
            count++;
        }
        if (count > thresh) {
            res.push_back(s[i]);
        } else {
            for (int k = 0; k < count; ++k) {
                res.push_back(s[i + k]);
            }
        }
        i += count;
    }
    return res;
}

inline std::vector<uint32_t> fix_pattern_repeats(const std::vector<uint32_t>& s, int thresh, int max_len = 20) {
    int n = static_cast<int>(s.size());
    int min_repeat_chars = thresh * 2;
    if (n < min_repeat_chars) {
        return s;
    }

    int i = 0;
    std::vector<uint32_t> result;
    bool found = false;

    while (i <= n - min_repeat_chars) {
        found = false;
        for (int k = 1; k <= max_len; ++k) {
            if (i + k * thresh > n) {
                break;
            }

            bool valid = true;
            for (int rep = 1; rep < thresh; ++rep) {
                int start_idx = i + rep * k;
                for (int p_idx = 0; p_idx < k; ++p_idx) {
                    if (s[start_idx + p_idx] != s[i + p_idx]) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }

            if (valid) {
                int total_rep = thresh;
                int end_index = i + thresh * k;
                while (end_index + k <= n) {
                    bool match = true;
                    for (int p_idx = 0; p_idx < k; ++p_idx) {
                        if (s[end_index + p_idx] != s[i + p_idx]) {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                    total_rep++;
                    end_index += k;
                }

                for (int p_idx = 0; p_idx < k; ++p_idx) {
                    result.push_back(s[i + p_idx]);
                }

                std::vector<uint32_t> rest(s.begin() + end_index, s.end());
                std::vector<uint32_t> rest_fixed = fix_pattern_repeats(rest, thresh, max_len);
                result.insert(result.end(), rest_fixed.begin(), rest_fixed.end());

                i = n;
                found = true;
                break;
            }
        }

        if (found) {
            break;
        } else {
            result.push_back(s[i]);
            i++;
        }
    }

    if (!found) {
        for (int idx = i; idx < n; ++idx) {
            result.push_back(s[idx]);
        }
    }

    return result;
}

inline std::string detect_and_fix_repetitions(const std::string& text, int threshold = 20) {
    std::vector<uint32_t> u32 = utf8_to_utf32(text);
    std::vector<uint32_t> fixed_char = fix_char_repeats(u32, threshold);
    std::vector<uint32_t> fixed_pattern = fix_pattern_repeats(fixed_char, threshold);
    return utf32_to_utf8(fixed_pattern);
}

inline std::string normalize_language_name(const std::string& language) {
    if (language.empty()) return "";
    std::string s = language;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (s.empty()) return "";

    s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    for (size_t i = 1; i < s.size(); ++i) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
    }
    return s;
}

inline std::string to_lowercase(const std::string& str) {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return s;
}

inline void parse_asr_output(const std::string& raw, const std::string& user_language, std::string& language_out, std::string& text_out) {
    language_out = "";
    text_out = "";
    if (raw.empty()) return;

    std::string s = raw;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(0, 1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (s.empty()) return;

    s = detect_and_fix_repetitions(s);

    if (!user_language.empty()) {
        language_out = normalize_language_name(user_language);
        text_out = s;
        return;
    }

    const std::string tag = "<asr_text>";
    size_t tag_pos = s.find(tag);

    std::string meta_part;
    std::string text_part;
    if (tag_pos != std::string::npos) {
        meta_part = s.substr(0, tag_pos);
        text_part = s.substr(tag_pos + tag.length());
    } else {
        text_out = s;
        while (!text_out.empty() && std::isspace(static_cast<unsigned char>(text_out.front()))) text_out.erase(0, 1);
        while (!text_out.empty() && std::isspace(static_cast<unsigned char>(text_out.back()))) text_out.pop_back();
        return;
    }

    std::string meta_lower = to_lowercase(meta_part);

    if (meta_lower.find("language none") != std::string::npos) {
        std::string t = text_part;
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.front()))) t.erase(0, 1);
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
        if (t.empty()) {
            return;
        }
        text_out = t;
        return;
    }

    std::string lang = "";
    std::string prefix = "language ";
    size_t line_start = 0;
    while (line_start < meta_part.size()) {
        size_t line_end = meta_part.find('\n', line_start);
        if (line_end == std::string::npos) {
            line_end = meta_part.size();
        }
        std::string line = meta_part.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(0, 1);
        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
        if (line.empty()) continue;

        std::string low = to_lowercase(line);
        if (low.rfind(prefix, 0) == 0) {
            std::string val = line.substr(prefix.length());
            while (!val.empty() && std::isspace(static_cast<unsigned char>(val.front()))) val.erase(0, 1);
            while (!val.empty() && std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
            if (!val.empty()) {
                lang = normalize_language_name(val);
            }
            break;
        }
    }

    while (!text_part.empty() && std::isspace(static_cast<unsigned char>(text_part.front()))) text_part.erase(0, 1);
    while (!text_part.empty() && std::isspace(static_cast<unsigned char>(text_part.back()))) text_part.pop_back();

    language_out = lang;
    text_out = text_part;
}

} // namespace aila_asr

// ============================================================
// Inference Engine: orchestrates loading, tokenization, inference
// ============================================================
class InferenceEngine {
public:
    using ChatStreamCallback = std::function<bool(const aila::chat::StructuredStreamEvent&)>;

    InferenceEngine() = default;

    // Initialize: load model + tokenizer from model directory
    bool init(const std::string& model_dir, int max_seq_len = 4096,
              const std::string& lora_dir = "") {
        clear_error();
        model_dir_ = model_dir;
        lora_dir_ = lora_dir;

        AILA_LOG_INFO("========================================");
        AILA_LOG_INFO("  Aila Inference Engine");
        AILA_LOG_INFO("========================================");

        // 1. Create context
        AILA_LOG_INFO("[1/3] Initializing GPU context...");
        ctx_ = std::make_unique<Context>();

        // Detection models are self-contained and intentionally do not initialize
        // tokenizer, generation backend, KV cache, or token warmup state.
        {
            ModelSpec detected_spec;
            std::string detected_error;
            const bool detected = aila::modelspec::load_from_dir(
                model_dir, detected_spec, &detected_error);
            if (!detected && detected_spec.model_type == "yolo26") {
                set_error(EngineErrorCode::InvalidArgument, detected_error);
                AILA_LOG_ERROR("[YOLO26] Invalid model package: %s", detected_error.c_str());
                return false;
            }
            if (detected && detected_spec.family == ModelFamily::Yolo26) {
                if (!lora_dir.empty()) {
                    set_error(EngineErrorCode::ModelCapability,
                              "YOLO26 detection models do not support LoRA adapters");
                    return false;
                }
                model_spec_ = std::move(detected_spec);
                try {
                    AILA_LOG_INFO("[2/3] Loading YOLO26 %s FP16 weights...",
                                  model_spec_.yolo26.scale.c_str());
                    weights_ = std::make_unique<ModelWeights>(
                        LoadModelWeightsFromDir(model_dir, *ctx_));
                    detector_ = std::make_unique<aila::vision::Yolo26Detector>();
                    std::string detector_error;
                    if (!detector_->init(*ctx_, *weights_, model_spec_.yolo26, &detector_error)) {
                        set_error(EngineErrorCode::RuntimeError, detector_error);
                        AILA_LOG_ERROR("[YOLO26] Detector initialization failed: %s", detector_error.c_str());
                        detector_.reset();
                        return false;
                    }
                } catch (const std::exception& error) {
                    set_error(EngineErrorCode::RuntimeError, error.what());
                    AILA_LOG_ERROR("[YOLO26] Detector initialization failed: %s", error.what());
                    detector_.reset();
                    return false;
                }
                AILA_LOG_INFO("[3/3] YOLO26 detector ready (input=%dx%d classes=%d)",
                              model_spec_.yolo26.input_width,
                              model_spec_.yolo26.input_height,
                              model_spec_.yolo26.num_classes);
                return true;
            }
        }

        // 2. Load tokenizer
        AILA_LOG_INFO("[2/4] Loading tokenizer...");
        if (!tokenizer_.load(model_dir)) {
            AILA_LOG_ERROR("Failed to load tokenizer");
            return false;
        }

        auto validate_quantization = [&](const ModelSpec& spec, std::string* error_message) -> bool {
            if (!spec.is_quantized()) {
                return true;
            }
            const auto& quant = spec.quantization;
            if (!spec.is_bitsandbytes_4bit()) {
                if (error_message) {
                    *error_message = "Only bitsandbytes 4-bit checkpoints are supported in the quantized path";
                }
                return false;
            }
            if (quant.bnb_4bit_quant_type != "nf4") {
                if (error_message) {
                    *error_message = "Only bitsandbytes NF4 checkpoints are supported";
                }
                return false;
            }
            if (quant.bnb_4bit_quant_storage != "uint8") {
                if (error_message) {
                    *error_message = "Only bitsandbytes uint8 packed storage is supported";
                }
                return false;
            }
            if (quant.bnb_4bit_compute_dtype != "float16") {
                if (error_message) {
                    *error_message = "On XPU, quantized bitsandbytes checkpoints must use bnb_4bit_compute_dtype=float16";
                }
                return false;
            }
            if (spec.family == ModelFamily::Qwen3Dense || spec.family == ModelFamily::Qwen3ASR ||
                spec.family == ModelFamily::Qwen3ForceAligner) {
                return true;
            }
            if (spec.family == ModelFamily::Qwen35Hybrid) {
                if (!is_supported_qwen35_hybrid_text_spec(spec.qwen35_text)) {
                    if (error_message) {
                        *error_message = "Qwen3.5 bitsandbytes v1 currently supports only the exact supported hybrid specs";
                    }
                    return false;
                }
                if (!aila::env::read_flag("AILA_Q35_LINEAR_DELTA", true)) {
                    if (error_message) {
                        *error_message = "Qwen3.5 bitsandbytes v1 requires AILA_Q35_LINEAR_DELTA=1";
                    }
                    return false;
                }
                return true;
            }
            if (error_message) {
                *error_message = "bitsandbytes quantized inference is currently supported only for Qwen3 dense and Qwen3.5 hybrid text models";
            }
            return false;
        };

        // 3. Load unified model spec
        AILA_LOG_INFO("[3/4] Loading model metadata...");
        {
            std::string spec_error;
            if (!aila::modelspec::load_from_dir(model_dir, model_spec_, &spec_error)) {
                AILA_LOG_WARN("[ModelSpec] %s (fallback to legacy qwen3 defaults)", spec_error.c_str());
                model_spec_.family = ModelFamily::Qwen3Dense;
                model_spec_.qwen3 = config_;
            }

            if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
                AILA_LOG_INFO("[ModelSpec] model_type=%s family=qwen3_5_hybrid text_hidden=%d layers=%d vocab=%d",
                              model_spec_.model_type.c_str(),
                              model_spec_.qwen35_text.hidden_size,
                              model_spec_.qwen35_text.num_hidden_layers,
                              model_spec_.qwen35_text.vocab_size);
                config_.hidden_size = model_spec_.qwen35_text.hidden_size;
                config_.num_hidden_layers = model_spec_.qwen35_text.num_hidden_layers;
                config_.num_attention_heads = model_spec_.qwen35_text.num_attention_heads;
                config_.num_key_value_heads = model_spec_.qwen35_text.num_key_value_heads;
                config_.head_dim = model_spec_.qwen35_text.head_dim;
                config_.intermediate_size = model_spec_.qwen35_text.intermediate_size;
                config_.vocab_size = model_spec_.qwen35_text.vocab_size;
                config_.max_position_embeddings = model_spec_.qwen35_text.max_position_embeddings;
                config_.rope_theta = model_spec_.qwen35_text.rope.rope_theta;
                config_.rms_norm_eps = model_spec_.qwen35_text.rms_norm_eps;
                config_.tie_word_embeddings = model_spec_.qwen35_text.tie_word_embeddings;
                config_.eos_token_id = model_spec_.qwen35_text.eos_token_id;
                if (model_spec_.vision.enabled) {
                    AILA_LOG_INFO("[ModelSpec] vision encoder found");
                }

                if (!model_spec_.vision.enabled && system_prompt_ == "You are a helpful assistant.") {
                    system_prompt_ =
                        "You are a helpful text assistant. "
                        "Reply in the user's language when possible. "
                        "Do not assume images or videos unless explicitly provided.";
                    AILA_LOG_INFO("[Config] Applied Qwen3.5 text-only default system prompt");
                }
            } else if (model_spec_.family == ModelFamily::Qwen3ASR) {
                config_ = model_spec_.qwen3;
                AILA_LOG_INFO("[ModelSpec] model_type=qwen3_asr text_hidden=%d layers=%d vocab=%d audio_enc_layers=%d audio_d_model=%d",
                              config_.hidden_size, config_.num_hidden_layers, config_.vocab_size,
                              model_spec_.audio.encoder_layers, model_spec_.audio.d_model);
            } else if (model_spec_.family == ModelFamily::Qwen3ForceAligner) {
                config_ = model_spec_.qwen3;
                AILA_LOG_INFO("[ModelSpec] model_type=qwen3_asr subtype=forced_aligner classify_num=%d "
                              "text_hidden=%d layers=%d audio_enc_layers=%d audio_d_model=%d",
                              model_spec_.classify_num,
                              config_.hidden_size, config_.num_hidden_layers,
                              model_spec_.audio.encoder_layers, model_spec_.audio.d_model);
            } else {
                config_ = model_spec_.qwen3;
                AILA_LOG_INFO("[ModelSpec] model_type=%s family=qwen3_dense hidden=%d layers=%d vocab=%d",
                              model_spec_.model_type.empty() ? "qwen3" : model_spec_.model_type.c_str(),
                              config_.hidden_size, config_.num_hidden_layers, config_.vocab_size);
            }

            if (model_spec_.is_quantized()) {
                AILA_LOG_INFO("[ModelSpec] quantization method=%s 4bit=%s type=%s compute_dtype=%s storage=%s double_quant=%s",
                              model_spec_.quantization.quant_method.c_str(),
                              model_spec_.quantization.load_in_4bit ? "true" : "false",
                              model_spec_.quantization.bnb_4bit_quant_type.c_str(),
                              model_spec_.quantization.bnb_4bit_compute_dtype.c_str(),
                              model_spec_.quantization.bnb_4bit_quant_storage.c_str(),
                              model_spec_.quantization.bnb_4bit_use_double_quant ? "true" : "false");
                std::string quant_error;
                if (!validate_quantization(model_spec_, &quant_error)) {
                    AILA_LOG_ERROR("[ModelSpec] %s", quant_error.c_str());
                    return false;
                }
            }
        }

        if (max_seq_len > config_.max_position_embeddings) {
            AILA_LOG_WARN("[Config] max_seq_len=%d exceeds model max_position_embeddings=%d, clamping",
                          max_seq_len, config_.max_position_embeddings);
            max_seq_len = config_.max_position_embeddings;
        }

        // 4. Load model weights
        AILA_LOG_INFO("[4/4] Loading model weights...");
        weights_ = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(model_dir, *ctx_));

        // 4.5. Load LoRA adapter (if provided)
        aila::lora::LoraAdapter lora_adapter;
        bool has_nf4_lora = false;
        if (!lora_dir_.empty()) {
            AILA_LOG_INFO("[LoRA] Loading adapter from: %s", lora_dir_.c_str());
            std::string lora_error;
            if (!aila::lora::LoraLoader::load(lora_dir_, lora_adapter, &lora_error)) {
                AILA_LOG_ERROR("[LoRA] Failed to load adapter: %s", lora_error.c_str());
                return false;
            }
            AILA_LOG_INFO("[LoRA] Adapter loaded: r=%d alpha=%d scaling=%.2f pairs=%zu",
                          lora_adapter.config.r, lora_adapter.config.lora_alpha,
                          lora_adapter.config.scaling, lora_adapter.pairs.size());

            // Dense models: merge LoRA delta into base weights before backend sees them.
            // NF4 models: defer to backend (weights are packed uint8, cannot merge in-place).
            if (!model_spec_.is_bitsandbytes_4bit()) {
                std::string merge_error;
                int merged = aila::lora::LoraLoader::merge_into_weights(*ctx_, lora_adapter, *weights_, &merge_error);
                if (merged < 0) {
                    AILA_LOG_ERROR("[LoRA] Merge failed: %s", merge_error.c_str());
                    return false;
                }
                AILA_LOG_INFO("[LoRA] Merged %d weight matrices (dense)", merged);
            } else {
                has_nf4_lora = true;
            }
        }

        // 5. Initialize backend
        if (model_spec_.family == ModelFamily::Qwen3TTS) {
            backend_ = std::make_unique<Qwen3TTSBackend>();
        } else if (model_spec_.family == ModelFamily::Qwen3ASR) {
            if (model_spec_.is_bitsandbytes_4bit()) {
                backend_ = std::make_unique<Qwen3ASRBnb4Backend>();
            } else {
                backend_ = std::make_unique<Qwen3ASRBackend>();
            }
        } else if (model_spec_.family == ModelFamily::Qwen3ForceAligner) {
            if (model_spec_.is_bitsandbytes_4bit()) {
                backend_ = std::make_unique<Qwen3ForceAlignerBnb4Backend>();
            } else {
                backend_ = std::make_unique<Qwen3ForceAlignerBackend>();
            }
        } else if (model_spec_.is_bitsandbytes_4bit()) {
            if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
                backend_ = std::make_unique<Qwen35HybridBnb4Backend>();
            } else {
                backend_ = std::make_unique<Qwen3Bnb4Backend>();
            }
        } else if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
            backend_ = std::make_unique<Qwen35HybridTextBackend>();
        } else {
            backend_ = std::make_unique<Qwen3DenseBackend>();
        }

        std::string backend_error;
        if (!backend_->load(*ctx_, *weights_, model_spec_, max_seq_len, &backend_error)) {
            AILA_LOG_ERROR("Failed to initialize model backend: %s", backend_error.c_str());
            return false;
        }

        if (model_spec_.family == ModelFamily::Qwen3TTS) {
            auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
            if (tts_backend) {
                std::string mimi_error;
                if (!tts_backend->load_mimi_vocoder(*ctx_, model_dir_, &mimi_error)) {
                    AILA_LOG_ERROR("Failed to load Mimi Vocoder for Qwen3TTS: %s", mimi_error.c_str());
                    return false;
                }
            }
            std::string tokenizer_path = model_dir_ + "/speech_tokenizer/model.safetensors";
            if (std::filesystem::exists(tokenizer_path)) {
                mimi_encoder_ = std::make_unique<aila::audio::MimiEncoder>();
                std::string enc_error;
                if (mimi_encoder_->loadWeights(tokenizer_path, &enc_error)) {
                    AILA_LOG_INFO("[TTS] Mimi 12Hz Audio Encoder loaded successfully");
                } else {
                    AILA_LOG_WARN("[TTS] Failed to load Mimi Audio Encoder: %s", enc_error.c_str());
                    mimi_encoder_.reset();
                }
            }
        }

        // Apply LoRA to NF4 backend (weights are packed uint8, must be done at runtime)
        if (has_nf4_lora) {
            std::string lora_apply_error;
            if (!backend_->apply_lora(*ctx_, lora_adapter, &lora_apply_error)) {
                AILA_LOG_ERROR("[LoRA] NF4 apply_lora failed: %s", lora_apply_error.c_str());
                return false;
            }
            AILA_LOG_INFO("[LoRA] NF4 runtime LoRA applied to backend");
        }

        audio_encoder_.reset();
        if (model_spec_.family == ModelFamily::Qwen3ASR ||
            model_spec_.family == ModelFamily::Qwen3ForceAligner) {
            audio_encoder_ = std::make_unique<aila::audio::Qwen3ASRAudioEncoder>();
            std::string audio_error;
            if (audio_encoder_->load(*ctx_, *weights_, model_spec_.audio, &audio_error)) {
                AILA_LOG_INFO("[Audio] Qwen3-ASR audio encoder loaded (output_dim=%d)",
                              audio_encoder_->output_dim());
            } else {
                AILA_LOG_ERROR("[Audio] Audio encoder init failed: %s", audio_error.c_str());
                return false;
            }
        }

        vision_backend_enabled_ = false;
        vision_encoder_.reset();
        if (model_spec_.family == ModelFamily::Qwen35Hybrid && model_spec_.vision.enabled) {
            vision_encoder_ = std::make_unique<aila::vision::Qwen35VisionEncoder>();
            std::string vision_error;
            if (vision_encoder_->load(*ctx_, *weights_, model_spec_, model_dir_, &vision_error)) {
                vision_backend_enabled_ = true;
                AILA_LOG_INFO("[Vision] Qwen3.5 vision encoder loaded (out_hidden=%d)",
                              vision_encoder_->out_hidden_size());
            } else {
                AILA_LOG_WARN("[Vision] Vision encoder init failed, falling back to text-only: %s",
                              vision_error.c_str());
                vision_encoder_.reset();
                vision_backend_enabled_ = false;
            }
        }

        auto to_mb = [](size_t bytes) -> double {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };
        AILA_LOG_INFO("[Memory] After model load: current=%.2f MB, peak=%.2f MB",
                      to_mb(ctx_->current_allocated_bytes()),
                      to_mb(ctx_->peak_allocated_bytes()));

        // 5. Warmup to amortize first-run JIT/primitive costs
        bool is_exact_q35_0p8b_spec =
            (model_spec_.family == ModelFamily::Qwen35Hybrid) &&
            is_exact_qwen35_hybrid_0p8b_spec(model_spec_.qwen35_text);
        int init_warmup_mode = aila::env::read_int_raw("AILA_INIT_WARMUP", -1);
        bool run_init_warmup = true;
        if (init_warmup_mode == 0) {
            run_init_warmup = false;
        } else if (init_warmup_mode == 1) {
            run_init_warmup = true;
        } else if (model_spec_.family == ModelFamily::Qwen35Hybrid && !is_supported_qwen35_hybrid_text_spec(model_spec_.qwen35_text)) {
            run_init_warmup = false;
        } else if (model_spec_.family == ModelFamily::Qwen3TTS) {
            run_init_warmup = false;
        }

        if (run_init_warmup) {
            if (init_warmup_mode == 1) {
                AILA_LOG_INFO("[Warmup] Running init warmup (forced via AILA_INIT_WARMUP=1)");
            } else if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
                AILA_LOG_INFO("[Warmup] Running init warmup for Qwen3.5 hybrid spec (hidden=%d layers=%d)",
                              model_spec_.qwen35_text.hidden_size,
                              model_spec_.qwen35_text.num_hidden_layers);
            } else {
                AILA_LOG_INFO("[Warmup] Running init warmup");
            }

            backend_->reset();

            // Lightweight warmup: 2-token prefill (like llama.cpp's approach).
            // A pure decode (seq_len=1) would skip prefill-score buffer allocation;
            // the first real prefill would then allocate a large buffer directly
            // and occasionally trigger GPU-level crashes on Arc A770.  Two tokens
            // trigger the prefill path with a 2×2 score buffer (~tens of bytes),
            // exercising the same allocation path as a real prefill at minimal cost.
            int warmup_token_id = (config_.bos_token_id >= 0) ? config_.bos_token_id : 0;
            int warmup_arr[2] = {warmup_token_id, warmup_token_id};
            int* warmup_token_ids = static_cast<int*>(ctx_->alloc_device(2 * sizeof(int)));
            ctx_->memcpy_h2d_async(warmup_token_ids, warmup_arr, 2 * sizeof(int));

            auto t_warmup_start = std::chrono::high_resolution_clock::now();
            Tensor& warmup_logits = backend_->forward(*ctx_, warmup_token_ids, 2);
            int* warmup_argmax = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            ops::argmax(*ctx_, warmup_logits, config_.vocab_size, warmup_argmax);
            ctx_->synchronize();
            auto t_warmup_end = std::chrono::high_resolution_clock::now();

            ctx_->free_device(warmup_argmax);
            ctx_->free_device(warmup_token_ids);
            backend_->reset();

            double warmup_ms = std::chrono::duration<double, std::milli>(t_warmup_end - t_warmup_start).count();
            AILA_LOG_INFO("[Warmup] Completed in %.2f ms", warmup_ms);
            AILA_LOG_INFO("[Memory] After warmup: current=%.2f MB, peak=%.2f MB",
                          to_mb(ctx_->current_allocated_bytes()),
                          to_mb(ctx_->peak_allocated_bytes()));

            if (vision_backend_enabled_ && vision_encoder_) {
                auto t_vision_warmup_start = std::chrono::high_resolution_clock::now();
                std::string vision_warmup_error;
                if (vision_encoder_->warmup(&vision_warmup_error)) {
                    auto t_vision_warmup_end = std::chrono::high_resolution_clock::now();
                    double vision_warmup_ms = std::chrono::duration<double, std::milli>(
                        t_vision_warmup_end - t_vision_warmup_start).count();
                    AILA_LOG_INFO("[Warmup] Vision warmup completed in %.2f ms", vision_warmup_ms);
                } else {
                    AILA_LOG_WARN("[Warmup] Vision warmup failed: %s", vision_warmup_error.c_str());
                }
            }
        } else {
            backend_->reset();
            if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
                if (init_warmup_mode == 0) {
                    AILA_LOG_INFO("[Warmup] Skipping init warmup (AILA_INIT_WARMUP=0, hidden=%d layers=%d attn_heads=%d kv_heads=%d)",
                                  model_spec_.qwen35_text.hidden_size,
                                  model_spec_.qwen35_text.num_hidden_layers,
                                  model_spec_.qwen35_text.num_attention_heads,
                                  model_spec_.qwen35_text.num_key_value_heads);
                } else {
                    AILA_LOG_INFO("[Warmup] Skipping init warmup for unsupported Qwen3.5 hybrid spec (hidden=%d layers=%d attn_heads=%d kv_heads=%d). Set AILA_INIT_WARMUP=1 to force.",
                                  model_spec_.qwen35_text.hidden_size,
                                  model_spec_.qwen35_text.num_hidden_layers,
                                  model_spec_.qwen35_text.num_attention_heads,
                                  model_spec_.qwen35_text.num_key_value_heads);
                }
            } else {
                AILA_LOG_INFO("[Warmup] Skipping init warmup (AILA_INIT_WARMUP=0)");
            }
            AILA_LOG_INFO("[Memory] Warmup skipped: current=%.2f MB, peak=%.2f MB",
                          to_mb(ctx_->current_allocated_bytes()),
                          to_mb(ctx_->peak_allocated_bytes()));
        }

        AILA_LOG_INFO("========================================");
        AILA_LOG_INFO("  Engine ready!");
        AILA_LOG_INFO("========================================");

        return true;
    }

    // ============================================================
    // Context management
    // ============================================================

    void set_system_prompt(const std::string& prompt) { system_prompt_ = prompt; }
    const std::string& system_prompt() const { return system_prompt_; }

    void reset_context() {
        history_.clear();
        mm_history_.clear();
        chat_session_.clear();
        cached_ids_.clear();
        benchmark_seed_ready_ = false;
        if (backend_) backend_->reset();
        AILA_LOG_INFO("[Context] Conversation reset");
    }

    int context_length() const { return static_cast<int>(cached_ids_.size()); }
    int max_context_length() const { return backend_ ? backend_->max_seq_len() : 0; }
    const ChatHistory& history() const { return history_; }
    size_t active_history_message_count() const {
        if (chat_session_.message_count_without_system() > 0) {
            return chat_session_.message_count_without_system();
        }
        if (!mm_history_.empty()) {
            size_t count = 0;
            for (const auto& msg : mm_history_) {
                if (msg.role != "system") ++count;
            }
            return count;
        }
        return history_.size();
    }
    size_t active_history_turn_count() const {
        size_t count = 0;
        if (chat_session_.turn_count() > 0) {
            return chat_session_.turn_count();
        }
        if (!mm_history_.empty()) {
            for (const auto& msg : mm_history_) {
                if (msg.role == "user") ++count;
            }
            return count;
        }
        for (const auto& msg : history_.messages()) {
            if (msg.role == "user") ++count;
        }
        return count;
    }
    int conversation_context_length() const {
        auto render_chat_request_len = [&](const aila::chat::ChatRequest& request) -> int {
            aila::chat::ChatFormatTextResult rendered;
            std::string error;
            if (!chat_formatter_.render_text(make_chat_format_input(request), false, rendered, &error)) {
                return -1;
            }
            return static_cast<int>(tokenizer_.encode(rendered.text).size());
        };

        if (chat_session_.message_count_without_system() > 0) {
            aila::chat::ChatRequest request = chat_session_.to_request_without_reasoning();
            int len = render_chat_request_len(request);
            if (len >= 0) {
                return len;
            }
            return context_length();
        }
        if (!mm_history_.empty()) {
            aila::chat::ChatRequest request =
                make_chat_request_from_legacy(mm_history_, GenerationConfig{});
            int len = render_chat_request_len(request);
            if (len >= 0) {
                return len;
            }
            return context_length();
        }
        if (!history_.empty()) {
            return static_cast<int>(
                tokenizer_.apply_chat_template(system_prompt_, history_).size());
        }
        return 0;
    }

    void align_backend_context_to_sequence(const std::vector<int>& prompt_ids,
                                           const std::vector<int>& generated_ids,
                                           int prompt_len,
                                           const char* log_tag) {
        if (!backend_ || !ctx_) return;
        int accepted_len = prompt_len + static_cast<int>(generated_ids.size());
        int actual_len = backend_->get_current_context_len();
        if (actual_len == accepted_len) return;

        auto replay_range = [&](const std::vector<int>& ids, int start, int len) {
            if (len <= 0) return;
            Tensor* ignored_logits = forward_prompt_tokens(ids, start, len);
            (void)ignored_logits;
        };

        if (actual_len > accepted_len) {
            AILA_LOG_INFO("%s KV decode overrun: backend_len=%d accepted_len=%d; replaying accepted tail",
                          log_tag, actual_len, accepted_len);
            if (!backend_->truncate_kv_cache(prompt_len)) {
                backend_->reset();
                actual_len = 0;
            } else {
                actual_len = backend_->get_current_context_len();
            }
        }

        if (actual_len < prompt_len) {
            replay_range(prompt_ids, actual_len, prompt_len - actual_len);
            actual_len = backend_->get_current_context_len();
        } else if (actual_len > prompt_len) {
            if (!backend_->truncate_kv_cache(prompt_len)) {
                backend_->reset();
                replay_range(prompt_ids, 0, prompt_len);
            }
            actual_len = backend_->get_current_context_len();
        }

        int generated_start = std::max(0, actual_len - prompt_len);
        if (generated_start < static_cast<int>(generated_ids.size())) {
            replay_range(generated_ids,
                         generated_start,
                         static_cast<int>(generated_ids.size()) - generated_start);
        }
        ctx_->synchronize();

        int final_len = backend_->get_current_context_len();
        if (final_len != accepted_len) {
            AILA_LOG_WARN("%s KV context alignment ended at %d, expected %d",
                          log_tag, final_len, accepted_len);
        }
    }

    Tensor* forward_prompt_tokens(const std::vector<int>& token_ids,
                                  int start,
                                  int len,
                                  int snapshot_after_len = 0,
                                  bool async_h2d = false) {
        if (!backend_ || !ctx_ || len <= 0) return nullptr;

        int chunk_size = len;
        if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
            int configured = aila::env::read_int("AILA_Q35_PREFILL_CHUNK", 512);
            if (configured > 0) {
                chunk_size = std::max(1, std::min(configured, len));
            }
        }

        Tensor* logits_ptr = nullptr;
        const int end = start + len;
        int pos = start;
        while (pos < end) {
            int next = std::min(end, pos + chunk_size);
            if (snapshot_after_len > pos && snapshot_after_len < next) {
                next = snapshot_after_len;
            }
            const int slice_len = next - pos;
            int* token_ids_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(slice_len) * sizeof(int)));
            if (async_h2d) {
                ctx_->memcpy_h2d_async(token_ids_device,
                                       token_ids.data() + pos,
                                       static_cast<size_t>(slice_len) * sizeof(int));
            } else {
                ctx_->memcpy_h2d(token_ids_device,
                                 token_ids.data() + pos,
                                 static_cast<size_t>(slice_len) * sizeof(int));
            }
            logits_ptr = &backend_->forward(*ctx_, token_ids_device, slice_len);
            ctx_->synchronize();
            ctx_->free_device(token_ids_device);

            pos = next;
        }
        return logits_ptr;
    }

    std::string decode_with_special_tokens(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) {
            std::string raw = tokenizer_.raw_token(id);
            if (!raw.empty() && tokenizer_.special_token_id(raw) == id) {
                out += raw;
            } else {
                out += tokenizer_.decode(id);
            }
        }
        return out;
    }

    void debug_dump_prompt_text(const char* label,
                                const std::string& prompt_text,
                                int prompt_tokens) const {
        if (!aila::env::read_flag("AILA_DEBUG_PROMPT_TEXT", false)) return;
        int max_chars = aila::env::read_int("AILA_DEBUG_PROMPT_TEXT_MAX_CHARS", 20000);
        std::string shown = prompt_text;
        bool truncated = false;
        if (max_chars > 0 && static_cast<int>(shown.size()) > max_chars) {
            shown.resize(static_cast<size_t>(max_chars));
            truncated = true;
        }
        AILA_LOG_INFO("[DebugPromptText] %s tokens=%d chars=%zu%s\n----- BEGIN RENDERED PROMPT -----\n%s\n----- END RENDERED PROMPT -----",
                      label,
                      prompt_tokens,
                      prompt_text.size(),
                      truncated ? " (truncated)" : "",
                      shown.c_str());
    }

    aila::chat::ChatRequest make_chat_request_from_legacy(
        const std::vector<Message>& messages,
        const GenerationConfig& config,
        const aila::chat::ChatRequest* source_request = nullptr) const {
        aila::chat::ChatRequest request;
        request.generation_config = config;
        request.messages.reserve(messages.size());
        for (size_t i = 0; i < messages.size(); ++i) {
            const auto& legacy_msg = messages[i];
            aila::chat::ChatMessage msg;
            msg.role = aila::chat::role_from_string(legacy_msg.role);
            for (const auto& part : legacy_msg.content) {
                aila::chat::ChatContentPart out;
                out.type = part.type;
                out.text = part.text;
                out.uri = part.uri;
                out.binary_data = part.binary_data;
                out.media_format = part.media_format;
                msg.content.push_back(std::move(out));
            }
            if (source_request && i < source_request->messages.size()) {
                const auto& source_msg = source_request->messages[i];
                msg.reasoning_content = source_msg.reasoning_content;
                msg.tool_calls = source_msg.tool_calls;
                msg.name = source_msg.name;
                msg.tool_call_id = source_msg.tool_call_id;
            }
            request.messages.push_back(std::move(msg));
        }
        if (source_request) {
            request.tools = source_request->tools;
            request.tool_choice = source_request->tool_choice;
            request.tool_choice_function_name = source_request->tool_choice_function_name;
            request.template_options = source_request->template_options;
        }
        return request;
    }

    aila::chat::ChatFormatInput make_chat_format_input(const aila::chat::ChatRequest& request) const {
        aila::chat::ChatFormatInput input;
        input.request = &request;
        input.family = model_spec_.family;
        input.is_qwen35_0p8b =
            (model_spec_.family == ModelFamily::Qwen35Hybrid) &&
            is_exact_qwen35_hybrid_0p8b_spec(model_spec_.qwen35_text);
        input.tokenizer_chat_template = tokenizer_.chat_template();
        input.bos_token = tokenizer_.bos_token();
        input.eos_token = tokenizer_.eos_token();
        return input;
    }

    std::string current_model_family_name() const {
        switch (model_spec_.family) {
        case ModelFamily::Yolo26: return "yolo26";
        case ModelFamily::Qwen35Hybrid: return "qwen3_5_hybrid";
        case ModelFamily::Qwen3Dense: return "qwen3";
        default: return "unknown";
        }
    }

    std::string tool_choice_name(const aila::chat::ChatRequest& request) const {
        switch (request.tool_choice) {
        case aila::chat::ToolChoice::None: return "none";
        case aila::chat::ToolChoice::Required: return "required";
        case aila::chat::ToolChoice::Function: return "function:" + request.tool_choice_function_name;
        case aila::chat::ToolChoice::Auto: return "auto";
        }
        return "auto";
    }

    std::string tool_policy_name(aila::chat::ToolPolicyMode policy) const {
        return policy == aila::chat::ToolPolicyMode::Strict ? "strict" : "warn";
    }

    std::vector<int> thinking_close_token_ids() const {
        std::vector<int> ids;
        int end_think = tokenizer_.special_token_id("</think>");
        if (end_think >= 0) {
            ids.push_back(end_think);
        } else {
            auto fallback = tokenizer_.encode("</think>");
            ids.insert(ids.end(), fallback.begin(), fallback.end());
        }
        auto suffix = tokenizer_.encode("\n\n");
        ids.insert(ids.end(), suffix.begin(), suffix.end());
        return ids;
    }

    // ============================================================
    // Generate response (multi-turn with incremental prefill)
    //
    // Context tracking: we maintain cached_ids_ which is the EXACT
    // token ID sequence currently stored in the KV cache. This avoids
    // decode-then-re-encode mismatches. New turns are built by
    // appending raw token IDs directly, never by re-encoding text.
    // ============================================================
    std::string generate(const std::string& user_message,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        if (model_spec_.family == ModelFamily::Yolo26) {
            set_error(EngineErrorCode::ModelCapability,
                      "Text generation is not supported by a YOLO26 detection model");
            return "";
        }
        if (!backend_) {
            set_error(EngineErrorCode::RuntimeError, "Backend is not initialized");
            return "";
        }

        chat_session_.set_system_prompt(system_prompt_);
        chat_session_.add_user_text(user_message);

        auto make_session_request = [&]() {
            aila::chat::ChatRequest request = chat_session_.to_request_without_reasoning();
            request.generation_config = gen_config;
            return request;
        };

        aila::chat::ChatRequest request = make_session_request();
        std::string out = generate_chat_request_text(request, token_callback);
        while (last_error_code_ == EngineErrorCode::ContextOverflow) {
            if (!chat_session_.drop_oldest_turn()) {
                break;
            }
            AILA_LOG_WARN("[Generate] Chat history truncated to fit context window (%zu messages remaining)",
                          chat_session_.message_count_without_system());
            request = make_session_request();
            out = generate_chat_request_text(request, token_callback);
        }

        if (last_error_code_ != EngineErrorCode::Ok) {
            chat_session_.remove_last_user_message();
            return "";
        }

        aila::chat::AssistantChatResult result = aila::chat::parse_assistant_output(out);
        chat_session_.add_assistant_result(result, false);
        return out;

    }

    std::string generate_messages(const std::vector<Message>& messages,
                                  const GenerationConfig& gen_config = GenerationConfig(),
                                  std::function<void(const std::string&)> token_callback = nullptr,
                                  const aila::chat::ChatRequest* structured_request = nullptr) {
        clear_error();
        last_generation_finish_reason_ = "stop";
        last_generation_forced_think_close_ = false;
        last_generation_think_close_truncated_ = false;
        last_generation_template_name_.clear();
        if (reject_yolo_capability("Text generation")) return "";
        if (!backend_) {
            set_error(EngineErrorCode::RuntimeError, "Backend is not initialized");
            return "";
        }
        if (backend_->supports_vision_embedding_override()) {
            backend_->clear_embedding_overrides();
            backend_->clear_mrope_positions();
        }

        GenerationConfig tuned_cfg = gen_config;
        if (model_spec_.family == ModelFamily::Qwen35Hybrid && tuned_cfg.do_sample) {
            bool tuned = false;
            if (tuned_cfg.repetition_penalty <= 1.0f) {
                tuned_cfg.repetition_penalty = 1.12f;
                tuned = true;
            }
            if (tuned_cfg.presence_penalty == 0.0f) {
                tuned_cfg.presence_penalty = 0.05f;
                tuned = true;
            }
            if (tuned_cfg.frequency_penalty == 0.0f) {
                tuned_cfg.frequency_penalty = 0.10f;
                tuned = true;
            }
            if (tuned_cfg.top_k < 30) {
                tuned_cfg.top_k = 40;
                tuned = true;
            }
            if (tuned_cfg.temperature < 0.75f) {
                tuned_cfg.temperature = 0.80f;
                tuned = true;
            }
            if (tuned) {
                AILA_LOG_INFO("[Qwen3.5] Applied anti-loop sampling defaults "
                              "(temp=%.2f top_k=%d rep=%.2f pres=%.2f freq=%.2f)",
                              tuned_cfg.temperature, tuned_cfg.top_k,
                              tuned_cfg.repetition_penalty,
                              tuned_cfg.presence_penalty,
                              tuned_cfg.frequency_penalty);
            }
        }

        std::vector<Message> render_messages;
        render_messages.reserve(messages.size());
        std::vector<sycl::ext::oneapi::bfloat16> vision_embeddings_flat;
        size_t total_vision_tokens = 0;
        std::vector<sycl::ext::oneapi::bfloat16> audio_embeddings_flat;
        size_t total_audio_tokens = 0;
        using bf16 = sycl::ext::oneapi::bfloat16;
        struct VisionSegment {
            int token_count = 0;
            int llm_grid_t = 1;
            int llm_grid_h = 0;
            int llm_grid_w = 0;
        };
        std::vector<VisionSegment> vision_segments;

        for (const auto& m : messages) {
            Message out_msg;
            out_msg.role = m.role;
            for (const auto& p : m.content) {
                if (p.type == ContentType::Text) {
                    out_msg.content.push_back(p);
                    continue;
                }
                if (p.type == ContentType::Video) {
                    set_error(EngineErrorCode::TemplateError,
                              "Video content is not enabled yet in this backend");
                    return "";
                }
                if (p.type == ContentType::Image) {
                    if (!vision_backend_enabled_ || !vision_encoder_ || !vision_encoder_->ready()) {
                        set_error(EngineErrorCode::VisionNotEnabled,
                                  "Vision content is not enabled for this backend");
                        return "";
                    }

                    aila::vision::VisionEncodeResult encoded;
                    std::string vision_err;
                    if (!vision_encoder_->encode_image(p, encoded, &vision_err)) {
                        set_error(EngineErrorCode::RuntimeError,
                                  "Vision encode failed: " + vision_err);
                        return "";
                    }
                    if (encoded.token_count <= 0) {
                        set_error(EngineErrorCode::RuntimeError,
                                  "Vision encode produced zero tokens");
                        return "";
                    }
                    if (encoded.embeddings.size() !=
                        static_cast<size_t>(encoded.token_count) * static_cast<size_t>(config_.hidden_size)) {
                        set_error(EngineErrorCode::RuntimeError,
                                  "Vision embedding size mismatch with language hidden size");
                        return "";
                    }
                    AILA_LOG_INFO("[Vision] Encoded image '%s' -> %d tokens",
                                  p.uri.c_str(), encoded.token_count);

                    ContentPart ph;
                    ph.type = ContentType::Text;
                    ph.text.reserve(static_cast<size_t>(encoded.token_count) * 12u + 32u);
                    ph.text += "<|vision_start|>";
                    for (int i = 0; i < encoded.token_count; ++i) {
                        ph.text += "<|image_pad|>";
                    }
                    ph.text += "<|vision_end|>";
                    out_msg.content.push_back(std::move(ph));

                    total_vision_tokens += static_cast<size_t>(encoded.token_count);
                    vision_segments.push_back(VisionSegment{
                        encoded.token_count,
                        encoded.llm_grid_t,
                        encoded.llm_grid_h,
                        encoded.llm_grid_w,
                    });
                    vision_embeddings_flat.insert(
                        vision_embeddings_flat.end(),
                        encoded.embeddings.begin(),
                        encoded.embeddings.end());
                    continue;
                }
                if (p.type == ContentType::Audio) {
                    if (!audio_encoder_) {
                        set_error(EngineErrorCode::RuntimeError,
                                  "Audio encoder is not initialized");
                        return "";
                    }

                    AudioBuffer audio;
                    std::string audio_err;
                    bool loaded = false;
                    if (!p.binary_data.empty()) {
                        loaded = load_audio_from_memory(p.binary_data.data(), p.binary_data.size(), p.media_format, audio, &audio_err);
                    } else {
                        loaded = load_audio(p.uri, audio, &audio_err);
                    }

                    if (!loaded) {
                        set_error(EngineErrorCode::RuntimeError,
                                  "Audio loading failed: " + audio_err);
                        return "";
                    }

                    std::vector<float> mono;
                    if (audio.channels > 1) {
                        mono.resize(audio.samples.size() / audio.channels);
                        for (size_t i = 0; i < mono.size(); ++i) {
                            float sum = 0.0f;
                            for (int c = 0; c < audio.channels; ++c)
                                sum += audio.samples[i * audio.channels + c];
                            mono[i] = sum / static_cast<float>(audio.channels);
                        }
                    } else {
                        mono = std::move(audio.samples);
                    }

                    std::vector<float> mono_16k;
                    resample_to_16k(mono, audio.sample_rate, mono_16k);

                    MelSpectrogram mel;
                    if (!compute_mel_spectrogram(mono_16k, mel, &audio_err)) {
                        set_error(EngineErrorCode::RuntimeError, "Spectrogram calculation failed: " + audio_err);
                        return "";
                    }

                    int mel_padded_frames = mel.n_frames;
                    int mel_actual_frames = mel.actual_frames;
                    int nM = mel.n_mels;
                    std::vector<bf16> mel_bf16(static_cast<size_t>(mel_padded_frames) * nM);
                    for (int f = 0; f < mel_padded_frames; ++f)
                        for (int m = 0; m < nM; ++m)
                            mel_bf16[m * mel_padded_frames + f] = bf16(mel.data[f * nM + m]);

                    Tensor mel_device = Tensor::allocate(*ctx_, {1, nM, mel_padded_frames});
                    ctx_->memcpy_h2d(mel_device.data(), mel_bf16.data(), mel_bf16.size() * sizeof(bf16));

                    int audio_len = 0;
                    int od = model_spec_.audio.output_dim;
                    int max_audio_len = ((mel_actual_frames + 99) / 100) * 13 + 32;
                    Tensor af_tmp = Tensor::allocate(*ctx_, {max_audio_len, od});
                    std::string enc_error;
                    if (!audio_encoder_->encode(*ctx_, mel_device, mel_actual_frames,
                                                 af_tmp, audio_len, &enc_error)) {
                        set_error(EngineErrorCode::RuntimeError, "Audio encoding failed: " + enc_error);
                        return "";
                    }

                    Tensor audio_features = Tensor::allocate(*ctx_, {audio_len, od});
                    bf16* af_dst = audio_features.data_as<bf16>();
                    bf16* af_src = af_tmp.data_as<bf16>();
                    ctx_->queue().memcpy(af_dst, af_src, static_cast<size_t>(audio_len) * od * sizeof(bf16));

                    std::vector<bf16> audio_bf16(static_cast<size_t>(audio_len) * od);
                    ctx_->memcpy_d2h(audio_bf16.data(), audio_features.data(), audio_bf16.size() * sizeof(bf16));
                    ctx_->synchronize();

                    ContentPart ph;
                    ph.type = ContentType::Text;
                    ph.text.reserve(static_cast<size_t>(audio_len) * 12u + 32u);
                    ph.text += "<|audio_start|>";
                    for (int i = 0; i < audio_len; ++i) {
                        ph.text += "<|audio_pad|>";
                    }
                    ph.text += "<|audio_end|>";
                    out_msg.content.push_back(std::move(ph));

                    total_audio_tokens += static_cast<size_t>(audio_len);
                    audio_embeddings_flat.insert(
                        audio_embeddings_flat.end(),
                        audio_bf16.begin(),
                        audio_bf16.end());
                    continue;
                }
                set_error(EngineErrorCode::TemplateError, "Unknown content part type");
                return "";
            }
            render_messages.push_back(std::move(out_msg));
        }

        aila::chat::ChatRequest chat_request =
            make_chat_request_from_legacy(render_messages, tuned_cfg, structured_request);
        aila::chat::ChatFormatTextResult rendered;
        std::string render_error;
        if (!chat_formatter_.render_text(make_chat_format_input(chat_request), true, rendered, &render_error)) {
            AILA_LOG_ERROR("[GenerateMessages] Template render failed: %s", render_error.c_str());
            set_error(EngineErrorCode::TemplateError, render_error);
            return "";
        }
        last_generation_template_name_ = rendered.template_name;
        if (aila::env::read_flag("AILA_DEBUG_CHAT_TEMPLATE", false) ||
            aila::env::read_flag("AILA_DEBUG_PROMPT_TEXT", false)) {
            AILA_LOG_INFO("[ChatTemplate] source=%s", rendered.template_name.c_str());
        }
        std::vector<int> full_ids = tokenizer_.encode(rendered.text);
        auto rendered_ends_with = [](const std::string& text, const std::string& suffix) -> bool {
            return text.size() >= suffix.size() &&
                   text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        bool restore_open_think_prefix =
            model_spec_.family == ModelFamily::Qwen35Hybrid &&
            (rendered_ends_with(rendered.text, "<think>\n") ||
             rendered_ends_with(rendered.text, "<think>\r\n") ||
             rendered_ends_with(rendered.text, "<think>"));
        int recurrent_prefill_anchor = 0;
        if (model_spec_.family == ModelFamily::Qwen35Hybrid &&
            total_vision_tokens == 0 && total_audio_tokens == 0) {
            std::string stable_err;
            aila::chat::ChatFormatTextResult stable_rendered;
            if (chat_formatter_.render_text(make_chat_format_input(chat_request), false, stable_rendered, &stable_err)) {
                std::vector<int> stable_ids = tokenizer_.encode(stable_rendered.text);
                if (!stable_ids.empty() && stable_ids.size() < full_ids.size() &&
                    std::equal(stable_ids.begin(), stable_ids.end(), full_ids.begin())) {
                    recurrent_prefill_anchor = static_cast<int>(stable_ids.size());
                    std::vector<int> assistant_header;
                    assistant_header.push_back(tokenizer_.im_start_id());
                    auto assistant_header_text = tokenizer_.encode("assistant\n");
                    assistant_header.insert(assistant_header.end(),
                                            assistant_header_text.begin(),
                                            assistant_header_text.end());
                    if (stable_ids.size() + assistant_header.size() < full_ids.size() &&
                        std::equal(assistant_header.begin(), assistant_header.end(),
                                   full_ids.begin() + static_cast<std::ptrdiff_t>(stable_ids.size()))) {
                        recurrent_prefill_anchor += static_cast<int>(assistant_header.size());
                    }
                }
            } else {
                AILA_LOG_WARN("[GenerateMessages] Stable-prefix render failed; skipping recurrent prefill anchor: %s",
                              stable_err.c_str());
            }
        }
        if (aila::env::read_flag("AILA_DEBUG_PROMPT_TEXT", false)) {
            debug_dump_prompt_text("[GenerateMessages]", rendered.text,
                                   static_cast<int>(full_ids.size()));
        }

        if (total_vision_tokens > 0) {
            if (!backend_->supports_vision_embedding_override()) {
                set_error(EngineErrorCode::RuntimeError,
                          "Vision embedding override requires a backend with multimodal injection support");
                return "";
            }

            int image_pad_id = model_spec_.vision.image_token_id;
            if (image_pad_id < 0) {
                image_pad_id = tokenizer_.special_token_id("<|image_pad|>");
            }
            if (image_pad_id < 0) {
                set_error(EngineErrorCode::RuntimeError,
                          "Tokenizer/model does not provide image token id");
                return "";
            }

            std::vector<int> image_positions;
            image_positions.reserve(total_vision_tokens);
            for (int i = 0; i < static_cast<int>(full_ids.size()); ++i) {
                if (full_ids[static_cast<size_t>(i)] == image_pad_id) {
                    image_positions.push_back(i);
                }
            }

            if (image_positions.size() != total_vision_tokens) {
                set_error(EngineErrorCode::RuntimeError,
                          "Mismatch between rendered image tokens and encoded vision tokens");
                return "";
            }

            backend_->set_embedding_overrides(image_positions, vision_embeddings_flat, config_.hidden_size);
            std::vector<int> pos_t(full_ids.size(), 0);
            std::vector<int> pos_h(full_ids.size(), 0);
            std::vector<int> pos_w(full_ids.size(), 0);
            size_t next_segment = 0;
            int current_pos = 0;
            for (size_t i = 0; i < full_ids.size();) {
                if (full_ids[i] != image_pad_id) {
                    size_t j = i + 1;
                    while (j < full_ids.size() && full_ids[j] != image_pad_id) ++j;
                    for (size_t k = i; k < j; ++k) {
                        int text_pos = current_pos + static_cast<int>(k - i);
                        pos_t[k] = text_pos;
                        pos_h[k] = text_pos;
                        pos_w[k] = text_pos;
                    }
                    current_pos += static_cast<int>(j - i);
                    i = j;
                    continue;
                }

                size_t j = i + 1;
                while (j < full_ids.size() && full_ids[j] == image_pad_id) ++j;
                if (next_segment >= vision_segments.size()) {
                    set_error(EngineErrorCode::RuntimeError,
                              "Missing multimodal segment metadata for image tokens");
                    return "";
                }
                const auto& seg = vision_segments[next_segment++];
                const int seq_len = static_cast<int>(j - i);
                const int grid_t = std::max(1, seg.llm_grid_t);
                const int grid_h = std::max(1, seg.llm_grid_h);
                const int grid_w = std::max(1, seg.llm_grid_w);
                if (seq_len != seg.token_count || grid_t * grid_h * grid_w != seq_len) {
                    set_error(EngineErrorCode::RuntimeError,
                              "Vision grid metadata does not match rendered image token count");
                    return "";
                }
                size_t out_idx = i;
                for (int t = 0; t < grid_t; ++t) {
                    for (int h = 0; h < grid_h; ++h) {
                        for (int w = 0; w < grid_w; ++w) {
                            pos_t[out_idx] = current_pos + t;
                            pos_h[out_idx] = current_pos + h;
                            pos_w[out_idx] = current_pos + w;
                            ++out_idx;
                        }
                    }
                }
                current_pos += std::max({grid_t, grid_h, grid_w});
                i = j;
            }
            int max_pos = 0;
            for (size_t i = 0; i < full_ids.size(); ++i) {
                max_pos = std::max(max_pos, std::max({pos_t[i], pos_h[i], pos_w[i]}));
            }
            const int text_pos_delta = max_pos + 1 - static_cast<int>(full_ids.size());
            backend_->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, text_pos_delta);
            AILA_LOG_INFO("[Vision] Injecting %zu image embeddings into prompt", total_vision_tokens);
            AILA_LOG_INFO("[Vision] Applied multimodal text RoPE positions (delta=%d)", text_pos_delta);
        }

        if (total_audio_tokens > 0) {
            int audio_pad_id = model_spec_.audio_token_id;
            if (audio_pad_id < 0) {
                audio_pad_id = tokenizer_.special_token_id("<|audio_pad|>");
            }
            if (audio_pad_id < 0) {
                set_error(EngineErrorCode::RuntimeError,
                          "Tokenizer/model does not provide audio token id");
                return "";
            }

            std::vector<int> audio_positions;
            audio_positions.reserve(total_audio_tokens);
            for (int i = 0; i < static_cast<int>(full_ids.size()); ++i) {
                if (full_ids[static_cast<size_t>(i)] == audio_pad_id) {
                    audio_positions.push_back(i);
                }
            }

            if (audio_positions.size() != total_audio_tokens) {
                set_error(EngineErrorCode::RuntimeError,
                          "Mismatch between rendered audio tokens and encoded audio tokens");
                return "";
            }

            backend_->set_embedding_overrides(audio_positions, audio_embeddings_flat, model_spec_.audio.output_dim);

            if (!aila::env::read_flag("AILA_NO_MROPE", false)) {
                int total_len = static_cast<int>(full_ids.size());
                std::vector<int> pos_t(total_len), pos_h(total_len), pos_w(total_len);
                for (int i = 0; i < total_len; ++i) {
                    pos_t[i] = i; pos_h[i] = i; pos_w[i] = i;
                }
                backend_->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, 0);
            }
            AILA_LOG_INFO("[Audio] Injecting %zu audio embeddings into prompt", total_audio_tokens);
        }

        int max_ctx = backend_->max_seq_len();
        int total_prompt_len = static_cast<int>(full_ids.size());
        bool debug_token_ids = aila::env::read_flag("AILA_DEBUG_TOKEN_IDS", false);
        if (debug_token_ids) {
            int show_n = std::min<int>(static_cast<int>(full_ids.size()), 64);
            AILA_LOG_INFO("[DebugToken] prompt_tokens=%d (showing %d)", total_prompt_len, show_n);
            for (int i = 0; i < show_n; ++i) {
                int tid = full_ids[(size_t)i];
                std::string piece = tokenizer_.decode(tid);
                AILA_LOG_INFO("[DebugToken] prompt[%d]=%d text='%s'", i, tid, piece.c_str());
            }
        }
        int available_decode_tokens = max_ctx - total_prompt_len;
        if (available_decode_tokens <= 0) {
            AILA_LOG_WARN("[GenerateMessages] Prompt exceeds context window (prompt=%d max_seq=%d)",
                          total_prompt_len, max_ctx);
            set_error(EngineErrorCode::ContextOverflow, "Prompt exceeds context window");
            return "";
        }
        int max_new_tokens = std::min(gen_config.max_new_tokens, available_decode_tokens);

        int reusable_prefix = 0;
        bool allow_incremental_prefill = (total_vision_tokens == 0 && total_audio_tokens == 0);
        if (allow_incremental_prefill) {
            int max_possible_match = std::min(static_cast<int>(cached_ids_.size()),
                                              static_cast<int>(full_ids.size()));
            while (reusable_prefix < max_possible_match &&
                   cached_ids_[reusable_prefix] == full_ids[reusable_prefix]) {
                reusable_prefix++;
            }
        }
        int requested_reusable_prefix = reusable_prefix;
        if (backend_) {
            bool trunc_ok = backend_->truncate_kv_cache(reusable_prefix);
            if (!trunc_ok) {
                // Backend could not partially truncate (e.g. DeltaNet
                // recurrent state requires a full rebuild).  Reset and
                // fall back to a full prefill of all prompt tokens.
                backend_->reset();
                cached_ids_.clear();
                reusable_prefix = 0;
            } else {
                reusable_prefix = backend_->get_current_context_len();
                if (requested_reusable_prefix > reusable_prefix) {
                    AILA_LOG_INFO("[GenerateMessages] Incremental prefill rollback: requested reuse=%d actual reuse=%d reprefill=%d",
                                  requested_reusable_prefix, reusable_prefix,
                                  requested_reusable_prefix - reusable_prefix);
                }
                cached_ids_.resize(reusable_prefix);
            }
        } else {
            cached_ids_.resize(reusable_prefix);
        }

        int prefill_start = reusable_prefix;
        int new_tokens_to_prefill = total_prompt_len - prefill_start;
        if (new_tokens_to_prefill <= 0) {
            if (backend_) backend_->reset();
            cached_ids_.clear();
            prefill_start = 0;
            new_tokens_to_prefill = total_prompt_len;
        }

        if (prefill_start > 0) {
            AILA_LOG_INFO("[GenerateMessages] Incremental prefill: reusing %d cached tokens, prefilling %d new tokens",
                          prefill_start, new_tokens_to_prefill);
        } else {
            AILA_LOG_INFO("[GenerateMessages] Full prefill: %d tokens", total_prompt_len);
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        Tensor* logits_ptr = nullptr;
        bool tokenwise_prefill = (model_spec_.family == ModelFamily::Qwen35Hybrid) &&
                                 aila::env::read_flag("AILA_Q35_PREFILL_TOKENWISE", false);
        if (!tokenwise_prefill) {
            int snapshot_after_len =
                (recurrent_prefill_anchor > prefill_start &&
                 recurrent_prefill_anchor < total_prompt_len)
                    ? recurrent_prefill_anchor
                    : 0;
            logits_ptr = forward_prompt_tokens(full_ids, prefill_start,
                                               new_tokens_to_prefill,
                                               snapshot_after_len);
        } else {
            AILA_LOG_INFO("[Qwen3.5] Tokenwise prefill enabled for debug");
            int* one_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            for (int i = 0; i < new_tokens_to_prefill; ++i) {
                int tok = full_ids[static_cast<size_t>(prefill_start + i)];
                ctx_->memcpy_h2d(one_token_device, &tok, sizeof(int));
                logits_ptr = &backend_->forward(*ctx_, one_token_device, 1);
            }
            ctx_->free_device(one_token_device);
        }
        Tensor& logits = *logits_ptr;

        if (aila::env::read_flag("AILA_DEBUG_Q35_LOGITS", false)) {
            int vocab = config_.vocab_size;
            std::vector<float> host_logits((size_t)vocab, 0.0f);
            if (logits.dtype() == dnnl::memory::data_type::f32) {
                ctx_->memcpy_d2h(host_logits.data(), logits.data(),
                                 host_logits.size() * sizeof(float));
            } else {
                using bf16 = sycl::ext::oneapi::bfloat16;
                std::vector<bf16> tmp((size_t)vocab);
                ctx_->memcpy_d2h(tmp.data(), logits.data(), tmp.size() * sizeof(bf16));
                for (int i = 0; i < vocab; ++i) {
                    host_logits[(size_t)i] = static_cast<float>(tmp[(size_t)i]);
                }
            }

            std::vector<int> order((size_t)vocab);
            for (int i = 0; i < vocab; ++i) order[(size_t)i] = i;
            int topn = std::min(10, vocab);
            std::partial_sort(order.begin(), order.begin() + topn, order.end(),
                              [&](int a, int b) { return host_logits[(size_t)a] > host_logits[(size_t)b]; });

            AILA_LOG_INFO("[DebugLogits] top-%d after prefill:", topn);
            for (int i = 0; i < topn; ++i) {
                int tid = order[(size_t)i];
                std::string piece = tokenizer_.decode(tid);
                AILA_LOG_INFO("  rank=%d id=%d logit=%.4f text='%s'",
                              i + 1, tid, host_logits[(size_t)tid], piece.c_str());
            }
        }

        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(max_new_tokens));

        if (gen_config.do_sample && gen_config.use_fixed_seed) {
            ops::set_sampling_seed(tuned_cfg.sampling_seed);
        }
        bool can_use_device_sample = ops::can_use_device_sampling(config_.vocab_size, tuned_cfg);

        std::string output_text;
        bool streaming = (token_callback != nullptr);
        bool think_prefix_restored = false;
        auto emit_stream_piece = [&](const std::string& piece) {
            if (generation_abort_requested_) {
                return;
            }
            if (restore_open_think_prefix && !think_prefix_restored) {
                think_prefix_restored = true;
                if (piece != "<think>") {
                    output_text += "<think>\n";
                    token_callback("<think>\n");
                    if (generation_abort_requested_) {
                        return;
                    }
                }
            }
            output_text += piece;
            token_callback(piece);
        };
        aila::chat::ThinkingBudgetController think_budget;
        think_budget.start(
            restore_open_think_prefix,
            tuned_cfg.thinking_budget_tokens,
            tokenizer_.special_token_id("<think>"),
            tokenizer_.special_token_id("</think>"));
        int effective_chunk_size = streaming ? std::max(1, tuned_cfg.stream_chunk_size)
                                             : std::max(1, tuned_cfg.decode_chunk_size);
        bool use_chunked_fast_decode =
            !think_budget.should_disable_chunked_decode() &&
            (!tuned_cfg.has_penalties()) &&
            (!tuned_cfg.do_sample || can_use_device_sample);
        int same_token_run = 0;
        int last_token = -1;
        int generated_count = 0;
        bool hit_eos = false;
        bool hit_loop_guard = false;
        if (use_chunked_fast_decode) {
            bool stop_decode = false;
            int available_tokens = 1;
            std::vector<int> host_tokens(static_cast<size_t>(max_new_tokens));
            int* generated_tokens_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(max_new_tokens + 1) * sizeof(int)));
            if (tuned_cfg.do_sample) {
                ops::sample_with_config_device(*ctx_, logits, config_.vocab_size,
                                               tuned_cfg, ops::next_sampling_uniform(),
                                               generated_tokens_device);
            } else {
                ops::argmax(*ctx_, logits, config_.vocab_size, generated_tokens_device);
            }

            while (generated_count < max_new_tokens && !stop_decode) {
                int chunk_begin = generated_count;
                int chunk_end = std::min(max_new_tokens, chunk_begin + effective_chunk_size);

                while (available_tokens < chunk_end) {
                    int current_index = available_tokens - 1;
                    Tensor& logits_next = backend_->forward(*ctx_, generated_tokens_device + current_index, 1);
                    if (tuned_cfg.do_sample) {
                        ops::sample_with_config_device(*ctx_, logits_next, config_.vocab_size,
                                                       tuned_cfg, ops::next_sampling_uniform(),
                                                       generated_tokens_device + available_tokens);
                    } else {
                        ops::argmax(*ctx_, logits_next, config_.vocab_size,
                                    generated_tokens_device + available_tokens);
                    }
                    ++available_tokens;
                }

                int copied = chunk_end - chunk_begin;
                auto copy_evt = ctx_->memcpy_d2h_async(host_tokens.data() + chunk_begin,
                                                       generated_tokens_device + chunk_begin,
                                                       static_cast<size_t>(copied) * sizeof(int));
                copy_evt.wait();

                for (int i = chunk_begin; i < chunk_end; ++i) {
                    int current_token = host_tokens[i];
                    if (tokenizer_.is_eos(current_token)) {
                        hit_eos = true;
                        stop_decode = true;
                        break;
                    }

                    if (current_token == last_token) same_token_run++;
                    else {
                        same_token_run = 1;
                        last_token = current_token;
                    }

                    generated_token_ids.push_back(current_token);
                    int step_index = generated_count;
                    generated_count++;
                    if (same_token_run >= 48) {
                        AILA_LOG_WARN("[GenerateMessages] Loop guard triggered (token=%d run=%d)",
                                      current_token, same_token_run);
                        hit_loop_guard = true;
                        stop_decode = true;
                        break;
                    }

                    if (debug_token_ids && step_index < 64) {
                        std::string piece = tokenizer_.decode(current_token);
                        AILA_LOG_INFO("[DebugToken] step=%d id=%d text='%s'",
                                      step_index, current_token, piece.c_str());
                    }
                    if (streaming) {
                        std::string token_text = tokenizer_.decode(current_token);
                        emit_stream_piece(token_text);
                        if (generation_abort_requested_) {
                            stop_decode = true;
                            break;
                        }
                    }
                }
            }
            ctx_->free_device(generated_tokens_device);
        } else {
                int next_token = ops::sample_with_config(*ctx_, logits, config_.vocab_size,
                                                         tuned_cfg, generated_token_ids);
                int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
                int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
                ctx_->memcpy_h2d(current_token_device, &next_token, sizeof(int));

                int current_token = next_token;
                std::vector<int> forced_tokens;
                size_t forced_token_index = 0;
                auto has_forced_token = [&]() -> bool {
                    return forced_token_index < forced_tokens.size();
                };
                auto pop_forced_token = [&]() -> int {
                    return forced_tokens[forced_token_index++];
                };

                while (generated_count < max_new_tokens) {
                    if (tokenizer_.is_eos(current_token)) {
                        hit_eos = true;
                        break;
                    }

                    if (current_token == last_token) same_token_run++;
                    else {
                        same_token_run = 1;
                        last_token = current_token;
                    }

                    generated_token_ids.push_back(current_token);
                    int step_index = generated_count;
                    generated_count++;
                    if (same_token_run >= 48) {
                        AILA_LOG_WARN("[GenerateMessages] Loop guard triggered (token=%d run=%d)",
                                      current_token, same_token_run);
                        hit_loop_guard = true;
                        break;
                    }

                    if (debug_token_ids && step_index < 64) {
                        std::string piece = tokenizer_.decode(current_token);
                        AILA_LOG_INFO("[DebugToken] step=%d id=%d text='%s'",
                                      step_index, current_token, piece.c_str());
                    }
                    if (streaming) {
                        std::string token_text = tokenizer_.decode(current_token);
                        emit_stream_piece(token_text);
                        if (generation_abort_requested_) {
                            break;
                        }
                    }

                    think_budget.observe_generated_token(current_token);
                    if (think_budget.needs_forced_close()) {
                        std::vector<int> close_ids = thinking_close_token_ids();
                        const int remaining_slots = max_new_tokens - generated_count;
                        if (remaining_slots < static_cast<int>(close_ids.size())) {
                            last_generation_think_close_truncated_ = true;
                            break;
                        }
                        forced_tokens = std::move(close_ids);
                        forced_token_index = 0;
                        think_budget.mark_forced_close();
                        last_generation_forced_think_close_ = true;
                    }

                    if (generated_count >= max_new_tokens) {
                        break;
                    }

                    Tensor& logits_next = backend_->forward(*ctx_, current_token_device, 1);
                    if (has_forced_token()) {
                        next_token = pop_forced_token();
                    } else {
                        next_token = ops::sample_with_config(*ctx_, logits_next, config_.vocab_size,
                                                             tuned_cfg, generated_token_ids);
                    }
                    ctx_->memcpy_h2d(next_token_device, &next_token, sizeof(int));
                    std::swap(current_token_device, next_token_device);
                    current_token = next_token;
                }
                ctx_->synchronize();
                ctx_->free_device(current_token_device);
                ctx_->free_device(next_token_device);
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        bool hit_length = !hit_eos && !hit_loop_guard &&
                          (generated_count >= max_new_tokens ||
                           last_generation_think_close_truncated_);
        last_generation_finish_reason_ =
            aila::chat::decode_finish_reason(hit_loop_guard, hit_length, false);

        if (!streaming && !generated_token_ids.empty()) {
            output_text = tokenizer_.decode(generated_token_ids);
        }
        if (restore_open_think_prefix && !output_text.empty() &&
            output_text.compare(0, 7, "<think>") != 0) {
            output_text.insert(0, "<think>\n");
        }
        align_backend_context_to_sequence(full_ids, generated_token_ids,
                                          total_prompt_len, "[GenerateMessages]");

        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        AILA_LOG_INFO("[GenerateMessages] Prompt=%d Generated=%zu in %.2f ms",
                      total_prompt_len, generated_token_ids.size(), ms);

        cached_ids_ = full_ids;
        cached_ids_.insert(cached_ids_.end(), generated_token_ids.begin(), generated_token_ids.end());
        return output_text;
    }

    std::string generate_chat_request_text(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback = nullptr) {
        std::vector<Message> legacy_messages;
        legacy_messages.reserve(request.messages.size());
        for (const auto& chat_msg : request.messages) {
            Message legacy;
            legacy.role = aila::chat::role_to_string(chat_msg.role);
            for (const auto& part : chat_msg.content) {
                ContentPart out;
                out.type = part.type;
                out.text = part.text;
                out.uri = part.uri;
                out.binary_data = part.binary_data;
                out.media_format = part.media_format;
                legacy.content.push_back(std::move(out));
            }
            legacy_messages.push_back(std::move(legacy));
        }
        return generate_messages(legacy_messages, request.generation_config, token_callback, &request);
    }

    aila::chat::AssistantChatResult generate_chat_request_result(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback = nullptr) {
        std::string raw = generate_chat_request_text(request, token_callback);
        if (last_error_code_ != EngineErrorCode::Ok) {
            return {};
        }
        aila::chat::AssistantChatResult result = aila::chat::parse_assistant_output(raw);
        result.finish_reason =
            aila::chat::decode_finish_reason(
                last_generation_finish_reason_ == "loop_guard",
                last_generation_finish_reason_ == "length",
                !result.tool_calls.empty());
        if (last_generation_forced_think_close_) {
            result.warnings.emplace_back("reasoning budget exhausted; forced </think>");
        }
        if (last_generation_think_close_truncated_) {
            result.warnings.emplace_back("reasoning budget exhausted but max_new_tokens prevented forced close");
        }
        const aila::chat::ToolPolicyValidation policy_validation =
            aila::chat::validate_tool_policy(request, result);
        result.warnings.insert(result.warnings.end(),
                               policy_validation.warnings.begin(),
                               policy_validation.warnings.end());
        if (policy_validation.hard_error) {
            result.finish_reason = "tool_policy";
        }
        result.metadata.template_name = last_generation_template_name_;
        result.metadata.model_family = current_model_family_name();
        result.metadata.reasoning_budget_tokens = request.generation_config.thinking_budget_tokens;
        result.metadata.reasoning_budget_forced_close = last_generation_forced_think_close_;
        result.metadata.reasoning_budget_truncated = last_generation_think_close_truncated_;
        result.metadata.tool_policy = tool_policy_name(request.tool_policy);
        result.metadata.tool_choice = tool_choice_name(request);
        return result;
    }

    aila::chat::StructuredStreamEvent make_final_stream_event(
        const aila::chat::AssistantChatResult& result) const {
        return aila::chat::final_stream_event_from_result(result);
    }

    int generate_chat_request_stream(
        const aila::chat::ChatRequest& request,
        ChatStreamCallback callback) {
        if (!callback) {
            set_error(EngineErrorCode::InvalidArgument, "chat stream callback is null");
            return -1;
        }

        aila::chat::StructuredStreamParser parser;
        bool aborted = false;

        generation_abort_requested_ = false;
        auto token_callback = [&](const std::string& piece) {
            if (aborted) {
                generation_abort_requested_ = true;
                return;
            }

            std::vector<aila::chat::StructuredStreamEvent> events;
            parser.push(piece, events);
            for (const auto& event : events) {
                if (!callback(event)) {
                    aborted = true;
                    generation_abort_requested_ = true;
                    break;
                }
            }
        };

        aila::chat::AssistantChatResult result =
            generate_chat_request_result(request, token_callback);
        generation_abort_requested_ = false;

        if (last_error_code_ != EngineErrorCode::Ok) {
            return -1;
        }
        if (aborted) {
            return 1;
        }

        std::vector<aila::chat::StructuredStreamEvent> tail_events;
        parser.finish(tail_events);
        for (const auto& event : tail_events) {
            if (!callback(event)) {
                return 1;
            }
        }

        if (!callback(make_final_stream_event(result))) {
            return 1;
        }
        return 0;
    }

    std::string generate_messages_json(const std::string& messages_json,
                                       const GenerationConfig& gen_config = GenerationConfig(),
                                       std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        aila::chat::ChatRequest request;
        std::string parse_error;
        if (!aila::chat::parse_chat_request_json(messages_json, gen_config, request, &parse_error)) {
            AILA_LOG_ERROR("[GenerateMessages] Invalid messages JSON: %s", parse_error.c_str());
            set_error(EngineErrorCode::JsonParseError, parse_error);
            return "";
        }
        return generate_chat_request_text(request, token_callback);
    }

    std::string generate_chat_json(const std::string& chat_request_json,
                                   const GenerationConfig& gen_config = GenerationConfig(),
                                   std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        aila::chat::ChatRequest request;
        std::string parse_error;
        if (!aila::chat::parse_chat_request_json(chat_request_json, gen_config, request, &parse_error)) {
            AILA_LOG_ERROR("[GenerateChatJson] Invalid chat request JSON: %s", parse_error.c_str());
            set_error(EngineErrorCode::JsonParseError, parse_error);
            return "";
        }

        aila::chat::AssistantChatResult result = generate_chat_request_result(request, token_callback);
        if (last_error_code_ != EngineErrorCode::Ok) {
            return "";
        }
        return aila::chat::assistant_result_to_json(result);
    }

    // ============================================================
    // Raw prefill for benchmark (no decode, no history)
    // ============================================================

    static int find_split_point(const float* samples, int n_samples, int target_sample, float search_sec) {
        int search_half = static_cast<int>(search_sec * 16000);
        int lo = target_sample - search_half;
        int hi = target_sample + search_half;
        if (lo < 0) lo = 0;
        if (hi > n_samples) hi = n_samples;

        int win_samples = 1600; // 100ms at 16kHz
        float best_energy = 1e30f;
        int best_center = target_sample;

        for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
            float energy = 0.0f;
            int end = pos + win_samples;
            if (end > n_samples) end = n_samples;
            for (int j = pos; j < end; j++) {
                energy += samples[j] * samples[j];
            }
            energy /= (end - pos);
            if (energy < best_energy) {
                best_energy = energy;
                best_center = pos + (end - pos) / 2;
            }
        }
        return best_center;
    }

    static bool should_insert_boundary_space(int prev_ch, int next_ch) {
        if (prev_ch <= 0 || next_ch <= 0) return false;
        if (std::isspace(static_cast<unsigned char>(prev_ch))) return false;
        if (std::isspace(static_cast<unsigned char>(next_ch))) return false;
        if (std::ispunct(static_cast<unsigned char>(next_ch))) return false;
        return true;
    }

    // ASR transcription of a single raw segment
    std::string transcribe_segment_raw(const std::vector<float>& seg_buf,
                                       const GenerationConfig& gen_config,
                                       const std::string& forced_language,
                                       const std::string& system_prompt,
                                       const std::string& past_text,
                                       std::string* language_out = nullptr) {
        using bf16 = sycl::ext::oneapi::bfloat16;

        int audio_start_id = model_spec_.audio_start_token_id;
        int audio_end_id   = model_spec_.audio_end_token_id;
        int audio_pad_id   = model_spec_.audio_token_id;
        int im_start_id    = config_.im_start_id;
        int im_end_id      = config_.im_end_id;

        // 2.1 Compute Mel spectrogram for the segment
        MelSpectrogram mel;
        std::string prep_err;
        if (!compute_mel_spectrogram(seg_buf, mel, &prep_err)) {
            set_error(EngineErrorCode::RuntimeError, "Segment spectrogram failed: " + prep_err);
            return "";
        }

        // 2.2 Upload Mel to GPU and run Audio Encoder
        int mel_padded_frames = mel.n_frames;
        int mel_actual_frames = mel.actual_frames;
        int nM = mel.n_mels;
        std::vector<bf16> mel_bf16(static_cast<size_t>(mel_padded_frames) * nM);
        for (int f = 0; f < mel_padded_frames; ++f)
            for (int m = 0; m < nM; ++m)
                mel_bf16[m * mel_padded_frames + f] = bf16(mel.data[f * nM + m]);

        Tensor mel_device = Tensor::allocate(*ctx_, {1, nM, mel_padded_frames});
        ctx_->memcpy_h2d(mel_device.data(), mel_bf16.data(), mel_bf16.size() * sizeof(bf16));

        int audio_len = 0;
        int od = model_spec_.audio.output_dim;
        int max_audio_len = ((mel_actual_frames + 99) / 100) * 13 + 32;
        Tensor af_tmp = Tensor::allocate(*ctx_, {max_audio_len, od});
        std::string enc_error;
        if (!audio_encoder_->encode(*ctx_, mel_device, mel_actual_frames,
                                     af_tmp, audio_len, &enc_error)) {
            set_error(EngineErrorCode::RuntimeError, "Audio encoding failed: " + enc_error);
            return "";
        }
        Tensor audio_features = Tensor::allocate(*ctx_, {audio_len, od});
        bf16* af_dst = audio_features.data_as<bf16>();
        bf16* af_src = af_tmp.data_as<bf16>();
        ctx_->queue().memcpy(af_dst, af_src, static_cast<size_t>(audio_len) * od * sizeof(bf16));

        std::vector<bf16> audio_bf16(static_cast<size_t>(audio_len) * model_spec_.audio.output_dim);
        ctx_->memcpy_d2h(audio_bf16.data(), audio_features.data(), audio_bf16.size() * sizeof(bf16));
        ctx_->synchronize();

        // Collapse guard loop
        bool use_past = (!past_text.empty());
        int retry_count = 0;
        std::string seg_text = "";
        std::string seg_lang = "";

        while (retry_count < 2) {
            std::vector<int> prompt_ids;
            auto add_text = [&](const std::string& t) {
                auto ids = tokenizer_.encode(t);
                prompt_ids.insert(prompt_ids.end(), ids.begin(), ids.end());
            };

            // <|im_start|>system\n[system prompt]\n<|im_end|>\n
            prompt_ids.push_back(im_start_id);
            add_text("system\n");
            if (!system_prompt.empty()) {
                add_text(system_prompt);
            }
            prompt_ids.push_back(im_end_id);
            add_text("\n");

            // <|im_start|>user\n<|audio_start|><|audio_pad|>xN<|audio_end|>\n<|im_end|>\n
            prompt_ids.push_back(im_start_id);
            add_text("user\n");
            prompt_ids.push_back(audio_start_id);
            for (int i = 0; i < audio_len; ++i) prompt_ids.push_back(audio_pad_id);
            prompt_ids.push_back(audio_end_id);
            add_text("\n");
            prompt_ids.push_back(im_end_id);
            add_text("\n");

            // <|im_start|>assistant\n
            prompt_ids.push_back(im_start_id);
            add_text("assistant\n");

            if (!forced_language.empty()) {
                std::string normalized_forced = aila_asr::normalize_language_name(forced_language);
                add_text("language " + normalized_forced);
                int asr_text_id = tokenizer_.special_token_id("<asr_text>");
                if (asr_text_id != -1) {
                    prompt_ids.push_back(asr_text_id);
                }
            }

            if (use_past && !past_text.empty()) {
                std::vector<int> past_ids = tokenizer_.encode(past_text);
                prompt_ids.insert(prompt_ids.end(), past_ids.begin(), past_ids.end());
                int asr_text_id = tokenizer_.special_token_id("<asr_text>");
                if (asr_text_id != -1) {
                    prompt_ids.push_back(asr_text_id);
                }
            }

            std::vector<int> pad_positions;
            for (size_t i = 0; i < prompt_ids.size(); ++i)
                if (prompt_ids[i] == audio_pad_id) pad_positions.push_back(static_cast<int>(i));

            backend_->reset();
            cached_ids_.clear();
            backend_->set_embedding_overrides(pad_positions, audio_bf16, model_spec_.audio.output_dim);

            if (!aila::env::read_flag("AILA_NO_MROPE", false)) {
                int total_len = static_cast<int>(prompt_ids.size());
                std::vector<int> pos_t(total_len), pos_h(total_len), pos_w(total_len);
                for (int i = 0; i < total_len; ++i) {
                    pos_t[i] = i; pos_h[i] = i; pos_w[i] = i;
                }
                backend_->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, 0);
            }

            int* device_ids = static_cast<int*>(ctx_->alloc_device(prompt_ids.size() * sizeof(int)));
            ctx_->memcpy_h2d_async(device_ids, prompt_ids.data(), prompt_ids.size() * sizeof(int));
            GenerationConfig tuned_gen = gen_config;
            if (tuned_gen.max_new_tokens <= 0) tuned_gen.max_new_tokens = 256;

            int prompt_len = static_cast<int>(prompt_ids.size());
            Tensor* logits_ptr = &backend_->forward(*ctx_, device_ids, prompt_len);
            ctx_->free_device(device_ids);

            if (backend_->supports_vision_embedding_override())
                backend_->clear_mrope_positions();

            std::vector<int> generated_ids;
            int* one_token_dev = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            for (int step = 0; step < tuned_gen.max_new_tokens; ++step) {
                int next_token;
                if (!tuned_gen.do_sample) {
                    int* argmax_dev = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
                    ops::argmax(*ctx_, *logits_ptr, config_.vocab_size, argmax_dev);
                    ctx_->memcpy_d2h(&next_token, argmax_dev, sizeof(int));
                    ctx_->free_device(argmax_dev);
                } else {
                    next_token = ops::sample_with_config(*ctx_, *logits_ptr, config_.vocab_size,
                                                         tuned_gen, generated_ids);
                }

                if (next_token == config_.eos_token_id || next_token == config_.im_end_id) break;
                generated_ids.push_back(next_token);
                ctx_->memcpy_h2d(one_token_dev, &next_token, sizeof(int));
                logits_ptr = &backend_->forward(*ctx_, one_token_dev, 1);
            }
            ctx_->free_device(one_token_dev);

            std::string raw = tokenizer_.decode(generated_ids);
            aila_asr::parse_asr_output(raw, forced_language, seg_lang, seg_text);

            // Collapse detection: if core audio is long (>= 8s) and we generated too few tokens
            bool collapse = false;
            if (use_past && !past_text.empty()) {
                float core_sec = static_cast<float>(seg_buf.size()) / 16000.0f;
                if (core_sec >= 8.0f) {
                    int min_tokens = static_cast<int>(core_sec * 1.75f);
                    if (min_tokens < 12) min_tokens = 12;
                    if (static_cast<int>(generated_ids.size()) < min_tokens) {
                        collapse = true;
                    }
                }
            }

            if (collapse) {
                AILA_LOG_WARN("[Transcribe] Collapse detected (tokens=%zu). Retrying without past text history.", generated_ids.size());
                use_past = false;
                retry_count++;
                continue;
            }
            last_transcribe_tokens_ += static_cast<int>(generated_ids.size());
            break;
        }

        if (language_out) {
            *language_out = seg_lang;
        }

        return seg_text;
    }

    // TTS audio code synthesis from text
    bool synthesize_codes(const std::vector<int>& text_tokens,
                          const std::vector<float>& speaker_embedding,
                          const GenerationConfig& gen_config,
                          std::vector<int32_t>& out_codes,
                          int& out_n_frames) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;
        if (model_spec_.family != ModelFamily::Qwen3TTS || !backend_) {
            set_error(EngineErrorCode::RuntimeError, "TTS backend not initialized");
            return false;
        }

        auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
        if (!tts_backend) {
            set_error(EngineErrorCode::RuntimeError, "Invalid TTS backend class");
            return false;
        }

        return tts_backend->synthesize_codes(*ctx_, text_tokens, speaker_embedding, 0, {}, 0, gen_config, out_codes, out_n_frames);
    }

    // TTS audio samples synthesis from text using Mimi Decoder
    bool synthesize_wav(const std::vector<int>& text_tokens,
                        const std::vector<float>& speaker_embedding,
                        const GenerationConfig& gen_config,
                        std::vector<float>& out_samples) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;
        if (model_spec_.family != ModelFamily::Qwen3TTS || !backend_) {
            set_error(EngineErrorCode::RuntimeError, "TTS backend not initialized");
            return false;
        }

        auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
        if (!tts_backend) {
            set_error(EngineErrorCode::RuntimeError, "Invalid TTS backend class");
            return false;
        }

        std::vector<int32_t> codes;
        int n_frames = 0;
        bool ok = tts_backend->synthesize_codes(*ctx_, text_tokens, speaker_embedding, 0, {}, 0, gen_config, codes, n_frames);
        if (!ok) {
            set_error(EngineErrorCode::RuntimeError, "TTS synthesize_codes failed");
            return false;
        }

        return tts_backend->decode_mimi_vocoder(*ctx_, codes, n_frames, out_samples);
    }

    // TTS audio samples synthesis directly from raw text using Mimi Decoder
    bool synthesize_text_to_wav(const std::string& text,
                               const std::vector<float>& speaker_embedding,
                               const GenerationConfig& gen_config,
                               std::vector<float>& out_samples) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;
        if (model_spec_.family != ModelFamily::Qwen3TTS || !backend_) {
            set_error(EngineErrorCode::RuntimeError, "TTS backend not initialized");
            return false;
        }

        std::string formatted_text = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> tokens = tokenizer_.encode(formatted_text);

        std::string ids_str = "";
        for (int id : tokens) {
            ids_str += std::to_string(id) + " ";
        }
        AILA_LOG_INFO("[TTS] Input: \"%s\", Tokenized into %zu tokens: [ %s]",
                      text.c_str(), tokens.size(), ids_str.c_str());

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = synthesize_wav(tokens, speaker_embedding, gen_config, out_samples);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (ok && !out_samples.empty()) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double audio_s = static_cast<double>(out_samples.size()) / 24000.0;
            double rtf = elapsed_ms / 1000.0 / audio_s;
            AILA_LOG_INFO("[TTS] Synthesis complete: %.0f ms, %.2f s audio, RTF=%.3f (%srealtime)",
                          elapsed_ms, audio_s, rtf, (rtf < 1.0 ? "" : "slower than "));
        }
        return ok;
    }

    // Unified TTS speech synthesis supporting Base (voice cloning), CustomVoice
    // (speaker name), and VoiceDesign (instruct text) models.
    // Exactly one of reference_audio_path / speaker_name / instruct_text should be
    // non-empty to select the mode; all empty = default voice.
    bool synthesizeSpeech(const std::string& text,
                          const std::string& reference_audio_path,
                          const std::string& speaker_name,
                          const std::string& instruct_text,
                          const std::string& language,
                          const GenerationConfig& gen_config,
                          std::vector<float>& out_samples,
                          const std::string& reference_text = "",
                          VoiceCloneMode clone_mode = VoiceCloneMode::Auto) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;
        if (model_spec_.family != ModelFamily::Qwen3TTS || !backend_) {
            set_error(EngineErrorCode::RuntimeError, "TTS backend not initialized");
            return false;
        }

        auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
        if (!tts_backend) {
            set_error(EngineErrorCode::RuntimeError, "Invalid TTS backend class");
            return false;
        }

        // 1. Determine mode: Base (voice cloning), CustomVoice (speaker name), or
        //    VoiceDesign / default (instruct-only or no identity)
        std::vector<float> spk_emb;
        int spk_id = 0;
        VoiceClonePrompt clone_prompt;
        const VoiceClonePrompt* p_clone_prompt = nullptr;

        if (!reference_audio_path.empty()) {
            // Base mode: ECAPA-TDNN voice cloning from reference audio
            // extractSpeakerEmbedding sets its own detailed error on failure
            if (!extractSpeakerEmbedding(reference_audio_path, spk_emb)) {
                return false;
            }

            clone_prompt.mode = clone_mode;
            clone_prompt.speaker_embedding = spk_emb;

            bool enable_icl = false;
            if (clone_mode == VoiceCloneMode::Icl) {
                enable_icl = true;
            } else if (clone_mode == VoiceCloneMode::Auto && !reference_text.empty()) {
                enable_icl = true;
            }

            if (enable_icl) {
                TTSReferenceCodes ref_codes;
                if (!lookupRefCodesCache(reference_audio_path, ref_codes)) {
                    if (!mimi_encoder_ || !mimi_encoder_->isLoaded()) {
                        std::string tokenizer_path = model_dir_ + "/speech_tokenizer/model.safetensors";
                        mimi_encoder_ = std::make_unique<aila::audio::MimiEncoder>();
                        std::string enc_err;
                        if (!mimi_encoder_->loadWeights(tokenizer_path, &enc_err)) {
                            set_error(EngineErrorCode::RuntimeError, "Failed to load Mimi Audio Encoder: " + enc_err);
                            return false;
                        }
                    }
                    std::string enc_err;
                    if (!mimi_encoder_->encodeFromFile(reference_audio_path, ref_codes, &enc_err)) {
                        set_error(EngineErrorCode::RuntimeError, "Failed to encode reference audio codes: " + enc_err);
                        return false;
                    }
                    cacheRefCodes(reference_audio_path, ref_codes);
                }

                clone_prompt.reference_codes = std::move(ref_codes);
                clone_prompt.reference_text = reference_text;
                if (!reference_text.empty()) {
                    clone_prompt.reference_text_tokens = tokenizer_.encode(reference_text);
                }
                p_clone_prompt = &clone_prompt;
                AILA_LOG_INFO("[TTS] ICL mode active: %d ref codes frames, %zu ref text tokens",
                              clone_prompt.reference_codes.frames,
                              clone_prompt.reference_text_tokens.size());
            }
        } else if (!speaker_name.empty()) {
            // CustomVoice mode: use spk_id token from model's pre-trained voice map
            spk_id = getSpeakerId(speaker_name);
            if (spk_id == 0) {
                set_error(EngineErrorCode::RuntimeError,
                    "Unknown speaker: " + speaker_name);
                return false;
            }
            AILA_LOG_INFO("[TTS] CustomVoice speaker: %s (spk_id=%d)",
                speaker_name.c_str(), spk_id);
        }
        // else: no speaker identity -> default voice or VoiceDesign (instruct-only)

        // 2. Tokenize instruct text (optional, for VoiceDesign or style override)
        // IMPORTANT: instruct text is raw text, NOT ChatML-formatted.
        // The Python reference tokenizes instruct directly without <|im_start|>/<|im_end|>.
        std::vector<int> instruct_tokens;
        if (!instruct_text.empty()) {
            instruct_tokens = tokenizer_.encode(instruct_text);
            AILA_LOG_INFO("[TTS] Instruct: \"%s\", %zu tokens",
                instruct_text.c_str(), instruct_tokens.size());
        }

        // 3. Get language codec ID
        int lang_id = getLanguageId(language);
        if (!language.empty()) {
            AILA_LOG_INFO("[TTS] Language: %s (id=%d)", language.c_str(), lang_id);
        }

        // 4. Format and tokenize the main text
        std::string formatted_text = "<|im_start|>assistant\n" + text
            + "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<int> tokens = tokenizer_.encode(formatted_text);
        AILA_LOG_INFO("[TTS] Input: \"%s\", %zu tokens", text.c_str(), tokens.size());

        // 5. Synthesize codes
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> codes;
        int n_frames = 0;
        bool ok = tts_backend->synthesize_codes(*ctx_, tokens, spk_emb, spk_id,
                                                 instruct_tokens, lang_id,
                                                 gen_config, codes, n_frames,
                                                 p_clone_prompt);
        if (!ok) {
            set_error(EngineErrorCode::RuntimeError, "TTS synthesize_codes failed");
            return false;
        }

        // 6. Decode codes to audio via Mimi vocoder
        if (p_clone_prompt && p_clone_prompt->is_icl() && p_clone_prompt->reference_codes.frames > 0) {
            int ref_len = p_clone_prompt->reference_codes.frames;
            int total_len = ref_len + n_frames;
            std::vector<int32_t> codes_for_decode;
            codes_for_decode.reserve(static_cast<size_t>(total_len) * 16);
            codes_for_decode.insert(codes_for_decode.end(),
                                    p_clone_prompt->reference_codes.codes.begin(),
                                    p_clone_prompt->reference_codes.codes.end());
            codes_for_decode.insert(codes_for_decode.end(), codes.begin(), codes.end());

            std::vector<float> full_decoded;
            ok = tts_backend->decode_mimi_vocoder(*ctx_, codes_for_decode, total_len, full_decoded);
            if (ok && !full_decoded.empty()) {
                size_t cut = static_cast<size_t>(static_cast<double>(ref_len) / std::max(total_len, 1) * full_decoded.size());
                if (cut < full_decoded.size()) {
                    out_samples.assign(full_decoded.begin() + cut, full_decoded.end());
                } else {
                    out_samples.clear();
                }
            }
        } else {
            ok = tts_backend->decode_mimi_vocoder(*ctx_, codes, n_frames, out_samples);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        if (ok && !out_samples.empty()) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double audio_s = static_cast<double>(out_samples.size()) / 24000.0;
            double rtf = elapsed_ms / 1000.0 / audio_s;
            AILA_LOG_INFO("[TTS] Synthesis complete: %.0f ms, %.2f s audio, RTF=%.3f",
                          elapsed_ms, audio_s, rtf);
        }
        return ok;
    }

    // Streaming TTS: calls callback from background thread with PCM audio chunks.
    // Returns immediately; callback fires from a worker thread as audio is generated.
    std::thread synthesizeSpeechStream(
        const std::string& text,
        const std::string& reference_audio_path,
        const std::string& speaker_name,
        const std::string& instruct_text,
        const std::string& language,
        const GenerationConfig& gen_config,
        std::function<void(const float*, int)> callback,
        int stream_batch_frames = 6,
        const std::string& reference_text = "",
        VoiceCloneMode clone_mode = VoiceCloneMode::Auto
    ) {
        clear_error();
        if (reject_yolo_capability("TTS")) return {};
        return std::thread([this, text, reference_audio_path, speaker_name,
                            instruct_text, language, gen_config, callback,
                            stream_batch_frames, reference_text, clone_mode]() {
            // Determine mode
            std::vector<float> spk_emb;
            int spk_id = 0;
            VoiceClonePrompt clone_prompt;
            const VoiceClonePrompt* p_clone_prompt = nullptr;

            if (!reference_audio_path.empty()) {
                extractSpeakerEmbedding(reference_audio_path, spk_emb);
                clone_prompt.mode = clone_mode;
                clone_prompt.speaker_embedding = spk_emb;

                bool enable_icl = (clone_mode == VoiceCloneMode::Icl) ||
                                  (clone_mode == VoiceCloneMode::Auto && !reference_text.empty());
                if (enable_icl) {
                    TTSReferenceCodes ref_codes;
                    if (!lookupRefCodesCache(reference_audio_path, ref_codes)) {
                        if (!mimi_encoder_ || !mimi_encoder_->isLoaded()) {
                            std::string tokenizer_path = model_dir_ + "/speech_tokenizer/model.safetensors";
                            mimi_encoder_ = std::make_unique<aila::audio::MimiEncoder>();
                            std::string enc_err;
                            mimi_encoder_->loadWeights(tokenizer_path, &enc_err);
                        }
                        if (mimi_encoder_ && mimi_encoder_->isLoaded()) {
                            std::string enc_err;
                            if (mimi_encoder_->encodeFromFile(reference_audio_path, ref_codes, &enc_err)) {
                                cacheRefCodes(reference_audio_path, ref_codes);
                            }
                        }
                    }
                    clone_prompt.reference_codes = std::move(ref_codes);
                    clone_prompt.reference_text = reference_text;
                    if (!reference_text.empty()) {
                        clone_prompt.reference_text_tokens = tokenizer_.encode(reference_text);
                    }
                    p_clone_prompt = &clone_prompt;
                }
            } else if (!speaker_name.empty()) {
                spk_id = getSpeakerId(speaker_name);
            }

            // Tokenize
            std::vector<int> instruct_tokens;
            if (!instruct_text.empty()) {
                instruct_tokens = tokenizer_.encode(instruct_text);
            }
            int lang_id = getLanguageId(language);
            std::string formatted_text = "<|im_start|>assistant\n" + text
                + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int> tokens = tokenizer_.encode(formatted_text);

            // Run streaming synthesis
            auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
            if (!tts_backend) return;

            tts_backend->synthesize_codes_stream(
                *ctx_, tokens, spk_emb, spk_id, instruct_tokens, lang_id,
                gen_config, stream_batch_frames,
                [&](const std::vector<float>& chunk) {
                    if (!chunk.empty()) {
                        callback(chunk.data(), static_cast<int>(chunk.size()));
                    }
                },
                p_clone_prompt);
        });
    }

    // TTS audio discrete codes decoding using Mimi Decoder
    bool decode_mimi_vocoder(const std::vector<int32_t>& codes,
                             int n_frames,
                             std::vector<float>& out_samples) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;
        if (model_spec_.family != ModelFamily::Qwen3TTS || !backend_) {
            set_error(EngineErrorCode::RuntimeError, "TTS backend not initialized");
            return false;
        }

        auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
        if (!tts_backend) {
            set_error(EngineErrorCode::RuntimeError, "Invalid TTS backend class");
            return false;
        }

        return tts_backend->decode_mimi_vocoder(*ctx_, codes, n_frames, out_samples);
    }

    // Extract speaker embedding from a reference audio file for TTS voice cloning.
    // Uses GPU-accelerated ECAPA-TDNN (SYCL kernels) for the forward pass;
    // mel spectrogram is computed on CPU.
    //
    // Caching: results are cached in memory (keyed by audio file path) and
    // optionally persisted to disk.  Disk cache location:
    //   - If ref_cache_dir_ is set: <cache_dir>/<basename>.ref.bin
    //   - Otherwise: <audio_path>.ref.bin alongside the reference audio
    //
    // The audio file must be mono, and will be automatically resampled to 24kHz if needed.
    // Supported formats: WAV, MP3, FLAC.
    // Returns true on success, false on error (call last_error_message() for details).
    bool extractSpeakerEmbedding(const std::string& audio_path,
                                 std::vector<float>& embedding) {
        clear_error();
        if (reject_yolo_capability("TTS")) return false;

        // 1. Load CPU encoder first to determine the correct embedding dimension
        //    (1024 for 0.6B, 2048 for 1.7B) — this is fast (~100ms, weights are mmap'd).
        std::string safetensors_path = model_dir_ + "/model.safetensors";
        std::string error;
        aila::audio::SpeakerEncoder cpu_enc;
        if (!cpu_enc.loadWeights(safetensors_path, &error)) {
            set_error(EngineErrorCode::RuntimeError, "Failed to load speaker encoder weights: " + error);
            return false;
        }
        int spk_dim = cpu_enc.embeddingDim();

        // 2. Check cache with correct dimension
        if (lookupRefCache(audio_path, spk_dim, embedding)) {
            AILA_LOG_INFO("[TTS] Speaker embedding loaded from cache (dim=%d)", spk_dim);
            return true;
        }

        // 3. Cache miss — extract (GPU-accelerated)

        AudioBuffer audio;
        if (!load_audio(audio_path, audio, &error)) {
            set_error(EngineErrorCode::RuntimeError, "Failed to load audio: " + error);
            return false;
        }
        std::vector<float> mono;
        if (audio.channels > 1) {
            mono.resize(audio.samples.size() / audio.channels);
            for (size_t i = 0; i < mono.size(); ++i) {
                float sum = 0.0f;
                for (int c = 0; c < audio.channels; ++c)
                    sum += audio.samples[i * audio.channels + c];
                mono[i] = sum / audio.channels;
            }
        } else {
            mono = std::move(audio.samples);
        }
        std::vector<float> resampled;
        if (audio.sample_rate != 24000) {
            double ratio = (double)audio.sample_rate / 24000.0;
            size_t outSize = (size_t)std::round(mono.size() / ratio);
            resampled.resize(outSize);
            auto get_sample = [&](int idx) -> float {
                if (idx < 0) return mono[0];
                if (idx >= (int)mono.size()) return mono[mono.size() - 1];
                return mono[idx];
            };
            for (size_t i = 0; i < outSize; ++i) {
                double t = i * ratio;
                int idx = (int)std::floor(t);
                double f = t - idx;
                float y0 = get_sample(idx - 1), y1 = get_sample(idx);
                float y2 = get_sample(idx + 1), y3 = get_sample(idx + 2);
                float a0 = -0.5f*y0+1.5f*y1-1.5f*y2+0.5f*y3;
                float a1 = y0-2.5f*y1+2.0f*y2-0.5f*y3;
                float a2 = -0.5f*y0+0.5f*y2;
                resampled[i] = (float)(((a0*f+a1)*f+a2)*f+y1);
            }
        } else {
            resampled = std::move(mono);
        }

        // Step 3: run ECAPA-TDNN forward pass on CPU (f32 precision, proven quality).
        // extractEmbedding internally computes mel spectrogram from raw audio samples.
        if (!cpu_enc.extractEmbedding(resampled.data(), (int)resampled.size(), embedding, &error)) {
            set_error(EngineErrorCode::RuntimeError, "Speaker encoder forward pass failed: " + error);
            return false;
        }

        // 4. Save to caches
        cacheRefEmbedding(audio_path, embedding);
        AILA_LOG_INFO("[TTS] Speaker embedding extracted and cached (dim=%d)", spk_dim);
        return true;
    }

    // Forced alignment: given audio samples + text → word-level timestamps.
    std::vector<aila::AlignedWord> align(
        const std::vector<float>& audio_samples, int sample_rate,
        const std::string& text, const std::string& language) {

        using bf16 = sycl::ext::oneapi::bfloat16;
        clear_error();
        if (reject_yolo_capability("ASR/forced alignment")) return {};

        if (model_spec_.family != ModelFamily::Qwen3ForceAligner || !audio_encoder_) {
            set_error(EngineErrorCode::RuntimeError, "ForceAligner backend not initialized");
            return {};
        }

        auto fa_bf16 = dynamic_cast<Qwen3ForceAlignerBackend*>(backend_.get());
        auto fa_nf4 = dynamic_cast<Qwen3ForceAlignerBnb4Backend*>(backend_.get());
        if (!fa_bf16 && !fa_nf4) {
            set_error(EngineErrorCode::RuntimeError, "Invalid ForceAligner backend class");
            return {};
        }

        // 1. Resample to 16kHz
        std::vector<float> mono_16k;
        {
            std::vector<float> mono = audio_samples;
            if (sample_rate != 16000) {
                double ratio = static_cast<double>(sample_rate) / 16000.0;
                size_t out_size = static_cast<size_t>(std::round(mono.size() / ratio));
                mono_16k.resize(out_size);
                for (size_t i = 0; i < out_size; ++i) {
                    double t = i * ratio;
                    int idx = static_cast<int>(std::floor(t));
                    double f = t - idx;
                    auto get_s = [&](int ii) -> float {
                        if (ii < 0) return mono[0];
                        if (ii >= static_cast<int>(mono.size())) return mono[mono.size() - 1];
                        return mono[ii];
                    };
                    float y0 = get_s(idx - 1), y1 = get_s(idx);
                    float y2 = get_s(idx + 1), y3 = get_s(idx + 2);
                    float a0 = -0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3;
                    float a1 = y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3;
                    float a2 = -0.5f*y0 + 0.5f*y2;
                    mono_16k[i] = static_cast<float>(((a0 * f + a1) * f + a2) * f + y1);
                }
            } else {
                mono_16k = std::move(mono);
            }
        }
        int audio_n_samples = static_cast<int>(mono_16k.size());
        // 2. Encode input text
        auto [word_list, prompt_text] = aila::ForceAlignerPostProcess::encode_input(text, language);

        // 3. Compute Mel spectrogram
        MelSpectrogram mel;
        std::string prep_err;
        if (!compute_mel_spectrogram(mono_16k, mel, &prep_err)) {
            set_error(EngineErrorCode::RuntimeError, "Mel spectrogram failed: " + prep_err);
            return {};
        }

        // 4. Audio encoder forward
        int mel_padded_frames = mel.n_frames;
        int mel_actual_frames = mel.actual_frames;
        int nM = mel.n_mels;
        int od = model_spec_.audio.output_dim;

        std::vector<bf16> mel_bf16(static_cast<size_t>(mel_padded_frames) * nM);
        for (int f = 0; f < mel_padded_frames; ++f)
            for (int m = 0; m < nM; ++m)
                mel_bf16[m * mel_padded_frames + f] = bf16(mel.data[f * nM + m]);

        Tensor mel_device = Tensor::allocate(*ctx_, {1, nM, mel_padded_frames});
        ctx_->memcpy_h2d(mel_device.data(), mel_bf16.data(), mel_bf16.size() * sizeof(bf16));

        int audio_len = 0;
        int max_audio_len = ((mel_actual_frames + 99) / 100) * 13 + 32;
        Tensor af_tmp = Tensor::allocate(*ctx_, {max_audio_len, od});
        std::string enc_error;
        if (!audio_encoder_->encode(*ctx_, mel_device, mel_actual_frames,
                                     af_tmp, audio_len, &enc_error)) {
            set_error(EngineErrorCode::RuntimeError, "Audio encoding failed: " + enc_error);
            return {};
        }
        Tensor audio_features = Tensor::allocate(*ctx_, {audio_len, od});
        {
            bf16* af_dst = audio_features.data_as<bf16>();
            bf16* af_src = af_tmp.data_as<bf16>();
            ctx_->queue().memcpy(af_dst, af_src, static_cast<size_t>(audio_len) * od * sizeof(bf16));
        }

        std::vector<bf16> audio_bf16(static_cast<size_t>(audio_len) * od);
        ctx_->memcpy_d2h(audio_bf16.data(), audio_features.data(), audio_bf16.size() * sizeof(bf16));
        ctx_->synchronize();

        // 5. Build prompt token IDs
        int audio_start_id = model_spec_.audio_start_token_id;
        int audio_end_id   = model_spec_.audio_end_token_id;
        int audio_pad_id   = model_spec_.audio_token_id;
        int ts_token_id    = model_spec_.timestamp_token_id;

        std::vector<int> prompt_ids;
        prompt_ids.push_back(audio_start_id);
        for (int i = 0; i < audio_len; ++i) prompt_ids.push_back(audio_pad_id);
        prompt_ids.push_back(audio_end_id);

        // Tokenize each word and insert explicit timestamp tokens between them.
        // This ensures timestamp_token_id (151705) is used, not text "<timestamp>".
        for (size_t wi = 0; wi < word_list.size(); ++wi) {
            auto word_ids = tokenizer_.encode(word_list[wi]);
            prompt_ids.insert(prompt_ids.end(), word_ids.begin(), word_ids.end());
            prompt_ids.push_back(ts_token_id);  // <timestamp>
            prompt_ids.push_back(ts_token_id);  // <timestamp>
        }

        std::vector<int> pad_positions;
        for (size_t i = 0; i < prompt_ids.size(); ++i)
            if (prompt_ids[i] == audio_pad_id) pad_positions.push_back(static_cast<int>(i));

        // 6. Run ForceAligner forward
        backend_->reset();
        cached_ids_.clear();
        backend_->set_embedding_overrides(pad_positions, audio_bf16, od);

        if (!aila::env::read_flag("AILA_NO_MROPE", false)) {
            int total_len = static_cast<int>(prompt_ids.size());
            std::vector<int> pos_t(total_len), pos_h(total_len), pos_w(total_len);
            for (int i = 0; i < total_len; ++i) {
                pos_t[i] = i; pos_h[i] = i; pos_w[i] = i;
            }
            backend_->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, 0);
        }

        int* device_ids = static_cast<int*>(ctx_->alloc_device(prompt_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d_async(device_ids, prompt_ids.data(), prompt_ids.size() * sizeof(int));
        int prompt_len = static_cast<int>(prompt_ids.size());

        int classify_num;
        Tensor* logits_all;
        if (fa_bf16) {
            logits_all = &fa_bf16->forward_all(*ctx_, device_ids, prompt_len);
            classify_num = fa_bf16->classify_num();
        } else {
            logits_all = &fa_nf4->forward_all(*ctx_, device_ids, prompt_len);
            classify_num = fa_nf4->classify_num();
        }
        ctx_->free_device(device_ids);

        if (backend_->supports_vision_embedding_override())
            backend_->clear_mrope_positions();

        // 7. Download logits (GPU bf16 → host bf16 → float conversion)
        size_t total_elts = static_cast<size_t>(prompt_len) * classify_num;
        std::vector<bf16> host_bf16(total_elts);
        ctx_->memcpy_d2h(host_bf16.data(), logits_all->data(), total_elts * sizeof(bf16));
        ctx_->synchronize();
        std::vector<float> host_logits(total_elts);
        for (size_t i = 0; i < total_elts; ++i) {
            host_logits[i] = static_cast<float>(host_bf16[i]);
        }

        // 8. Extract timestamps + fix + build output
        auto raw_ts = aila::ForceAlignerPostProcess::extract_timestamps(
            host_logits.data(), prompt_len, classify_num,
            prompt_ids.data(), prompt_len,
            ts_token_id, model_spec_.timestamp_segment_time);

        AILA_LOG_INFO("[Align] Raw timestamps: %zu values for %zu words",
                      raw_ts.size(), word_list.size());

        auto fixed_ts = aila::ForceAlignerPostProcess::fix_timestamp(raw_ts);
        auto result = aila::ForceAlignerPostProcess::build_output(word_list, fixed_ts);

        return result;
    }

    // Forced alignment with pre-tokenized word list (bypasses built-in tokenizer).
    std::vector<aila::AlignedWord> align_words(
        const std::vector<float>& audio_samples, int sample_rate,
        const std::vector<std::string>& word_list) {

        using bf16 = sycl::ext::oneapi::bfloat16;
        clear_error();
        if (reject_yolo_capability("ASR/forced alignment")) return {};

        if (model_spec_.family != ModelFamily::Qwen3ForceAligner || !audio_encoder_) {
            set_error(EngineErrorCode::RuntimeError, "ForceAligner backend not initialized");
            return {};
        }
        if (word_list.empty()) {
            set_error(EngineErrorCode::RuntimeError, "word_list is empty");
            return {};
        }

        // 1. Resample to 16kHz
        std::vector<float> mono_16k;
        {
            std::vector<float> mono = audio_samples;
            if (sample_rate != 16000) {
                double ratio = static_cast<double>(sample_rate) / 16000.0;
                size_t out_size = static_cast<size_t>(std::round(mono.size() / ratio));
                mono_16k.resize(out_size);
                for (size_t i = 0; i < out_size; ++i) {
                    double t = i * ratio;
                    int idx = static_cast<int>(std::floor(t));
                    double f = t - idx;
                    auto get_s = [&](int ii) -> float {
                        if (ii < 0) return mono[0];
                        if (ii >= static_cast<int>(mono.size())) return mono[mono.size() - 1];
                        return mono[ii];
                    };
                    float y0 = get_s(idx - 1), y1 = get_s(idx);
                    float y2 = get_s(idx + 1), y3 = get_s(idx + 2);
                    float a0 = -0.5f*y0 + 1.5f*y1 - 1.5f*y2 + 0.5f*y3;
                    float a1 = y0 - 2.5f*y1 + 2.0f*y2 - 0.5f*y3;
                    float a2 = -0.5f*y0 + 0.5f*y2;
                    mono_16k[i] = static_cast<float>(((a0 * f + a1) * f + a2) * f + y1);
                }
            } else {
                mono_16k = std::move(mono);
            }
        }
        int audio_n_samples = static_cast<int>(mono_16k.size());

        // 2. Compute Mel spectrogram
        MelSpectrogram mel;
        std::string prep_err;
        if (!compute_mel_spectrogram(mono_16k, mel, &prep_err)) {
            set_error(EngineErrorCode::RuntimeError, "Mel spectrogram failed: " + prep_err);
            return {};
        }

        // 3. Audio encoder forward
        int mel_padded_frames = mel.n_frames;
        int mel_actual_frames = mel.actual_frames;
        int nM = mel.n_mels;
        int od = model_spec_.audio.output_dim;

        std::vector<bf16> mel_bf16(static_cast<size_t>(mel_padded_frames) * nM);
        for (int f = 0; f < mel_padded_frames; ++f)
            for (int m = 0; m < nM; ++m)
                mel_bf16[m * mel_padded_frames + f] = bf16(mel.data[f * nM + m]);
        Tensor mel_device = Tensor::allocate(*ctx_, {1, nM, mel_padded_frames});
        ctx_->memcpy_h2d(mel_device.data(), mel_bf16.data(), mel_bf16.size() * sizeof(bf16));

        int audio_len = 0;
        int max_audio_len = ((mel_actual_frames + 99) / 100) * 13 + 32;
        Tensor af_tmp = Tensor::allocate(*ctx_, {max_audio_len, od});
        std::string enc_error;
        if (!audio_encoder_->encode(*ctx_, mel_device, mel_actual_frames,
                                     af_tmp, audio_len, &enc_error)) {
            set_error(EngineErrorCode::RuntimeError, "Audio encoding failed: " + enc_error);
            return {};
        }
        Tensor audio_features = Tensor::allocate(*ctx_, {audio_len, od});
        {
            bf16* af_dst = audio_features.data_as<bf16>();
            bf16* af_src = af_tmp.data_as<bf16>();
            ctx_->queue().memcpy(af_dst, af_src, static_cast<size_t>(audio_len) * od * sizeof(bf16));
        }
        std::vector<bf16> audio_bf16(static_cast<size_t>(audio_len) * od);
        ctx_->memcpy_d2h(audio_bf16.data(), audio_features.data(), audio_bf16.size() * sizeof(bf16));
        ctx_->synchronize();

        // 4. Build prompt token IDs from word list
        int audio_start_id = model_spec_.audio_start_token_id;
        int audio_end_id   = model_spec_.audio_end_token_id;
        int audio_pad_id   = model_spec_.audio_token_id;
        int ts_token_id    = model_spec_.timestamp_token_id;

        std::vector<int> prompt_ids;
        prompt_ids.push_back(audio_start_id);
        for (int i = 0; i < audio_len; ++i) prompt_ids.push_back(audio_pad_id);
        prompt_ids.push_back(audio_end_id);
        for (size_t wi = 0; wi < word_list.size(); ++wi) {
            auto word_ids = tokenizer_.encode(word_list[wi]);
            prompt_ids.insert(prompt_ids.end(), word_ids.begin(), word_ids.end());
            prompt_ids.push_back(ts_token_id);
            prompt_ids.push_back(ts_token_id);
        }

        std::vector<int> pad_positions;
        for (size_t i = 0; i < prompt_ids.size(); ++i)
            if (prompt_ids[i] == audio_pad_id) pad_positions.push_back(static_cast<int>(i));

        // 5. Run ForceAligner forward
        backend_->reset();
        cached_ids_.clear();
        backend_->set_embedding_overrides(pad_positions, audio_bf16, od);
        if (!aila::env::read_flag("AILA_NO_MROPE", false)) {
            int total_len = static_cast<int>(prompt_ids.size());
            std::vector<int> pos_t(total_len), pos_h(total_len), pos_w(total_len);
            for (int i = 0; i < total_len; ++i) {
                pos_t[i] = i; pos_h[i] = i; pos_w[i] = i;
            }
            backend_->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, 0);
        }

        int* device_ids = static_cast<int*>(ctx_->alloc_device(prompt_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d_async(device_ids, prompt_ids.data(), prompt_ids.size() * sizeof(int));
        int prompt_len = static_cast<int>(prompt_ids.size());

        // Dispatch to bf16 or NF4 backend
        auto* fa_bf16 = dynamic_cast<Qwen3ForceAlignerBackend*>(backend_.get());
        auto* fa_nf4  = dynamic_cast<Qwen3ForceAlignerBnb4Backend*>(backend_.get());
        Tensor* logits_all;
        int classify_num;
        if (fa_bf16) {
            logits_all = &fa_bf16->forward_all(*ctx_, device_ids, prompt_len);
            classify_num = fa_bf16->classify_num();
        } else {
            logits_all = &fa_nf4->forward_all(*ctx_, device_ids, prompt_len);
            classify_num = fa_nf4->classify_num();
        }
        ctx_->free_device(device_ids);
        if (backend_->supports_vision_embedding_override())
            backend_->clear_mrope_positions();

        // 6. Download + post-process
        size_t total_elts = static_cast<size_t>(prompt_len) * classify_num;
        std::vector<bf16> host_bf16(total_elts);
        ctx_->memcpy_d2h(host_bf16.data(), logits_all->data(), total_elts * sizeof(bf16));
        ctx_->synchronize();
        std::vector<float> host_logits(total_elts);
        for (size_t i = 0; i < total_elts; ++i)
            host_logits[i] = static_cast<float>(host_bf16[i]);

        auto raw_ts = aila::ForceAlignerPostProcess::extract_timestamps(
            host_logits.data(), prompt_len, classify_num,
            prompt_ids.data(), prompt_len,
            ts_token_id, model_spec_.timestamp_segment_time);
        auto fixed_ts = aila::ForceAlignerPostProcess::fix_timestamp(raw_ts);
        return aila::ForceAlignerPostProcess::build_output(word_list, fixed_ts);
    }

    // ASR transcription from WAV file
    std::string transcribe(const std::string& wav_path,
                           const GenerationConfig& gen_config = GenerationConfig(),
                           std::string* language_out = nullptr,
                           const std::string& forced_language = "",
                           const std::string& system_prompt = "",
                           float segment_sec = 0.0f,
                           bool past_text_conditioning = false,
                           std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        last_transcribe_duration_s_ = 0.0;
        last_transcribe_latency_ms_ = 0.0;
        last_transcribe_tokens_ = 0;
        if (reject_yolo_capability("ASR")) return "";
        auto t_start = std::chrono::high_resolution_clock::now();

        if (model_spec_.family != ModelFamily::Qwen3ASR || !audio_encoder_) {
            set_error(EngineErrorCode::RuntimeError, "ASR backend not initialized");
            return "";
        }

        // 1. Audio preprocessing: load wav, convert to mono and resample to 16kHz
        AILA_LOG_INFO("[Transcribe] Loading audio: %s", wav_path.c_str());
        AudioBuffer audio;
        std::string preprocess_error;
        if (!load_audio(wav_path, audio, &preprocess_error)) {
            set_error(EngineErrorCode::RuntimeError, "Audio loading failed: " + preprocess_error);
            return "";
        }

        std::vector<float> mono;
        if (audio.channels > 1) {
            mono.resize(audio.samples.size() / audio.channels);
            for (size_t i = 0; i < mono.size(); ++i) {
                float sum = 0.0f;
                for (int c = 0; c < audio.channels; ++c)
                    sum += audio.samples[i * audio.channels + c];
                mono[i] = sum / static_cast<float>(audio.channels);
            }
        } else {
            mono = std::move(audio.samples);
        }

        std::vector<float> mono_16k;
        resample_to_16k(mono, audio.sample_rate, mono_16k);
        int audio_n_samples = static_cast<int>(mono_16k.size());
        AILA_LOG_INFO("[Transcribe] Mono 16kHz audio: %d samples (%.1f seconds)", audio_n_samples, static_cast<float>(audio_n_samples) / 16000.0f);

        // 2. Segment detection
        float search_sec = 3.0f;
        if (search_sec > segment_sec / 2.0f) search_sec = segment_sec / 2.0f;
        int target_samples = static_cast<int>(segment_sec * 16000);
        int margin_samples = static_cast<int>(search_sec * 16000);

        std::vector<int> splits;
        splits.push_back(0);
        if (segment_sec <= 0.0f || audio_n_samples <= target_samples + margin_samples) {
            splits.push_back(audio_n_samples);
        } else {
            int pos = 0;
            while (pos + target_samples + margin_samples < audio_n_samples) {
                int split = find_split_point(mono_16k.data(), audio_n_samples, pos + target_samples, search_sec);
                splits.push_back(split);
                pos = split;
            }
            splits.push_back(audio_n_samples);
        }
        AILA_LOG_INFO("[Transcribe] Splitting audio into %zu segment(s)", splits.size() - 1);

        std::string accumulated_result = "";
        std::string first_seg_lang = "";
        int min_samples = 8000; // 0.5s minimum padding

        for (size_t s = 0; s < splits.size() - 1; ++s) {
            int seg_start = splits[s];
            int seg_end = splits[s + 1];
            int seg_samples = seg_end - seg_start;

            std::vector<float> seg_buf(mono_16k.begin() + seg_start, mono_16k.begin() + seg_end);
            if (seg_samples < min_samples) {
                seg_buf.resize(min_samples, 0.0f);
            }

            std::string seg_lang;
            std::string seg_text = transcribe_segment_raw(
                seg_buf,
                gen_config,
                forced_language,
                system_prompt,
                past_text_conditioning ? accumulated_result : "",
                &seg_lang
            );

            if (first_seg_lang.empty()) {
                first_seg_lang = seg_lang;
            }

            // Append segment result to accumulated result
            if (!seg_text.empty()) {
                bool need_space = false;
                if (!accumulated_result.empty()) {
                    int prev_ch = static_cast<unsigned char>(accumulated_result.back());
                    int next_ch = static_cast<unsigned char>(seg_text.front());
                    if (should_insert_boundary_space(prev_ch, next_ch)) {
                        need_space = true;
                    }
                }

                if (need_space) {
                    accumulated_result += " ";
                    if (token_callback) {
                        token_callback(" ");
                    }
                }
                accumulated_result += seg_text;
                if (token_callback) {
                    token_callback(seg_text);
                }
            }
        }

        if (language_out) {
            *language_out = first_seg_lang;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        last_transcribe_latency_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        last_transcribe_duration_s_ = static_cast<double>(audio_n_samples) / 16000.0;

        return accumulated_result;
    }

    // Real-time ASR transcribe streaming state machine
    class TranscribeStream {
    public:
        TranscribeStream(
            InferenceEngine* engine,
            const GenerationConfig& gen_config,
            const std::string& forced_language,
            const std::string& system_prompt
        ) : engine_(engine),
            gen_config_(gen_config),
            forced_language_(forced_language),
            system_prompt_(system_prompt),
            stable_samples_offset_(0)
        {}

        void feed_audio(const float* data, size_t count) {
            if (!data || count == 0) return;
            audio_buffer_.insert(audio_buffer_.end(), data, data + count);
        }

        void process() {
            if (!engine_) return;

            // 16kHz sampling rate
            size_t min_stable_samples = static_cast<size_t>(6.0f * 16000); // 6s threshold
            size_t target_chunk_samples = static_cast<size_t>(5.0f * 16000); // 5s target
            float search_sec = 1.0f; // 1s search window

            while (true) {
                size_t available = audio_buffer_.size() - stable_samples_offset_;
                if (available < min_stable_samples) {
                    break;
                }

                int target_relative = static_cast<int>(target_chunk_samples);
                int split_relative = InferenceEngine::find_split_point(
                    audio_buffer_.data() + stable_samples_offset_,
                    static_cast<int>(available),
                    target_relative,
                    search_sec
                );

                size_t split_absolute = stable_samples_offset_ + static_cast<size_t>(split_relative);

                // Decode stable chunk: [stable_samples_offset_, split_absolute]
                std::vector<float> seg_buf(
                    audio_buffer_.begin() + stable_samples_offset_,
                    audio_buffer_.begin() + split_absolute
                );
                size_t min_samples = 8000;
                if (seg_buf.size() < min_samples) {
                    seg_buf.resize(min_samples, 0.0f);
                }

                std::string seg_lang;
                std::string seg_text = engine_->transcribe_segment_raw(
                    seg_buf,
                    gen_config_,
                    forced_language_,
                    system_prompt_,
                    past_text_,
                    &seg_lang
                );

                if (!seg_text.empty()) {
                    bool need_space = false;
                    if (!stable_text_.empty()) {
                        int prev_ch = static_cast<unsigned char>(stable_text_.back());
                        int next_ch = static_cast<unsigned char>(seg_text.front());
                        if (InferenceEngine::should_insert_boundary_space(prev_ch, next_ch)) {
                            need_space = true;
                        }
                    }
                    if (need_space) {
                        stable_text_ += " ";
                    }
                    stable_text_ += seg_text;
                    past_text_ = seg_text;
                }

                stable_samples_offset_ = split_absolute;
            }

            // Decode partial chunk (remaining audio)
            size_t remaining = audio_buffer_.size() - stable_samples_offset_;
            size_t min_decode_samples = static_cast<size_t>(0.5f * 16000); // 0.5s minimum
            if (remaining >= min_decode_samples) {
                std::vector<float> seg_buf(
                    audio_buffer_.begin() + stable_samples_offset_,
                    audio_buffer_.end()
                );
                size_t min_samples = 8000;
                if (seg_buf.size() < min_samples) {
                    seg_buf.resize(min_samples, 0.0f);
                }

                partial_text_ = engine_->transcribe_segment_raw(
                    seg_buf,
                    gen_config_,
                    forced_language_,
                    system_prompt_,
                    past_text_
                );
            } else {
                partial_text_ = "";
            }
        }

        void get_text(std::string& out_stable, std::string& out_partial) {
            process();
            out_stable = stable_text_;
            out_partial = partial_text_;
        }

    private:
        InferenceEngine* engine_;
        GenerationConfig gen_config_;
        std::string forced_language_;
        std::string system_prompt_;

        std::vector<float> audio_buffer_;
        size_t stable_samples_offset_;

        std::string stable_text_;
        std::string partial_text_;
        std::string past_text_;
    };

    double benchmark_prefill(const std::vector<int>& token_ids) {
        if (backend_) backend_->reset();
        cached_ids_.clear();
        benchmark_seed_ready_ = false;

        int* device_ids = static_cast<int*>(ctx_->alloc_device(token_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d(device_ids, token_ids.data(), token_ids.size() * sizeof(int));

        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor& logits = backend_->forward(*ctx_, device_ids, static_cast<int>(token_ids.size()));
        ctx_->synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // Prepare a valid decode seed token from prefill logits
        int* bench_seed_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        ops::argmax(*ctx_, logits, config_.vocab_size, bench_seed_device);
        ctx_->memcpy_d2h(&benchmark_seed_token_, bench_seed_device, sizeof(int));
        ctx_->free_device(bench_seed_device);
        benchmark_seed_ready_ = true;

        ctx_->free_device(device_ids);
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Raw decode N tokens for benchmark (after prefill)
    double benchmark_decode(int num_tokens, const GenerationConfig* decode_config = nullptr) {
        if (num_tokens <= 0) {
            return 0.0;
        }
        if (!benchmark_seed_ready_) {
            AILA_LOG_WARN("[Bench] benchmark_decode called without a valid prefill seed, using BOS token as fallback");
            benchmark_seed_token_ = config_.bos_token_id;
            if (benchmark_seed_token_ < 0 || benchmark_seed_token_ >= config_.vocab_size) {
                benchmark_seed_token_ = std::max(0, config_.eos_token_id);
            }
        }

        GenerationConfig bench_gen_cfg;
        bench_gen_cfg.do_sample = false;
        if (decode_config) {
            bench_gen_cfg = *decode_config;
        }
        if (bench_gen_cfg.do_sample && bench_gen_cfg.use_fixed_seed) {
            ops::set_sampling_seed(bench_gen_cfg.sampling_seed);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        bool can_use_device_sample = ops::can_use_device_sampling(config_.vocab_size, bench_gen_cfg);
        bool use_fast_device_chain = (!bench_gen_cfg.has_penalties()) &&
                                     (!bench_gen_cfg.do_sample || can_use_device_sample);
        if (use_fast_device_chain) {
            int* token_chain_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(num_tokens + 1) * sizeof(int)));
            ctx_->memcpy_h2d(token_chain_device, &benchmark_seed_token_, sizeof(int));

            for (int i = 0; i < num_tokens; i++) {
                Tensor& logits = backend_->forward(*ctx_, token_chain_device + i, 1);
                if (!bench_gen_cfg.do_sample) {
                    ops::argmax(*ctx_, logits, config_.vocab_size, token_chain_device + i + 1);
                } else {
                    ops::sample_with_config_device(*ctx_, logits, config_.vocab_size,
                                                   bench_gen_cfg, ops::next_sampling_uniform(),
                                                   token_chain_device + i + 1);
                }
            }
            ctx_->synchronize();
            auto t1 = std::chrono::high_resolution_clock::now();

            ctx_->memcpy_d2h(&benchmark_seed_token_, token_chain_device + num_tokens, sizeof(int));
            benchmark_seed_ready_ = true;
            ctx_->free_device(token_chain_device);
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        ctx_->memcpy_h2d(current_token_device, &benchmark_seed_token_, sizeof(int));

        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(num_tokens));

        for (int i = 0; i < num_tokens; i++) {
            Tensor& logits = backend_->forward(*ctx_, current_token_device, 1);
            if (!bench_gen_cfg.do_sample) {
                ops::argmax(*ctx_, logits, config_.vocab_size, next_token_device);
            } else {
                int sampled = ops::sample_with_config(*ctx_, logits, config_.vocab_size,
                                                      bench_gen_cfg, generated_token_ids);
                generated_token_ids.push_back(sampled);
                ctx_->memcpy_h2d(next_token_device, &sampled, sizeof(int));
            }
            std::swap(current_token_device, next_token_device);
        }
        ctx_->synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // Keep seed token deterministic for any subsequent decode benchmark calls.
        ctx_->memcpy_d2h(&benchmark_seed_token_, current_token_device, sizeof(int));
        benchmark_seed_ready_ = true;

        ctx_->free_device(current_token_device);
        ctx_->free_device(next_token_device);
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::vector<Detection> detect_file(
            const std::string& path,
            const DetectionConfig& detection_config = DetectionConfig()) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_file(path, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

    std::vector<Detection> detect_encoded(
            const uint8_t* data, size_t size,
            const DetectionConfig& detection_config = DetectionConfig()) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_encoded(data, size, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

    std::vector<Detection> detect_pixels(
            const ImageView& image,
            const DetectionConfig& detection_config = DetectionConfig()) {
        clear_error();
        std::vector<Detection> detections;
        if (!detector_) {
            set_error(EngineErrorCode::ModelCapability,
                      "Object detection requires a YOLO26 model");
            return detections;
        }
        std::string error;
        try {
            if (detector_->detect_pixels(image, detection_config, detections, &error)) return detections;
            set_error(EngineErrorCode::InvalidArgument, error);
            detections.clear();
        } catch (const std::exception& exception) {
            set_error(EngineErrorCode::RuntimeError, exception.what());
            detections.clear();
        }
        return detections;
    }

    int last_detect_image_width() const {
        return detector_ ? detector_->last_image_width() : 0;
    }
    int last_detect_image_height() const {
        return detector_ ? detector_->last_image_height() : 0;
    }
    double last_detect_preprocess_ms() const {
        return detector_ ? detector_->last_preprocess_ms() : 0.0;
    }
    double last_detect_device_wall_ms() const {
        return detector_ ? detector_->last_device_wall_ms() : 0.0;
    }
    double last_detect_postprocess_ms() const {
        return detector_ ? detector_->last_postprocess_ms() : 0.0;
    }
    size_t last_detect_peak_device_bytes() const {
        return detector_ && ctx_ ? ctx_->peak_allocated_bytes() : 0;
    }

    Qwen3Config& config() { return config_; }
    const Qwen3Config& config() const { return config_; }
    const ModelSpec& model_spec() const { return model_spec_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    bool vision_enabled() const { return vision_backend_enabled_; }
    void clear_error() {
        last_error_code_ = EngineErrorCode::Ok;
        last_error_message_.clear();
    }

    bool reject_yolo_capability(const char* operation) {
        if (model_spec_.family != ModelFamily::Yolo26) return false;
        set_error(EngineErrorCode::ModelCapability,
                  std::string(operation) + " is not supported by a YOLO26 detection model");
        return true;
    }

    void set_error(EngineErrorCode code, const std::string& message) {
        last_error_code_ = code;
        last_error_message_ = message;
    }

    EngineErrorCode last_error_code() const { return last_error_code_; }
    const std::string& last_error_message() const { return last_error_message_; }
    double last_transcribe_duration_s() const { return last_transcribe_duration_s_; }
    double last_transcribe_latency_ms() const { return last_transcribe_latency_ms_; }
    int last_transcribe_tokens() const { return last_transcribe_tokens_; }
    const std::string& model_dir() const { return model_dir_; }

    // Speaker embedding cache: avoid re-extracting the same reference audio.
    // Set via AILA_REF_CACHE_DIR env var or --ref-cache-dir CLI argument.
    void setRefCacheDir(const std::string& dir) { ref_cache_dir_ = dir; }
    const std::string& refCacheDir() const { return ref_cache_dir_; }

    // Look up speaker token ID from spk_id map (CustomVoice).
    // Returns 0 if name is empty or not found.
    int getSpeakerId(const std::string& name) const {
        if (name.empty()) return 0;
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        auto it = model_spec_.qwen3.spk_id.find(lower);
        if (it != model_spec_.qwen3.spk_id.end()) {
            return it->second;
        }
        return 0;
    }

    // Look up language codec token ID from codec_language_id map.
    // Returns 0 if lang is empty or not found (0 = auto/nothink).
    int getLanguageId(const std::string& lang) const {
        if (lang.empty()) return 0;
        std::string lower = lang;
        std::transform(lower.begin(), lower.end(), lower.begin(),
            [](unsigned char c) { return std::tolower(c); });
        auto it = model_spec_.qwen3.codec_language_id.find(lower);
        if (it != model_spec_.qwen3.codec_language_id.end()) {
            return it->second;
        }
        return 0;
    }

    // Clear the in-memory speaker embedding cache.
    void clearRefCache() { ref_cache_.clear(); }

    // Build a cache key that includes the embedding dim so different models
    // (e.g. 0.6B=1024-dim vs 1.7B=2048-dim) don't share incompatible cache files.
    static std::string refCacheKey(const std::string& audio_path, int dim) {
        return audio_path + ":dim" + std::to_string(dim);
    }

    // Build disk cache path from audio path and embedding dimension.
    std::string refDiskCachePath(const std::string& audio_path, int dim) const {
        std::string suffix = ".dim" + std::to_string(dim) + ".ref.bin";
        if (!ref_cache_dir_.empty()) {
            size_t sep = audio_path.find_last_of("/\\");
            std::string base = (sep != std::string::npos) ? audio_path.substr(sep + 1) : audio_path;
            return ref_cache_dir_ + "/" + base + suffix;
        }
        return audio_path + suffix;
    }

    // Convenience: try common embedding dimensions (1024 for 0.6B, 2048 for 1.7B).
    bool lookupRefCache(const std::string& audio_path,
                            std::vector<float>& embedding) {
        return lookupRefCache(audio_path, 1024, embedding)
            || lookupRefCache(audio_path, 2048, embedding);
    }

    // Check if a speaker embedding is cached for a specific embedding dimension.
    // The 'dim_hint' is used to construct the cache key; pass the expected
    // embedding dimension (1024 for 0.6B, 2048 for 1.7B).
    bool lookupRefCache(const std::string& audio_path, int dim_hint,
                            std::vector<float>& embedding) {
        if (dim_hint <= 0) return false;
        std::string key = refCacheKey(audio_path, dim_hint);
        auto mem_it = ref_cache_.find(key);
        if (mem_it != ref_cache_.end()) {
            embedding = mem_it->second;
            return true;
        }
        std::string disk_path = refDiskCachePath(audio_path, dim_hint);
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            in.seekg(0, std::ios::end);
            size_t sz = in.tellg();
            in.seekg(0, std::ios::beg);
            if (sz >= 4 && (sz - 4) % sizeof(float) == 0) {
                int32_t hdr = 0;
                in.read(reinterpret_cast<char*>(&hdr), 4);
                int dim = static_cast<int>(hdr);
                if (dim == dim_hint && static_cast<size_t>(dim) * sizeof(float) == sz - 4) {
                    embedding.resize(dim);
                    in.read(reinterpret_cast<char*>(embedding.data()), dim * sizeof(float));
                    float norm_sq = 0.0f;
                    for (float v : embedding) norm_sq += v * v;
                    if (norm_sq > 1.0f) {
                        ref_cache_[key] = embedding;
                        return true;
                    }
                    embedding.clear();
                }
            }
        }
        return false;
    }

    // Cache an externally-extracted speaker embedding (in-memory + disk).
    void cacheRefEmbedding(const std::string& audio_path,
                               const std::vector<float>& embedding) {
        if (embedding.empty()) return;
        int dim = static_cast<int>(embedding.size());
        std::string key = refCacheKey(audio_path, dim);
        ref_cache_[key] = embedding;
        std::string disk_path = refDiskCachePath(audio_path, dim);
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(disk_path).parent_path(), ec);
        std::ofstream out(disk_path, std::ios::binary);
        if (out.is_open()) {
            int32_t hdr = dim;
            out.write(reinterpret_cast<const char*>(&hdr), 4);
            out.write(reinterpret_cast<const char*>(embedding.data()),
                     embedding.size() * sizeof(float));
        }
    }

    // Build disk cache path for 12Hz reference audio codes
    std::string refCodesDiskCachePath(const std::string& audio_path) const {
        std::string suffix = ".qwen3tts12hz.codes.bin";
        if (!ref_cache_dir_.empty()) {
            size_t sep = audio_path.find_last_of("/\\");
            std::string base = (sep != std::string::npos) ? audio_path.substr(sep + 1) : audio_path;
            return ref_cache_dir_ + "/" + base + suffix;
        }
        return audio_path + suffix;
    }

    bool lookupRefCodesCache(const std::string& audio_path, TTSReferenceCodes& outCodes) {
        auto mem_it = ref_codes_cache_.find(audio_path);
        if (mem_it != ref_codes_cache_.end()) {
            outCodes = mem_it->second;
            return true;
        }
        std::string disk_path = refCodesDiskCachePath(audio_path);
        std::ifstream in(disk_path, std::ios::binary);
        if (in.is_open()) {
            char magic[4];
            in.read(magic, 4);
            if (std::memcmp(magic, "MIMI", 4) == 0) {
                int32_t version = 0;
                int32_t frames = 0;
                int32_t codebooks = 0;
                in.read(reinterpret_cast<char*>(&version), 4);
                in.read(reinterpret_cast<char*>(&frames), 4);
                in.read(reinterpret_cast<char*>(&codebooks), 4);
                if (version == 1 && frames > 0 && codebooks == 16) {
                    outCodes.frames = frames;
                    outCodes.codebooks = codebooks;
                    outCodes.codes.resize(static_cast<size_t>(frames) * 16);
                    in.read(reinterpret_cast<char*>(outCodes.codes.data()),
                            outCodes.codes.size() * sizeof(int32_t));
                    ref_codes_cache_[audio_path] = outCodes;
                    return true;
                }
            }
        }
        return false;
    }

    void cacheRefCodes(const std::string& audio_path, const TTSReferenceCodes& codes) {
        if (codes.frames <= 0 || codes.codebooks != 16) return;
        ref_codes_cache_[audio_path] = codes;
        std::string disk_path = refCodesDiskCachePath(audio_path);
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(disk_path).parent_path(), ec);
        std::ofstream out(disk_path, std::ios::binary);
        if (out.is_open()) {
            const char magic[4] = {'M', 'I', 'M', 'I'};
            int32_t version = 1;
            int32_t frames = codes.frames;
            int32_t codebooks = 16;
            out.write(magic, 4);
            out.write(reinterpret_cast<const char*>(&version), 4);
            out.write(reinterpret_cast<const char*>(&frames), 4);
            out.write(reinterpret_cast<const char*>(&codebooks), 4);
            out.write(reinterpret_cast<const char*>(codes.codes.data()),
                      codes.codes.size() * sizeof(int32_t));
        }
    }

private:
    // In-memory speaker embedding cache (keyed by audio file path).
    std::unordered_map<std::string, std::vector<float>> ref_cache_;
    // In-memory reference codes cache (keyed by audio file path).
    std::unordered_map<std::string, TTSReferenceCodes> ref_codes_cache_;
    // Optional persistent cache directory for speaker embeddings.
    std::string ref_cache_dir_;

    std::string model_dir_;
    std::string lora_dir_;
    std::string system_prompt_ = "You are a helpful assistant.";
    Qwen3Config config_;
    ModelSpec model_spec_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<IModelBackend> backend_;
    std::unique_ptr<aila::vision::Yolo26Detector> detector_;
    std::unique_ptr<aila::vision::Qwen35VisionEncoder> vision_encoder_;
    std::unique_ptr<aila::audio::Qwen3ASRAudioEncoder> audio_encoder_;
    std::unique_ptr<aila::audio::MimiEncoder> mimi_encoder_;
    aila::chat::ChatFormatter chat_formatter_;
    Tokenizer tokenizer_;
    bool vision_backend_enabled_ = false;

    // Multi-turn conversation state
    ChatHistory history_;
    std::vector<Message> mm_history_;
    aila::chat::ChatSessionState chat_session_;
    // Exact token IDs currently stored in the KV cache.
    // This is the ground truth for incremental prefill.
    std::vector<int> cached_ids_;

    // Benchmark decode seed (token after prefill argmax)
    int benchmark_seed_token_ = -1;
    bool benchmark_seed_ready_ = false;
    EngineErrorCode last_error_code_ = EngineErrorCode::Ok;
    std::string last_error_message_;
    std::string last_generation_finish_reason_ = "stop";
    std::string last_generation_template_name_;
    bool last_generation_forced_think_close_ = false;
    bool last_generation_think_close_truncated_ = false;
    bool generation_abort_requested_ = false;
    double last_transcribe_duration_s_ = 0.0;
    double last_transcribe_latency_ms_ = 0.0;
    int last_transcribe_tokens_ = 0;
};
