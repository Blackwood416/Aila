#include "ModelSlot.hpp"

#include "../audio/Qwen3ASRAudioEncoder.hpp"
#include "../models/Qwen35HybridBnb4Backend.hpp"
#include "../models/Qwen3ASRBackend.hpp"
#include "../models/Qwen3ASRBnb4Backend.hpp"
#include "../models/Qwen3ForceAlignerBackend.hpp"
#include "../models/Qwen3ForceAlignerBnb4Backend.hpp"
#include "../models/Qwen3TTSBackend.hpp"
#include "../utils/EnvUtils.hpp"
#include "../utils/SafeTensors.hpp"
#include "../utils/Tokenizer.hpp"
#include "../vision/Qwen35VisionEncoder.hpp"

#include <exception>
#include <utility>

namespace aila::alia {

ModelSlot::ModelSlot() = default;
ModelSlot::~ModelSlot() = default;

void ModelSlot::configure(ModelRole role, std::string model_dir, Context* context) {
    clear_loaded_objects();
    role_ = role;
    model_dir_ = std::move(model_dir);
    context_ = context;
    backend_kind_ = BackendKind::Unknown;
    last_error_.clear();
    state_ = ModelSlotState::Configured;
}

bool ModelSlot::load_metadata() {
    if (state_ == ModelSlotState::Empty || role_ == ModelRole::Unknown || !context_) {
        last_error_ = "model slot is not configured";
        state_ = ModelSlotState::Failed;
        return false;
    }

    if (model_dir_.empty()) {
        last_error_.clear();
        return true;
    }

    std::string load_error;
    if (!aila::modelspec::load_from_dir(model_dir_, model_spec_, &load_error)) {
        last_error_ = load_error;
        state_ = ModelSlotState::Failed;
        return false;
    }

    if (!family_matches_role()) {
        last_error_ = std::string(role_name()) + " slot received incompatible model_type '" +
                      model_spec_.model_type + "'";
        state_ = ModelSlotState::Failed;
        return false;
    }

    if (!select_backend_kind()) {
        state_ = ModelSlotState::Failed;
        return false;
    }

    last_error_.clear();
    state_ = ModelSlotState::MetadataLoaded;
    return true;
}

bool ModelSlot::load_model(int max_seq_len) {
    clear_loaded_objects();

    if (max_seq_len <= 0) {
        return fail_load("max_seq_len must be positive");
    }

    if (model_dir_.empty()) {
        last_error_.clear();
        return true;
    }

    if (state_ != ModelSlotState::MetadataLoaded && !load_metadata()) {
        return false;
    }

    if (!validate_quantization()) {
        return fail_load(last_error_);
    }

    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load(model_dir_)) {
        return fail_load(std::string(role_name()) + " tokenizer load failed: " + model_dir_);
    }

    try {
        weights_ = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(model_dir_, *context_));
    } catch (const std::exception& e) {
        return fail_load(std::string(role_name()) + " weights load failed: " + e.what());
    } catch (...) {
        return fail_load(std::string(role_name()) + " weights load failed");
    }

    if (!create_backend()) {
        return false;
    }

    std::string backend_error;
    if (!backend_->load(*context_, *weights_, model_spec_, max_seq_len, &backend_error)) {
        return fail_load(std::string(role_name()) + " backend load failed: " + backend_error);
    }

    if (backend_kind_ == BackendKind::Qwen3Tts) {
        auto* tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
        if (tts_backend) {
            std::string mimi_error;
            if (!tts_backend->load_mimi_vocoder(*context_, model_dir_, &mimi_error)) {
                return fail_load(std::string(role_name()) +
                                 " Mimi vocoder load failed: " + mimi_error);
            }
        }
    }

    if (backend_kind_ == BackendKind::Qwen3AsrDense ||
        backend_kind_ == BackendKind::Qwen3AsrBnb4 ||
        backend_kind_ == BackendKind::Qwen3ForceAlignerDense ||
        backend_kind_ == BackendKind::Qwen3ForceAlignerBnb4) {
        audio_encoder_ = std::make_unique<aila::audio::Qwen3ASRAudioEncoder>();
        std::string audio_error;
        if (!audio_encoder_->load(*context_, *weights_, model_spec_.audio, &audio_error)) {
            return fail_load(std::string(role_name()) +
                             " audio encoder load failed: " + audio_error);
        }
    }

    vision_enabled_ = false;
    if (model_spec_.family == ModelFamily::Qwen35Hybrid && model_spec_.vision.enabled) {
        vision_encoder_ = std::make_unique<aila::vision::Qwen35VisionEncoder>();
        std::string vision_error;
        if (!vision_encoder_->load(*context_, *weights_, model_spec_, model_dir_, &vision_error)) {
            return fail_load(std::string(role_name()) +
                             " vision encoder load failed: " + vision_error);
        }
        vision_enabled_ = true;
    }

    backend_->reset();
    last_error_.clear();
    state_ = ModelSlotState::Loaded;
    return true;
}

void ModelSlot::configure_loaded_for_tests(
    ModelRole role,
    Context* context,
    std::unique_ptr<Tokenizer> tokenizer,
    std::unique_ptr<IModelBackend> backend,
    BackendKind backend_kind) {
    clear_loaded_objects();
    role_ = role;
    model_dir_.clear();
    context_ = context;
    backend_kind_ = backend_kind;
    tokenizer_ = std::move(tokenizer);
    backend_ = std::move(backend);
    weights_.reset();
    audio_encoder_.reset();
    vision_encoder_.reset();
    vision_enabled_ = false;
    last_error_.clear();
    state_ = (context_ && tokenizer_ && backend_)
        ? ModelSlotState::Loaded
        : ModelSlotState::Failed;
}

bool ModelSlot::family_matches_role() const {
    switch (role_) {
        case ModelRole::Asr:
            return model_spec_.family == ModelFamily::Qwen3ASR ||
                   model_spec_.family == ModelFamily::Qwen3ForceAligner;
        case ModelRole::ForegroundVlm:
        case ModelRole::BackgroundVlm:
            return model_spec_.family == ModelFamily::Qwen35Hybrid;
        case ModelRole::Tts:
            return model_spec_.family == ModelFamily::Qwen3TTS;
        case ModelRole::Unknown:
            return false;
    }
    return false;
}

bool ModelSlot::select_backend_kind() {
    backend_kind_ = BackendKind::Unknown;

    switch (model_spec_.family) {
        case ModelFamily::Qwen3ASR:
            backend_kind_ = model_spec_.is_bitsandbytes_4bit()
                ? BackendKind::Qwen3AsrBnb4
                : BackendKind::Qwen3AsrDense;
            return true;
        case ModelFamily::Qwen3ForceAligner:
            backend_kind_ = model_spec_.is_bitsandbytes_4bit()
                ? BackendKind::Qwen3ForceAlignerBnb4
                : BackendKind::Qwen3ForceAlignerDense;
            return true;
        case ModelFamily::Qwen35Hybrid:
            if (!model_spec_.is_bitsandbytes_4bit()) {
                last_error_ = std::string(role_name()) +
                              " slot requires a bitsandbytes 4-bit Qwen3.5 hybrid checkpoint";
                return false;
            }
            backend_kind_ = BackendKind::Qwen35HybridBnb4;
            return true;
        case ModelFamily::Qwen3TTS:
            backend_kind_ = BackendKind::Qwen3Tts;
            return true;
        case ModelFamily::Qwen3Dense:
        case ModelFamily::Unknown:
            last_error_ = std::string(role_name()) + " slot has no Alia backend for model_type '" +
                          model_spec_.model_type + "'";
            return false;
    }

    last_error_ = std::string(role_name()) + " slot has no Alia backend for model_type '" +
                  model_spec_.model_type + "'";
    return false;
}

bool ModelSlot::validate_quantization() {
    if (!model_spec_.is_quantized()) {
        return true;
    }

    const auto& quant = model_spec_.quantization;
    auto set_error = [&](const char* message) {
        last_error_ = std::string(role_name()) + " quantization is unsupported: " + message;
        return false;
    };

    if (!model_spec_.is_bitsandbytes_4bit()) {
        return set_error("only bitsandbytes 4-bit checkpoints are supported");
    }
    if (quant.bnb_4bit_quant_type != "nf4") {
        return set_error("only NF4 checkpoints are supported");
    }
    if (quant.bnb_4bit_quant_storage != "uint8") {
        return set_error("only uint8 packed storage is supported");
    }
    if (quant.bnb_4bit_compute_dtype != "float16") {
        return set_error("bnb_4bit_compute_dtype must be float16 on XPU");
    }
    if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
        if (!is_supported_qwen35_hybrid_text_spec(model_spec_.qwen35_text)) {
            return set_error("only the supported Qwen3.5 Hybrid specs are enabled");
        }
        if (!aila::env::read_flag("AILA_Q35_LINEAR_DELTA", true)) {
            return set_error("AILA_Q35_LINEAR_DELTA=1 is required");
        }
    }

    return true;
}

bool ModelSlot::create_backend() {
    switch (backend_kind_) {
        case BackendKind::Qwen3AsrDense:
            backend_ = std::make_unique<Qwen3ASRBackend>();
            return true;
        case BackendKind::Qwen3AsrBnb4:
            backend_ = std::make_unique<Qwen3ASRBnb4Backend>();
            return true;
        case BackendKind::Qwen3ForceAlignerDense:
            backend_ = std::make_unique<Qwen3ForceAlignerBackend>();
            return true;
        case BackendKind::Qwen3ForceAlignerBnb4:
            backend_ = std::make_unique<Qwen3ForceAlignerBnb4Backend>();
            return true;
        case BackendKind::Qwen35HybridBnb4:
            backend_ = std::make_unique<Qwen35HybridBnb4Backend>();
            return true;
        case BackendKind::Qwen3Tts:
            backend_ = std::make_unique<Qwen3TTSBackend>();
            return true;
        case BackendKind::Unknown:
            return fail_load(std::string(role_name()) + " backend kind is unknown");
    }

    return fail_load(std::string(role_name()) + " backend kind is unknown");
}

void ModelSlot::clear_loaded_objects() {
    vision_enabled_ = false;
    vision_encoder_.reset();
    audio_encoder_.reset();
    backend_.reset();
    weights_.reset();
    tokenizer_.reset();
}

bool ModelSlot::fail_load(std::string message) {
    clear_loaded_objects();
    last_error_ = std::move(message);
    state_ = ModelSlotState::Failed;
    return false;
}

const char* ModelSlot::role_name() const {
    switch (role_) {
        case ModelRole::Asr: return "ASR";
        case ModelRole::ForegroundVlm: return "foreground VLM";
        case ModelRole::BackgroundVlm: return "background VLM";
        case ModelRole::Tts: return "TTS";
        case ModelRole::Unknown: return "unknown";
    }
    return "unknown";
}

}  // namespace aila::alia
