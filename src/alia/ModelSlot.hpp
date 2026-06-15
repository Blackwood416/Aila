#pragma once

#include "../core/Context.hpp"
#include "../utils/ModelSpec.hpp"

#include <memory>
#include <string>

class IModelBackend;
class ModelWeights;
class Tokenizer;

namespace aila::audio {
class Qwen3ASRAudioEncoder;
}

namespace aila::vision {
class Qwen35VisionEncoder;
}

namespace aila::alia {

enum class ModelRole {
    Unknown,
    Asr,
    ForegroundVlm,
    BackgroundVlm,
    Tts
};

enum class ModelSlotState {
    Empty,
    Configured,
    MetadataLoaded,
    Loaded,
    Failed
};

enum class BackendKind {
    Unknown,
    Qwen3AsrDense,
    Qwen3AsrBnb4,
    Qwen3ForceAlignerDense,
    Qwen3ForceAlignerBnb4,
    Qwen35HybridBnb4,
    Qwen3Tts
};

class ModelSlot {
public:
    ModelSlot();
    ~ModelSlot();

    void configure(ModelRole role, std::string model_dir, Context* context);
    void set_lora_dir(std::string lora_dir);
    bool load_metadata();
    bool load_model(int max_seq_len);
    void configure_loaded_for_tests(ModelRole role,
                                    Context* context,
                                    std::unique_ptr<Tokenizer> tokenizer,
                                    std::unique_ptr<IModelBackend> backend,
                                    BackendKind backend_kind);

    ModelRole role() const { return role_; }
    ModelSlotState state() const { return state_; }
    const std::string& model_dir() const { return model_dir_; }
    Context* context() const { return context_; }
    const ModelSpec& model_spec() const { return model_spec_; }
    BackendKind backend_kind() const { return backend_kind_; }
    Tokenizer* tokenizer() const { return tokenizer_.get(); }
    ModelWeights* weights() const { return weights_.get(); }
    IModelBackend* backend() const { return backend_.get(); }
    aila::audio::Qwen3ASRAudioEncoder* audio_encoder() const { return audio_encoder_.get(); }
    aila::vision::Qwen35VisionEncoder* vision_encoder() const { return vision_encoder_.get(); }
    bool vision_enabled() const { return vision_enabled_; }
    const std::string& lora_dir() const { return lora_dir_; }
    bool lora_applied() const { return lora_applied_; }
    size_t lora_pair_count() const { return lora_pair_count_; }
    const std::string& last_error() const { return last_error_; }

private:
    bool family_matches_role() const;
    bool select_backend_kind();
    bool validate_quantization();
    bool apply_lora_adapter();
    bool create_backend();
    void clear_loaded_objects();
    bool fail_load(std::string message);
    const char* role_name() const;

    ModelRole role_ = ModelRole::Unknown;
    ModelSlotState state_ = ModelSlotState::Empty;
    std::string model_dir_;
    std::string lora_dir_;
    Context* context_ = nullptr;
    ModelSpec model_spec_{};
    BackendKind backend_kind_ = BackendKind::Unknown;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<IModelBackend> backend_;
    std::unique_ptr<aila::audio::Qwen3ASRAudioEncoder> audio_encoder_;
    std::unique_ptr<aila::vision::Qwen35VisionEncoder> vision_encoder_;
    bool vision_enabled_ = false;
    bool lora_applied_ = false;
    size_t lora_pair_count_ = 0;
    std::string last_error_;
};

}  // namespace aila::alia
