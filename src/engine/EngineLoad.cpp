#include "engine/Engine.hpp"

bool InferenceEngine::init(const std::string& model_dir, int max_seq_len,
              const std::string& lora_dir) {
        clear_error();
        model_dir_ = model_dir;
        lora_dir_ = lora_dir;

        AILA_LOG_INFO("========================================");
        AILA_LOG_INFO("  Aila Inference Engine");
        AILA_LOG_INFO("========================================");

        // 1. Create context
        AILA_LOG_INFO("[1/3] Initializing GPU context...");
        ctx_ = std::make_unique<Context>(nullptr,aila::env::read_flag("AILA_PROFILE_Q4E_COLD_DEVICE",false));

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
            if (!quant.bnb_4bit_quant_storage.empty() &&
                quant.bnb_4bit_quant_storage != "uint8") {
                if (error_message) {
                    *error_message = "Only bitsandbytes uint8 packed storage is supported";
                }
                return false;
            }
            const bool allow_bf16_compute =
                (spec.family == ModelFamily::DiffusionGemma || spec.family == ModelFamily::Qwen4Exp) &&
                quant.bnb_4bit_compute_dtype == "bfloat16";
            if (quant.bnb_4bit_compute_dtype != "float16" &&
                !allow_bf16_compute) {
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
            if (spec.family == ModelFamily::Qwen4Exp) {
                if (!is_exact_qwen4exp_text_spec(spec.qwen4exp_text)) {
                    if (error_message) {
                        *error_message = "Qwen4Exp bitsandbytes v1 currently supports only the exact supported hybrid specs";
                    }
                    return false;
                }
                return true;
            }
            if (spec.family == ModelFamily::DiffusionGemma) {
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
            } else if (model_spec_.family == ModelFamily::Qwen4Exp) {
                const auto& q4e_cfg = model_spec_.qwen4exp_text;
                AILA_LOG_INFO("[ModelSpec] model_type=%s family=qwen4_exp text_hidden=%d layers=%d vocab=%d",
                              model_spec_.model_type.c_str(),
                              q4e_cfg.hidden_size,
                              q4e_cfg.num_hidden_layers,
                              q4e_cfg.vocab_size);
                config_.hidden_size = q4e_cfg.hidden_size;
                config_.num_hidden_layers = q4e_cfg.num_hidden_layers;
                config_.num_attention_heads = q4e_cfg.num_attention_heads;
                config_.num_key_value_heads = q4e_cfg.num_key_value_heads;
                config_.head_dim = q4e_cfg.head_dim;
                config_.intermediate_size = q4e_cfg.moe_intermediate_size;
                config_.vocab_size = q4e_cfg.vocab_size;
                config_.max_position_embeddings = q4e_cfg.max_position_embeddings;
                config_.rope_theta = q4e_cfg.rope.rope_theta;
                config_.rms_norm_eps = q4e_cfg.rms_norm_eps;
                config_.tie_word_embeddings = q4e_cfg.tie_word_embeddings;
                config_.eos_token_id = q4e_cfg.eos_token_id;
                config_.bos_token_id = q4e_cfg.bos_token_id;
            } else if (model_spec_.family == ModelFamily::DiffusionGemma) {
                const auto& dg_cfg = model_spec_.diffusion_gemma.text;
                config_.hidden_size = dg_cfg.hidden_size;
                config_.num_hidden_layers = dg_cfg.num_hidden_layers;
                config_.num_attention_heads = dg_cfg.num_attention_heads;
                config_.num_key_value_heads = dg_cfg.num_key_value_heads;
                config_.head_dim = dg_cfg.head_dim;
                config_.intermediate_size = dg_cfg.intermediate_size;
                config_.vocab_size = dg_cfg.vocab_size;
                config_.max_position_embeddings = dg_cfg.max_position_embeddings;
                config_.rms_norm_eps = dg_cfg.rms_norm_eps;
                config_.tie_word_embeddings = dg_cfg.tie_word_embeddings;
                config_.bos_token_id = dg_cfg.bos_token_id;
                config_.eos_token_id = dg_cfg.eos_token_id;

                if (model_spec_.diffusion_gemma.boi_token_id > 0) {
                    tokenizer_.register_special_token("<|image>", static_cast<int>(model_spec_.diffusion_gemma.boi_token_id));
                }
                if (model_spec_.diffusion_gemma.image_token_id > 0) {
                    tokenizer_.register_special_token("<|image|>", static_cast<int>(model_spec_.diffusion_gemma.image_token_id));
                }
                if (model_spec_.diffusion_gemma.eoi_token_id > 0) {
                    tokenizer_.register_special_token("<image|>", static_cast<int>(model_spec_.diffusion_gemma.eoi_token_id));
                }

                AILA_LOG_INFO("[ModelSpec] model_type=%s family=diffusion_gemma text_hidden=%d layers=%d vocab=%d canvas=%d",
                              model_spec_.model_type.c_str(),
                              config_.hidden_size, config_.num_hidden_layers,
                              config_.vocab_size,
                              model_spec_.diffusion_gemma.canvas_length);
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
        if (model_spec_.family == ModelFamily::Qwen4Exp) {
            weights_ = std::make_unique<ModelWeights>(LoadModelWeightsLazy(model_dir, *ctx_));
        } else {
            weights_ = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(model_dir, *ctx_));
        }

        // Retain DiffusionGemma vision weights unless explicitly released by env
        const bool release_dg_vision = aila::env::read_flag("AILA_DG_RELEASE_VISION", false);
        if (model_spec_.family == ModelFamily::DiffusionGemma && release_dg_vision) {
            auto vision_stats = weights_->erase_with_prefix("model.encoder.vision_tower");
            auto embed_stats = weights_->erase_with_prefix("model.encoder.embed_vision");
            AILA_LOG_INFO("[DiffusionGemma] Released vision weights (%zu tensors, %.1f MB)",
                          vision_stats.count + embed_stats.count,
                          (vision_stats.bytes + embed_stats.bytes) / (1024.0 * 1024.0));
        }

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
        } else if (model_spec_.family == ModelFamily::Qwen4Exp) {
            backend_ = std::make_unique<Qwen4ExpBnb4Backend>();
        } else if (model_spec_.family == ModelFamily::DiffusionGemma) {
            backend_ = std::make_unique<DiffusionGemmaBnb4Backend>();
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
        } else if (model_spec_.family == ModelFamily::DiffusionGemma) {
            vision_encoder_ = std::make_unique<aila::vision::Gemma4VisionEncoder>();
            std::string vision_error;
            if (vision_encoder_->load(*ctx_, *weights_, model_spec_, model_dir_, &vision_error)) {
                vision_backend_enabled_ = true;
                AILA_LOG_INFO("[Vision] Gemma4/DiffusionGemma vision encoder loaded (out_hidden=%d)",
                              vision_encoder_->out_hidden_size());
            } else {
                AILA_LOG_WARN("[Vision] Gemma4 vision encoder init failed, falling back to text-only: %s",
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
        auto* q4e_warmup_backend=dynamic_cast<Qwen4ExpBnb4Backend*>(backend_.get());
        if(q4e_warmup_backend)q4e_warmup_backend->SetColdEpochEnabled(false);
        bool is_exact_q35_0p8b_spec =
            (model_spec_.family == ModelFamily::Qwen35Hybrid) &&
            is_exact_qwen35_hybrid_0p8b_spec(model_spec_.qwen35_text);
        token_input_staging_ = Tensor::allocate(*ctx_, {backend_->max_seq_len()},
                                               dnnl::memory::data_type::s32);
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
        } else if (model_spec_.family == ModelFamily::DiffusionGemma) {
            // vLLM-style warmup: one or more dummy full-canvas decoder passes
            // at init amortize SYCL kernel JIT, oneDNN primitive creation,
            // buffer allocation, and GPU clock ramp before the first real
            // request.  Auto-enable for DiffusionGemma; AILA_INIT_WARMUP=0
            // still disables it for cold-start measurement.
            run_init_warmup = true;
        }

        if (run_init_warmup) {
            if (init_warmup_mode == 1) {
                AILA_LOG_INFO("[Warmup] Running init warmup (forced via AILA_INIT_WARMUP=1)");
            } else if (model_spec_.family == ModelFamily::DiffusionGemma) {
                AILA_LOG_INFO("[Warmup] Running DiffusionGemma canvas warmup");
            } else if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
                AILA_LOG_INFO("[Warmup] Running init warmup for Qwen3.5 hybrid spec (hidden=%d layers=%d)",
                              model_spec_.qwen35_text.hidden_size,
                              model_spec_.qwen35_text.num_hidden_layers);
            } else {
                AILA_LOG_INFO("[Warmup] Running init warmup");
            }

            backend_->reset();

            auto t_warmup_start = std::chrono::high_resolution_clock::now();
            auto t_warmup_end = t_warmup_start;
            if (model_spec_.family == ModelFamily::DiffusionGemma) {
                auto* dg_backend =
                    dynamic_cast<DiffusionGemmaBnb4Backend*>(backend_.get());
                if (!dg_backend) {
                    AILA_LOG_WARN(
                        "[Warmup] DiffusionGemma backend unavailable; skipping warmup");
                    backend_->reset();
                } else {
                    const int canvas_len =
                        model_spec_.diffusion_gemma.canvas_length;
                    const int hidden =
                        model_spec_.diffusion_gemma.text.hidden_size;
                    const int warmup_passes = aila::env::read_int_raw(
                        "AILA_DG_WARMUP_PASSES", 1);
                    int warmup_canvas = aila::env::read_int_raw(
                        "AILA_DG_WARMUP_CANVAS", 0);
                    if (warmup_canvas <= 0) {
                        warmup_canvas = canvas_len;
                    }
                    warmup_canvas = std::min(warmup_canvas, canvas_len);
                    AILA_LOG_INFO(
                        "[Warmup] DiffusionGemma canvas=%d passes=%d",
                        warmup_canvas, warmup_passes);

                    // Dummy prompt encode warms the encode path (same decoder
                    // layers, small-M GEMM and masked attention).
                    constexpr int kWarmupPromptTokens = 8;
                    std::vector<int> dummy_prompt(kWarmupPromptTokens, 0);
                    int* prompt_dev = static_cast<int*>(
                        ctx_->alloc_device(kWarmupPromptTokens * sizeof(int)));
                    ctx_->memcpy_h2d_async(prompt_dev, dummy_prompt.data(),
                                           dummy_prompt.size() * sizeof(int));
                    std::string encode_error;
                    dg_backend->diffusion_encode(*ctx_, prompt_dev,
                                                 kWarmupPromptTokens,
                                                 &encode_error);
                    ctx_->synchronize();
                    ctx_->free_device(prompt_dev);

                    // Dummy full-canvas decoder passes with zero soft
                    // self-conditioning exercise every layer: full/linear
                    // attention, dense MLP, router, MoE, and logits.  The
                    // canvas tokens are discarded after warmup; KV cache
                    // positions written here are fully overwritten by the
                    // real encode + first canvas pass (reset clears
                    // current_len_).
                    std::vector<int> dummy_canvas(
                        static_cast<size_t>(warmup_canvas));
                    for (int i = 0; i < warmup_canvas; ++i) {
                        dummy_canvas[static_cast<size_t>(i)] =
                            i % model_spec_.diffusion_gemma.text.vocab_size;
                    }
                    int* canvas_dev = static_cast<int*>(
                        ctx_->alloc_device(
                            static_cast<size_t>(warmup_canvas) * sizeof(int)));
                    ctx_->memcpy_h2d_async(canvas_dev, dummy_canvas.data(),
                                           dummy_canvas.size() * sizeof(int));
                    auto* sc_dev = static_cast<sycl::ext::oneapi::bfloat16*>(
                        ctx_->alloc_device(
                            static_cast<size_t>(warmup_canvas) * hidden *
                            sizeof(sycl::ext::oneapi::bfloat16)));
                    ctx_->queue().memset(sc_dev, 0,
                                         static_cast<size_t>(warmup_canvas) *
                                             hidden *
                                             sizeof(sycl::ext::oneapi::bfloat16));

                    Tensor* warmup_logits = nullptr;
                    try {
                        for (int pass = 0; pass < warmup_passes; ++pass) {
                            warmup_logits = &dg_backend->diffusion_decode_canvas(
                                *ctx_, canvas_dev, sc_dev, warmup_canvas);
                        }
                        if (warmup_logits != nullptr) {
                            dg_backend->diffusion_compute_soft_embeds(
                                *ctx_, *warmup_logits, warmup_canvas);
                            const int vocab =
                                model_spec_.diffusion_gemma.text.vocab_size;
                            int* dummy_argmax_dev = static_cast<int*>(
                                ctx_->alloc_device(
                                    static_cast<size_t>(warmup_canvas) *
                                    sizeof(int)));
                            float* dummy_entropy_dev = static_cast<float*>(
                                ctx_->alloc_device(
                                    static_cast<size_t>(warmup_canvas) *
                                    sizeof(float)));
                            ops::diffusion_argmax_device(
                                *ctx_,
                                static_cast<const float*>(
                                    warmup_logits->data()),
                                dummy_argmax_dev, warmup_canvas, vocab);
                            ops::diffusion_softmax_entropy_device(
                                *ctx_,
                                static_cast<const float*>(
                                    warmup_logits->data()),
                                dummy_entropy_dev, warmup_canvas, vocab);
                            ctx_->synchronize();
                            ctx_->free_device(dummy_argmax_dev);
                            ctx_->free_device(dummy_entropy_dev);
                        }
                        ctx_->synchronize();
                    } catch (const std::exception& error) {
                        AILA_LOG_ERROR(
                            "[Warmup] DiffusionGemma warmup failed: %s",
                            error.what());
                        throw;
                    }

                    ctx_->free_device(canvas_dev);
                    ctx_->free_device(sc_dev);
                    backend_->reset();

                    t_warmup_end =
                        std::chrono::high_resolution_clock::now();
                }
            } else {
            // Lightweight warmup: 2-token prefill (like llama.cpp's approach).
            // A pure decode (seq_len=1) would skip prefill-score buffer allocation;
            // the first real prefill would then allocate a large buffer directly
            // and occasionally trigger GPU-level crashes on Arc A770.  Two tokens
            // trigger the prefill path with a 2×2 score buffer (~tens of bytes),
            // exercising the same allocation path as a real prefill at minimal cost.
            int warmup_token_id = (config_.bos_token_id >= 0) ? config_.bos_token_id : 0;
            const int warmup_prefill_len = q4e_warmup_backend ? 16 : 2;
            int warmup_arr[16] = {};
            for (int i = 0; i < warmup_prefill_len; ++i) warmup_arr[i] = warmup_token_id;
            int* warmup_token_ids = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(warmup_prefill_len) * sizeof(int)));
            ctx_->memcpy_h2d_async(warmup_token_ids, warmup_arr,
                                   static_cast<size_t>(warmup_prefill_len) * sizeof(int));

            int* warmup_argmax = nullptr;
            try {
                Tensor& warmup_logits =
                    backend_->forward(*ctx_, warmup_token_ids, warmup_prefill_len);
                warmup_argmax = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
                ops::argmax(*ctx_, warmup_logits, config_.vocab_size, warmup_argmax);
                ctx_->synchronize();

                if (q4e_warmup_backend) {
                    q4e_warmup_backend->WarmupPrefillKernels(*ctx_);
                }

                // H0: exercise count=1 decode step while session has cached tokens,
                // compiling decode GEMV, decode attention / JM, decode MoE, and LM head.
                Tensor& warmup_decode_logits = backend_->forward(*ctx_, warmup_token_ids, 1);
                ops::argmax(*ctx_, warmup_decode_logits, config_.vocab_size, warmup_argmax);
                ctx_->synchronize();
                t_warmup_end = std::chrono::high_resolution_clock::now();

                ctx_->free_device(warmup_argmax);
                warmup_argmax = nullptr;
                ctx_->free_device(warmup_token_ids);
                warmup_token_ids = nullptr;
                backend_->reset();

                AILA_LOG_INFO("[Warmup Manifest] reached: %d-token prefill (gemm, score buf), prefill kernel warmup, 1-token decode (gemv, decode attn, moe), state reset", warmup_prefill_len);
            } catch (const std::exception& e) {
                AILA_LOG_ERROR("[Warmup] Lightweight warmup failed: %s", e.what());
                try { ctx_->queue().wait_and_throw(); } catch (...) {}
                if (warmup_argmax) ctx_->free_device(warmup_argmax);
                ctx_->free_device(warmup_token_ids);
                throw;
            }
            }

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
            } else if (init_warmup_mode == 0) {
                AILA_LOG_INFO("[Warmup] Skipping init warmup (AILA_INIT_WARMUP=0)");
            } else {
                AILA_LOG_INFO("[Warmup] Skipping init warmup for this model family. Set AILA_INIT_WARMUP=1 to force.");
            }
            AILA_LOG_INFO("[Memory] Warmup skipped: current=%.2f MB, peak=%.2f MB",
                          to_mb(ctx_->current_allocated_bytes()),
                          to_mb(ctx_->peak_allocated_bytes()));
        }

        AILA_LOG_INFO("========================================");
        if(q4e_warmup_backend)q4e_warmup_backend->SetColdEpochEnabled(true);
        AILA_LOG_INFO("  Engine ready!");
        AILA_LOG_INFO("========================================");

        return true;
    }
