#include "engine/Engine.hpp"
#include "chat/AssistantOutputParser.hpp"

void InferenceEngine::align_backend_context_to_sequence(const std::vector<int>& prompt_ids,
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

        if (actual_len > accepted_len) {
            // Truncation left extra tokens behind; rebuild from scratch.
            backend_->reset();
            actual_len = 0;
        }
        if (actual_len < prompt_len) {
            replay_range(prompt_ids, actual_len, prompt_len - actual_len);
            actual_len = backend_->get_current_context_len();
        }

        // The backend now holds a prefix of prompt + generated (typically only the last
        // sampled token was never fed), so only the missing generated tail is replayed.
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

Tensor* InferenceEngine::forward_prompt_tokens(const std::vector<int>& token_ids,
                                  int start,
                                  int len,
                                  int snapshot_after_len,
                                  bool async_h2d) {
        if (!backend_ || !ctx_ || len <= 0) return nullptr;

        int chunk_size = len;
        if (model_spec_.family == ModelFamily::Qwen35Hybrid || model_spec_.family == ModelFamily::Qwen4Exp) {
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

std::string InferenceEngine::decode_with_special_tokens(const std::vector<int>& ids) const {
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

void InferenceEngine::debug_dump_prompt_text(const char* label,
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

aila::chat::ChatRequest InferenceEngine::make_chat_request_from_legacy(
        const std::vector<Message>& messages,
        const GenerationConfig& config,
        const aila::chat::ChatRequest* source_request) const {
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
                if (part.type == ContentType::Text && !part.text.empty()) {
                    // Strip thinking blocks from assistant history or previous turns
                    // to avoid multi-turn reasoning pollution.
                    if (msg.role == aila::chat::ChatRole::Assistant || i + 1 < messages.size()) {
                        out.text = aila::chat::strip_thinking_blocks(part.text);
                    }
                }
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

aila::chat::ChatFormatInput InferenceEngine::make_chat_format_input(const aila::chat::ChatRequest& request) const {
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

std::string InferenceEngine::current_model_family_name() const {
        switch (model_spec_.family) {
        case ModelFamily::Yolo26: return "yolo26";
        case ModelFamily::Qwen4Exp: return "qwen4_exp";
        case ModelFamily::Qwen35Hybrid: return "qwen3_5_hybrid";
        case ModelFamily::Qwen3Dense: return "qwen3";
        default: return "unknown";
        }
    }

std::string InferenceEngine::tool_choice_name(const aila::chat::ChatRequest& request) const {
        switch (request.tool_choice) {
        case aila::chat::ToolChoice::None: return "none";
        case aila::chat::ToolChoice::Required: return "required";
        case aila::chat::ToolChoice::Function: return "function:" + request.tool_choice_function_name;
        case aila::chat::ToolChoice::Auto: return "auto";
        }
        return "auto";
    }

std::string InferenceEngine::tool_policy_name(aila::chat::ToolPolicyMode policy) const {
        return policy == aila::chat::ToolPolicyMode::Strict ? "strict" : "warn";
    }

std::vector<int> InferenceEngine::thinking_close_token_ids() const {
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

std::string InferenceEngine::generate(const std::string& user_message,
                         const GenerationConfig& gen_config,
                         std::function<void(const std::string&)> token_callback) {
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

std::string InferenceEngine::generate_messages(const std::vector<Message>& messages,
                                  const GenerationConfig& gen_config,
                                  std::function<void(const std::string&)> token_callback,
                                  const aila::chat::ChatRequest* structured_request) {
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
                    if (model_spec_.family == ModelFamily::DiffusionGemma) {
                        ph.text += "<|image>";
                        for (int i = 0; i < encoded.token_count; ++i) {
                            ph.text += "<|image|>";
                        }
                        ph.text += "<image|>";
                    } else {
                        ph.text += "<|vision_start|>";
                        for (int i = 0; i < encoded.token_count; ++i) {
                            ph.text += "<|image_pad|>";
                        }
                        ph.text += "<|vision_end|>";
                    }
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
            (model_spec_.family == ModelFamily::Qwen35Hybrid || model_spec_.family == ModelFamily::Qwen4Exp) &&
            (rendered_ends_with(rendered.text, "<think>\n") ||
             rendered_ends_with(rendered.text, "<think>\r\n") ||
             rendered_ends_with(rendered.text, "<think>"));
        int recurrent_prefill_anchor = 0;
        if ((model_spec_.family == ModelFamily::Qwen35Hybrid || model_spec_.family == ModelFamily::Qwen4Exp) &&
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
            if (image_pad_id < 0 && model_spec_.family == ModelFamily::DiffusionGemma) {
                image_pad_id = static_cast<int>(model_spec_.diffusion_gemma.image_token_id);
            }
            if (image_pad_id < 0) {
                image_pad_id = tokenizer_.special_token_id("<|image|>");
            }
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

            if (model_spec_.family == ModelFamily::DiffusionGemma) {
                for (size_t k = 0; k < full_ids.size(); ++k) {
                    pos_t[k] = static_cast<int>(k);
                    pos_h[k] = static_cast<int>(k);
                    pos_w[k] = static_cast<int>(k);
                }
            } else {
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

        if (model_spec_.family == ModelFamily::DiffusionGemma) {
            if (total_audio_tokens > 0) {
                set_error(EngineErrorCode::ModelCapability,
                          "DiffusionGemma does not support audio prompts");
                return "";
            }
            auto* dg_backend =
                dynamic_cast<DiffusionGemmaBnb4Backend*>(backend_.get());
            if (!dg_backend) {
                set_error(EngineErrorCode::RuntimeError,
                          "DiffusionGemma backend is unavailable");
                return "";
            }
            const auto& dgc = model_spec_.diffusion_gemma;
            DiffusionGemmaGeneratorConfig dg_cfg;
            dg_cfg.canvas_length = dgc.canvas_length;
            // The 26B checkpoint converges reliably at 25 steps in the
            // trusted Transformers path. Keep that validated latency default
            // while still honoring checkpoints that request fewer steps.
            dg_cfg.max_denoising_steps =
                std::min(dgc.max_denoising_steps, 25);
            const int steps_override =
                aila::env::read_int_raw("AILA_DIFFUSION_STEPS", -1);
            if (steps_override > 0) {
                dg_cfg.max_denoising_steps = steps_override;
            }
            dg_cfg.t_min = dgc.t_min;
            dg_cfg.t_max = dgc.t_max;
            dg_cfg.entropy_bound = dgc.entropy_bound;
            dg_cfg.confidence_threshold = dgc.confidence_threshold;
            dg_cfg.stability_threshold = dgc.stability_threshold;
            dg_cfg.pad_token_id = dgc.pad_token_id;
            dg_cfg.eos_token_ids = dgc.eos_token_ids;
            dg_cfg.seed = gen_config.use_fixed_seed
                              ? gen_config.sampling_seed
                              : static_cast<uint64_t>(std::random_device{}());

            std::vector<int> generated_ids;
            std::string dg_error;
            DiffusionGemmaGenerator generator;
            DiffusionGemmaGenerator::CanvasCallback canvas_callback;
            if (token_callback) {
                canvas_callback = [&](const std::vector<int>& canvas_tokens) {
                    for (int token : canvas_tokens) {
                        token_callback(tokenizer_.decode(token));
                    }
                };
            }
            try {
                if (!generator.generate(*ctx_, *dg_backend, full_ids, dg_cfg,
                                        max_new_tokens, generated_ids, &dg_error,
                                        canvas_callback)) {
                    set_error(EngineErrorCode::RuntimeError, dg_error);
                    return "";
                }
            } catch (const std::exception& e) {
                set_error(EngineErrorCode::RuntimeError,
                          std::string("DiffusionGemma generation failed: ") +
                              e.what());
                return "";
            }
            std::string output_text;
            output_text.reserve(generated_ids.size() * 4);
            for (int token : generated_ids) {
                output_text += tokenizer_.decode(token);
            }
            last_generation_finish_reason_ =
                (generated_ids.size() >= static_cast<size_t>(max_new_tokens))
                    ? "length"
                    : "stop";
            return output_text;
        }

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
        bool tokenwise_prefill = (model_spec_.family == ModelFamily::Qwen35Hybrid || model_spec_.family == ModelFamily::Qwen4Exp) &&
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
        if (model_spec_.family == ModelFamily::Qwen4Exp) {
            // Every Qwen4Exp forward already syncs with the host per layer (expert routing),
            // so decoding ahead saves nothing and an EOS inside the chunk costs a replay.
            effective_chunk_size = 1;
        }
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

std::string InferenceEngine::generate_chat_request_text(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback) {
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

aila::chat::AssistantChatResult InferenceEngine::generate_chat_request_result(
        const aila::chat::ChatRequest& request,
        std::function<void(const std::string&)> token_callback) {
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

aila::chat::StructuredStreamEvent InferenceEngine::make_final_stream_event(
        const aila::chat::AssistantChatResult& result) const {
        return aila::chat::final_stream_event_from_result(result);
    }

int InferenceEngine::generate_chat_request_stream(
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

std::string InferenceEngine::generate_messages_json(const std::string& messages_json,
                                       const GenerationConfig& gen_config,
                                       std::function<void(const std::string&)> token_callback) {
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

std::string InferenceEngine::generate_chat_json(const std::string& chat_request_json,
                                   const GenerationConfig& gen_config,
                                   std::function<void(const std::string&)> token_callback) {
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
