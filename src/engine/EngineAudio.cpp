#include "engine/Engine.hpp"

int InferenceEngine::find_split_point(const float* samples, int n_samples, int target_sample, float search_sec) {
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

bool InferenceEngine::should_insert_boundary_space(int prev_ch, int next_ch) {
        if (prev_ch <= 0 || next_ch <= 0) return false;
        if (std::isspace(static_cast<unsigned char>(prev_ch))) return false;
        if (std::isspace(static_cast<unsigned char>(next_ch))) return false;
        if (std::ispunct(static_cast<unsigned char>(next_ch))) return false;
        return true;
    }

std::string InferenceEngine::transcribe_segment_raw(const std::vector<float>& seg_buf,
                                       const GenerationConfig& gen_config,
                                       const std::string& forced_language,
                                       const std::string& system_prompt,
                                       const std::string& past_text,
                                       std::string* language_out) {
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

bool InferenceEngine::synthesize_codes(const std::vector<int>& text_tokens,
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

bool InferenceEngine::synthesize_wav(const std::vector<int>& text_tokens,
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

bool InferenceEngine::synthesize_text_to_wav(const std::string& text,
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

std::vector<int> InferenceEngine::tokenize_tts_text(const std::string& text) const {
        return tokenizer_.encode(text);
    }

bool InferenceEngine::synthesizeSpeech(const std::string& text,
                          const std::string& reference_audio_path,
                          const std::string& speaker_name,
                          const std::string& instruct_text,
                          const std::string& language,
                          const GenerationConfig& gen_config,
                          std::vector<float>& out_samples,
                          const std::string& reference_text,
                          VoiceCloneMode clone_mode) {
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

        if (clone_mode == VoiceCloneMode::Icl) {
            if (reference_audio_path.empty()) {
                set_error(EngineErrorCode::InvalidArgument, "ICL voice clone requires reference_audio_path");
                return false;
            }
            if (reference_text.empty()) {
                set_error(EngineErrorCode::InvalidArgument, "ICL voice clone requires reference_text");
                return false;
            }
        }

        if (!reference_audio_path.empty()) {
            // Base mode: ECAPA-TDNN voice cloning from reference audio
            // extractSpeakerEmbedding sets its own detailed error on failure
            if (!extractSpeakerEmbedding(reference_audio_path, spk_emb)) {
                return false;
            }

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

        // 4. Tokenize the main text (pure text tokens without ChatML tags)
        std::vector<int> tokens = tokenize_tts_text(text);
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

        // 6. Decode codes to audio via Mimi vocoder (official: prepend ref_codes for smooth vocoder context, then slice)
        int ref_frames = (p_clone_prompt && p_clone_prompt->reference_codes.frames > 0)
                         ? p_clone_prompt->reference_codes.frames : 0;
        if (ref_frames > 0) {
            std::vector<int32_t> full_codes = p_clone_prompt->reference_codes.codes;
            full_codes.insert(full_codes.end(), codes.begin(), codes.end());
            std::vector<float> full_samples;
            ok = tts_backend->decode_mimi_vocoder(*ctx_, full_codes, ref_frames + n_frames, full_samples);
            if (ok) {
                // Official Mimi 12Hz upsample rate is exactly 1920 samples/frame (24000 / 12.5)
                size_t cut_samples = static_cast<size_t>(ref_frames) * 1920;
                if (full_samples.size() > cut_samples) {
                    out_samples.assign(full_samples.begin() + cut_samples, full_samples.end());
                } else {
                    out_samples = std::move(full_samples);
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

std::thread InferenceEngine::synthesizeSpeechStream(
        const std::string& text,
        const std::string& reference_audio_path,
        const std::string& speaker_name,
        const std::string& instruct_text,
        const std::string& language,
        const GenerationConfig& gen_config,
        std::function<void(const float*, int)> callback,
        int stream_batch_frames,
        const std::string& reference_text,
        VoiceCloneMode clone_mode,
        std::shared_ptr<std::atomic<int>> status_out,
        std::shared_ptr<std::string> err_msg_out) {
        clear_error();
        if (reject_yolo_capability("TTS")) {
            if (status_out) status_out->store(static_cast<int>(EngineErrorCode::ModelCapability));
            return {};
        }

        if (clone_mode == VoiceCloneMode::Icl) {
            if (reference_audio_path.empty()) {
                set_error(EngineErrorCode::InvalidArgument, "ICL voice clone requires reference_audio_path");
                if (status_out) status_out->store(static_cast<int>(EngineErrorCode::InvalidArgument));
                return {};
            }
            if (reference_text.empty()) {
                set_error(EngineErrorCode::InvalidArgument, "ICL voice clone requires reference_text");
                if (status_out) status_out->store(static_cast<int>(EngineErrorCode::InvalidArgument));
                return {};
            }
        }

        return std::thread([this, text, reference_audio_path, speaker_name,
                            instruct_text, language, gen_config, callback,
                            stream_batch_frames, reference_text, clone_mode,
                            status_out, err_msg_out]() {
            // Determine mode
            std::vector<float> spk_emb;
            int spk_id = 0;
            VoiceClonePrompt clone_prompt;
            const VoiceClonePrompt* p_clone_prompt = nullptr;

            if (!reference_audio_path.empty()) {
                if (!extractSpeakerEmbedding(reference_audio_path, spk_emb)) {
                    if (status_out) status_out->store(static_cast<int>(EngineErrorCode::RuntimeError));
                    if (err_msg_out) *err_msg_out = "Failed to extract speaker embedding";
                    return;
                }
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
                            } else {
                                if (status_out) status_out->store(static_cast<int>(EngineErrorCode::RuntimeError));
                                if (err_msg_out) *err_msg_out = "Failed to encode reference audio: " + enc_err;
                                return;
                            }
                        } else {
                            if (status_out) status_out->store(static_cast<int>(EngineErrorCode::RuntimeError));
                            if (err_msg_out) *err_msg_out = "Failed to load Mimi Audio Encoder";
                            return;
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
            // 4. Tokenize the main text (pure text tokens without ChatML tags, identical to blocking mode)
            std::vector<int> tokens = tokenize_tts_text(text);

            // Run streaming synthesis
            auto tts_backend = dynamic_cast<Qwen3TTSBackend*>(backend_.get());
            if (!tts_backend) {
                if (status_out) status_out->store(static_cast<int>(EngineErrorCode::RuntimeError));
                if (err_msg_out) *err_msg_out = "TTS backend not initialized";
                return;
            }

            bool ok = tts_backend->synthesize_codes_stream(
                *ctx_, tokens, spk_emb, spk_id, instruct_tokens, lang_id,
                gen_config, stream_batch_frames,
                [&](const std::vector<float>& chunk) {
                    if (!chunk.empty()) {
                        callback(chunk.data(), static_cast<int>(chunk.size()));
                    }
                },
                p_clone_prompt);
            if (!ok) {
                if (status_out) status_out->store(static_cast<int>(EngineErrorCode::RuntimeError));
                if (err_msg_out) *err_msg_out = "Streaming TTS synthesis failed";
            }
        });
    }

bool InferenceEngine::decode_mimi_vocoder(const std::vector<int32_t>& codes,
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

bool InferenceEngine::extractSpeakerEmbedding(const std::string& audio_path,
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

std::vector<aila::AlignedWord> InferenceEngine::align(
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

std::vector<aila::AlignedWord> InferenceEngine::align_words(
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

std::string InferenceEngine::transcribe(const std::string& wav_path,
                           const GenerationConfig& gen_config,
                           std::string* language_out,
                           const std::string& forced_language,
                           const std::string& system_prompt,
                           float segment_sec,
                           bool past_text_conditioning,
                           std::function<void(const std::string&)> token_callback) {
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

InferenceEngine::TranscribeStream::TranscribeStream(
            InferenceEngine* engine,
            const GenerationConfig& gen_config,
            const std::string& forced_language,
            const std::string& system_prompt) : engine_(engine),
            gen_config_(gen_config),
            forced_language_(forced_language),
            system_prompt_(system_prompt),
            stable_samples_offset_(0)
        {}

void InferenceEngine::TranscribeStream::feed_audio(const float* data, size_t count) {
            if (!data || count == 0) return;
            audio_buffer_.insert(audio_buffer_.end(), data, data + count);
        }

void InferenceEngine::TranscribeStream::process() {
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

void InferenceEngine::TranscribeStream::get_text(std::string& out_stable, std::string& out_partial) {
            process();
            out_stable = stable_text_;
            out_partial = partial_text_;
        }
