#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/IModelBackend.hpp"
#include "../src/models/Qwen3DenseBackend.hpp"
#include "../src/models/Qwen35HybridTextBackend.hpp"
#include "../src/vision/Qwen35VisionEncoder.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/ModelConfig.hpp"
#include "../src/utils/ModelSpec.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "../src/templates/TemplateRegistry.hpp"
#include "../src/profile/Profiling.hpp"
#include "../src/utils/EnvUtils.hpp"
#include "simdjson.h"
#include "Types.hpp"
#include <string>
#include <functional>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cctype>

// ============================================================
// Inference Engine: orchestrates loading, tokenization, inference
// ============================================================
class InferenceEngine {
public:
    InferenceEngine() = default;

    // Initialize: load model + tokenizer from model directory
    bool init(const std::string& model_dir, int max_seq_len = 4096) {
        clear_error();
        model_dir_ = model_dir;

        AILA_LOG_INFO("========================================");
        AILA_LOG_INFO("  Aila Inference Engine");
        AILA_LOG_INFO("========================================");

        // 1. Create context
        AILA_LOG_INFO("[1/3] Initializing GPU context...");
        ctx_ = std::make_unique<Context>();

        // 2. Load tokenizer
        AILA_LOG_INFO("[2/3] Loading tokenizer...");
        if (!tokenizer_.load(model_dir)) {
            AILA_LOG_ERROR("Failed to load tokenizer");
            return false;
        }

        // 3. Load model weights
        AILA_LOG_INFO("[3/3] Loading model weights...");
        weights_ = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(model_dir, *ctx_));

        // 3.5 Load unified model spec
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

                // Qwen3.5-0.8B is a VL-capable model. In text-only phase, a stronger
                // default system prompt avoids drifting to image-assumption replies.
                if (!model_spec_.vision.enabled && system_prompt_ == "You are a helpful assistant.") {
                    system_prompt_ =
                        "You are a helpful text assistant. "
                        "Reply in the user's language when possible. "
                        "Do not assume images or videos unless explicitly provided.";
                    AILA_LOG_INFO("[Config] Applied Qwen3.5 text-only default system prompt");
                }
            } else {
                config_ = model_spec_.qwen3;
                AILA_LOG_INFO("[ModelSpec] model_type=%s family=qwen3_dense hidden=%d layers=%d vocab=%d",
                              model_spec_.model_type.empty() ? "qwen3" : model_spec_.model_type.c_str(),
                              config_.hidden_size, config_.num_hidden_layers, config_.vocab_size);
            }
        }

        if (max_seq_len > config_.max_position_embeddings) {
            AILA_LOG_WARN("[Config] max_seq_len=%d exceeds model max_position_embeddings=%d, clamping",
                          max_seq_len, config_.max_position_embeddings);
            max_seq_len = config_.max_position_embeddings;
        }

        // 4. Initialize backend
        if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
            backend_ = std::make_unique<Qwen35HybridTextBackend>();
        } else {
            backend_ = std::make_unique<Qwen3DenseBackend>();
        }
        std::string backend_error;
        if (!backend_->load(*ctx_, *weights_, model_spec_, max_seq_len, &backend_error)) {
            AILA_LOG_ERROR("Failed to initialize model backend: %s", backend_error.c_str());
            return false;
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
        {
            backend_->reset();
            std::vector<Message> warmup_messages = {
                Message{"system", {ContentPart{ContentType::Text, "You are a helpful assistant.", ""}}},
                Message{"user", {ContentPart{ContentType::Text, "Warmup.", ""}}}
            };
            std::vector<int> warmup_ids;
            std::string warmup_err;
            if (!template_registry_.render(model_spec_.family, tokenizer_, warmup_messages,
                                           vision_backend_enabled_, true, warmup_ids, &warmup_err)) {
                AILA_LOG_WARN("[Warmup] Template render failed (%s), fallback to legacy tokenizer template",
                              warmup_err.c_str());
                warmup_ids = tokenizer_.apply_chat_template("You are a helpful assistant.", "Warmup.");
            }
            int* warmup_token_ids = static_cast<int*>(
                ctx_->alloc_device(warmup_ids.size() * sizeof(int)));
            ctx_->memcpy_h2d_async(warmup_token_ids, warmup_ids.data(),
                                   warmup_ids.size() * sizeof(int));

            auto t_warmup_start = std::chrono::high_resolution_clock::now();
            Tensor& warmup_logits = backend_->forward(*ctx_, warmup_token_ids, static_cast<int>(warmup_ids.size()));
            int* warmup_argmax = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            ops::argmax(*ctx_, warmup_logits, config_.vocab_size, warmup_argmax);
            ctx_->synchronize();
            auto t_warmup_end = std::chrono::high_resolution_clock::now();

            ctx_->free_device(warmup_argmax);
            ctx_->free_device(warmup_token_ids);
            backend_->reset();

            double warmup_ms = std::chrono::duration<double, std::milli>(t_warmup_end - t_warmup_start).count();
            AILA_LOG_INFO("[Warmup] Completed in %.2f ms", warmup_ms);
        }
        AILA_LOG_INFO("[Memory] After warmup: current=%.2f MB, peak=%.2f MB",
                      to_mb(ctx_->current_allocated_bytes()),
                      to_mb(ctx_->peak_allocated_bytes()));

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
        cached_ids_.clear();
        benchmark_seed_ready_ = false;
        if (backend_) backend_->reset();
        AILA_LOG_INFO("[Context] Conversation reset");
    }

    int context_length() const { return static_cast<int>(cached_ids_.size()); }
    int max_context_length() const { return backend_ ? backend_->max_seq_len() : 0; }
    const ChatHistory& history() const { return history_; }

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
        if (model_spec_.family == ModelFamily::Qwen35Hybrid) {
            auto rtrim_inplace = [](std::string& s) {
                while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
                    s.pop_back();
                }
            };
            auto strip_no_think_suffix = [&](std::string& s) -> bool {
                const std::string cmd = "/no_think";
                rtrim_inplace(s);
                if (s.size() < cmd.size()) return false;
                size_t pos = s.size() - cmd.size();
                if (s.compare(pos, cmd.size(), cmd) != 0) return false;
                bool boundary_ok = (pos == 0) ||
                                   std::isspace(static_cast<unsigned char>(s[pos - 1]));
                if (!boundary_ok) return false;
                s.erase(pos);
                rtrim_inplace(s);
                return true;
            };
            auto strip_think_blocks = [](std::string& text) {
                while (true) {
                    size_t start = text.find("<think>");
                    if (start == std::string::npos) break;
                    size_t end = text.find("</think>", start);
                    if (end == std::string::npos) {
                        text.erase(start);
                        break;
                    }
                    text.erase(start, end + 8 - start);
                }
                size_t i = 0;
                while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
                if (i > 0) text.erase(0, i);
                while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
                    text.pop_back();
                }
            };
            auto drop_oldest_mm_pair = [&]() -> bool {
                int first_user = -1;
                for (int i = 0; i < static_cast<int>(mm_history_.size()); ++i) {
                    if (mm_history_[(size_t)i].role == "user") {
                        first_user = i;
                        break;
                    }
                }
                if (first_user < 0) return false;
                int erase_count = 1;
                if (first_user + 1 < static_cast<int>(mm_history_.size()) &&
                    mm_history_[(size_t)(first_user + 1)].role == "assistant") {
                    erase_count = 2;
                }
                mm_history_.erase(mm_history_.begin() + first_user,
                                  mm_history_.begin() + first_user + erase_count);
                return true;
            };

            if ((mm_history_.empty() || mm_history_[0].role != "system") && !system_prompt_.empty()) {
                Message sys_msg;
                sys_msg.role = "system";
                sys_msg.content.push_back(ContentPart{ContentType::Text, system_prompt_, ""});
                mm_history_.insert(mm_history_.begin(), std::move(sys_msg));
            }

            std::string cleaned_user_message = user_message;
            strip_no_think_suffix(cleaned_user_message);
            Message user_msg;
            user_msg.role = "user";
            user_msg.content.push_back(ContentPart{ContentType::Text, cleaned_user_message, ""});
            mm_history_.push_back(user_msg);

            std::string out = generate_messages(mm_history_, gen_config, token_callback);
            while (last_error_code_ == EngineErrorCode::ContextOverflow) {
                if (!drop_oldest_mm_pair()) {
                    break;
                }
                AILA_LOG_WARN("[Generate] Qwen3.5 history truncated to fit context window (%zu messages remaining)",
                              mm_history_.size());
                out = generate_messages(mm_history_, gen_config, token_callback);
            }

            if (last_error_code_ != EngineErrorCode::Ok) {
                if (!mm_history_.empty() && mm_history_.back().role == "user") {
                    mm_history_.pop_back();
                }
                return "";
            }

            std::string history_text = out;
            strip_think_blocks(history_text);
            Message assistant_msg;
            assistant_msg.role = "assistant";
            assistant_msg.content.push_back(ContentPart{ContentType::Text, history_text, ""});
            mm_history_.push_back(assistant_msg);
            return out;
        }

        // --- Think-suppression detection ---
        auto trim_copy = [](const std::string& s) -> std::string {
            std::string out = s;
            while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
                out.pop_back();
            }
            return out;
        };
        auto ends_with_no_think = [&](const std::string& s) -> bool {
            const std::string cmd = "/no_think";
            std::string t = trim_copy(s);
            if (t.size() < cmd.size()) return false;
            size_t pos = t.size() - cmd.size();
            if (t.compare(pos, cmd.size(), cmd) != 0) return false;
            return (pos == 0) || std::isspace(static_cast<unsigned char>(t[pos - 1]));
        };
        auto strip_leading_think_artifacts = [](std::string& text) {
            auto ltrim = [](std::string& x) {
                size_t i = 0;
                while (i < x.size() && std::isspace(static_cast<unsigned char>(x[i]))) ++i;
                if (i > 0) x.erase(0, i);
            };

            bool changed = true;
            while (changed) {
                changed = false;
                ltrim(text);
                if (text.rfind("<think>", 0) == 0) {
                    text.erase(0, 7);
                    changed = true;
                    continue;
                }
                if (text.rfind("</think>", 0) == 0) {
                    text.erase(0, 8);
                    changed = true;
                    continue;
                }
            }
            ltrim(text);
        };

        bool no_think_requested = ends_with_no_think(user_message);

        // --- Add user message to history (for display / overflow rebuild) ---
        history_.add_user(user_message);

        // --- Build the token sequence for this turn ---
        // We always use apply_chat_template to cleanly construct the prompt,
        // which naturally handles dynamically stripped context (e.g. removed <think> blocks).
        std::vector<int> full_ids = tokenizer_.apply_chat_template(system_prompt_, history_);
        int reusable_prefix = 0;

        // Find the longest common prefix (LCP) with the current KV cache contents
        int max_possible_match = std::min(static_cast<int>(cached_ids_.size()), static_cast<int>(full_ids.size()));
        while (reusable_prefix < max_possible_match && 
               cached_ids_[reusable_prefix] == full_ids[reusable_prefix]) {
            reusable_prefix++;
        }

        // Truncate the physical KV cache inside the model to match the reusable prefix.
        // This is necessary because if we stripped <think> blocks, the sequence diverged,
        // and we cannot append new tokens to the end of the previous longer sequence.
        if (backend_) backend_->truncate_kv_cache(reusable_prefix);
        cached_ids_.resize(reusable_prefix);

        // --- Context overflow: if too long, clear and rebuild from scratch ---
        int max_ctx = backend_ ? backend_->max_seq_len() : 0;
        if (static_cast<int>(full_ids.size()) > max_ctx - 64) {
            // Truncate history until it fits
            while (static_cast<int>(full_ids.size()) > max_ctx - 64 && history_.size() > 1) {
                history_.truncate_oldest(static_cast<int>(history_.size()) - 2);
                full_ids = tokenizer_.apply_chat_template(system_prompt_, history_);
                AILA_LOG_WARN("[Context] History truncated to fit context window (%zu messages remaining)",
                              history_.size());
            }
            // Must do full prefill since we rebuilt from scratch
            if (backend_) backend_->reset();
            cached_ids_.clear();
            reusable_prefix = 0;
        }

        int total_prompt_len = static_cast<int>(full_ids.size());

        // --- Determine prefill range ---
        int prefill_start = reusable_prefix;
        int new_tokens_to_prefill = total_prompt_len - prefill_start;

        if (prefill_start > 0) {
            AILA_LOG_INFO("[Generate] Incremental prefill: reusing %d cached tokens, prefilling %d new tokens",
                          prefill_start, new_tokens_to_prefill);
        } else {
            AILA_LOG_INFO("[Generate] Full prefill: %d tokens", total_prompt_len);
        }


        int available_decode_tokens = max_ctx - total_prompt_len;
        if (available_decode_tokens <= 0) {
            AILA_LOG_ERROR("[Generate] Prompt exceeds context window (prompt=%d, max_seq_len=%d)",
                           total_prompt_len, max_ctx);
            set_error(EngineErrorCode::ContextOverflow, "Prompt exceeds context window");
            // Remove the user message we just added since we can't process it
            history_.truncate_oldest(static_cast<int>(history_.size()) - 1);
            return "";
        }
        int max_new_tokens = std::min(gen_config.max_new_tokens, available_decode_tokens);
        if (max_new_tokens < gen_config.max_new_tokens) {
            AILA_LOG_INFO("[Generate] max_new_tokens clamped from %d to %d due to context window limit",
                          gen_config.max_new_tokens, max_new_tokens);
        }

        // --- Upload new tokens to GPU and prefill ---
        int* token_ids_device = static_cast<int*>(
            ctx_->alloc_device(static_cast<size_t>(new_tokens_to_prefill) * sizeof(int)));
        ctx_->memcpy_h2d_async(token_ids_device,
                               full_ids.data() + prefill_start,
                               static_cast<size_t>(new_tokens_to_prefill) * sizeof(int));

        auto t_start = std::chrono::high_resolution_clock::now();
        Tensor& logits = backend_->forward(*ctx_, token_ids_device, new_tokens_to_prefill);
        ctx_->synchronize();
        auto t_prefill = std::chrono::high_resolution_clock::now();

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        AILA_LOG_INFO("[Generate] Prefill: %d tokens in %.2f ms (%.1f tok/s)",
                      new_tokens_to_prefill, prefill_ms,
                      new_tokens_to_prefill / (prefill_ms / 1000.0));

        ctx_->free_device(token_ids_device);

        // --- Decode loop ---
        // Collect generated tokens for penalty tracking
        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(max_new_tokens));

        // Sample first token using unified sampler
        if (gen_config.do_sample && gen_config.use_fixed_seed) {
            ops::set_sampling_seed(gen_config.sampling_seed);
        }
        int next_token = ops::sample_with_config(*ctx_, logits, config_.vocab_size,
                                                 gen_config, generated_token_ids);

        std::string output_text;
        bool streaming = (token_callback != nullptr);
        bool suppress_leading_think = no_think_requested;
        auto emit_stream_piece = [&](const std::string& piece) {
            if (no_think_requested && suppress_leading_think) {
                if (piece == "<think>" || piece == "</think>") {
                    return;
                }
                bool all_ws = true;
                for (unsigned char c : piece) {
                    if (!std::isspace(c)) {
                        all_ws = false;
                        break;
                    }
                }
                if (all_ws) {
                    return;
                }
                suppress_leading_think = false;
            }
            output_text += piece;
            token_callback(piece);
        };

        // Determine chunking strategy
        int effective_chunk_size = streaming ? std::max(1, gen_config.stream_chunk_size)
                                             : std::max(1, gen_config.decode_chunk_size);
        bool use_chunked_greedy = (!gen_config.do_sample && !gen_config.has_penalties()
                                   && effective_chunk_size > 1);
        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        ctx_->memcpy_h2d(current_token_device, &next_token, sizeof(int));

        int* generated_tokens_device = nullptr;
        if (use_chunked_greedy) {
            generated_tokens_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(max_new_tokens + 1) * sizeof(int)));
        }

        int generated_count = 0;
        std::vector<int> host_tokens(static_cast<size_t>(max_new_tokens));
        auto t_decode_start = std::chrono::high_resolution_clock::now();
        int last_token_seen = -1;
        int same_token_run = 0;
        auto check_loop_guard = [&](int token_id) -> bool {
            if (token_id == last_token_seen) {
                same_token_run++;
            } else {
                last_token_seen = token_id;
                same_token_run = 1;
            }
            if (same_token_run >= 24) {
                return true;
            }

            // Detect periodic n-gram loops (e.g., same phrase repeated).
            if ((generated_token_ids.size() % 8) == 0 && generated_token_ids.size() >= 48) {
                size_t n = generated_token_ids.size();
                for (size_t p = 4; p <= 24; p += 4) {
                    if (n < p * 4) continue;
                    bool periodic = true;
                    for (size_t i = 0; i < p; ++i) {
                        int a = generated_token_ids[n - 1 - i];
                        int b = generated_token_ids[n - 1 - p - i];
                        int c = generated_token_ids[n - 1 - 2 * p - i];
                        int d = generated_token_ids[n - 1 - 3 * p - i];
                        if (!(a == b && b == c && c == d)) {
                            periodic = false;
                            break;
                        }
                    }
                    if (periodic) {
                        return true;
                    }
                }
            }

            // Check low-diversity degeneration every 16 tokens
            if ((generated_token_ids.size() % 16) != 0 || generated_token_ids.size() < 64) {
                return false;
            }
            int unique_vals[16];
            int unique_count = 0;
            size_t start = generated_token_ids.size() - 64;
            for (size_t i = start; i < generated_token_ids.size(); ++i) {
                int v = generated_token_ids[i];
                bool seen = false;
                for (int u = 0; u < unique_count; ++u) {
                    if (unique_vals[u] == v) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    if (unique_count >= 16) {
                        return false;
                    }
                    unique_vals[unique_count++] = v;
                }
            }
            return unique_count <= 4;
        };

        if (use_chunked_greedy) {
            // Fast path: chunked greedy without penalties (fully async GPU)
            const int chunk_size = effective_chunk_size;
            bool eos_reached = false;
            ctx_->queue().memcpy(generated_tokens_device, current_token_device, sizeof(int));

            while (generated_count < max_new_tokens && !eos_reached) {
                int chunk_begin = generated_count;
                int chunk_end = std::min(max_new_tokens, chunk_begin + chunk_size);

                for (; generated_count < chunk_end; ++generated_count) {
                    int* current_ptr = generated_tokens_device + generated_count;
                    Tensor& logits_next = backend_->forward(*ctx_, current_ptr, 1);
                    ops::argmax(*ctx_, logits_next, config_.vocab_size,
                                generated_tokens_device + generated_count + 1);
                }

                int copied = generated_count - chunk_begin;
                auto copy_evt = ctx_->memcpy_d2h_async(host_tokens.data() + chunk_begin,
                                                       generated_tokens_device + chunk_begin,
                                                       static_cast<size_t>(copied) * sizeof(int));
                copy_evt.wait();

                for (int i = chunk_begin; i < generated_count; ++i) {
                    int token_id = host_tokens[i];
                    if (tokenizer_.is_eos(token_id)) {

                        generated_count = i;
                        eos_reached = true;
                        break;
                    }

                    generated_token_ids.push_back(token_id);
                    if (check_loop_guard(token_id)) {
                        AILA_LOG_WARN("[Generate] Loop guard triggered in greedy decode (token=%d, run=%d), stopping early",
                                      token_id, same_token_run);
                        eos_reached = true;
                        break;
                    }

                    if (streaming) {
                        std::string token_text = tokenizer_.decode(token_id);
                        emit_stream_piece(token_text);
                    }
                }
            }
        } else {
            // Unified path: penalties + sampling (goes through CPU each step)
            int current_token = next_token;
            while (generated_count < max_new_tokens) {
                if (tokenizer_.is_eos(current_token)) {
                    break;
                }

                generated_token_ids.push_back(current_token);
                if (check_loop_guard(current_token)) {
                    AILA_LOG_WARN("[Generate] Loop guard triggered in decode (token=%d, run=%d), stopping early",
                                  current_token, same_token_run);
                    break;
                }

                if (streaming) {
                    std::string token_text = tokenizer_.decode(current_token);
                    emit_stream_piece(token_text);
                }
                generated_count++;

                Tensor& logits_next = backend_->forward(*ctx_, current_token_device, 1);

                // Unified sampling with penalties
                next_token = ops::sample_with_config(*ctx_, logits_next, config_.vocab_size,
                                                      gen_config, generated_token_ids);
                ctx_->memcpy_h2d(next_token_device, &next_token, sizeof(int));
                std::swap(current_token_device, next_token_device);
                current_token = next_token;
            }

            ctx_->synchronize();
        }

        if (generated_tokens_device) {
            ctx_->free_device(generated_tokens_device);
        }
        ctx_->free_device(current_token_device);
        ctx_->free_device(next_token_device);

        auto t_decode_end = std::chrono::high_resolution_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

        AILA_LOG_INFO("[Generate] Decoded %d tokens in %.2f ms (%.1f tok/s)",
                      generated_count, decode_ms,
                      (generated_count > 0 ? generated_count / (decode_ms / 1000.0) : 0.0));

        if (!streaming && !generated_token_ids.empty()) {
            output_text = tokenizer_.decode(generated_token_ids);
        }
        if (no_think_requested) {
            strip_leading_think_artifacts(output_text);
        }

        // --- Update context state ---
        // cached_ids_ = the entire token sequence now in the KV cache
        // = full_ids (prompt) + generated_token_ids (response, no closing tokens yet)
        // We do NOT append <|im_end|>\n here; that happens at the start of the next turn.
        cached_ids_ = full_ids;
        cached_ids_.insert(cached_ids_.end(), generated_token_ids.begin(), generated_token_ids.end());

        // Add assistant response to history (for display and overflow rebuild)
        // Strip <think>...</think> from the text prior to adding to history so the model
        // isn't forced to hallucinate its own obsolete reasoning in future turns,
        // which prevents early EOS truncation behavior.
        std::string history_text = output_text;
        size_t start_think = history_text.find("<think>");
        if (start_think != std::string::npos) {
            size_t end_think = history_text.find("</think>");
            if (end_think != std::string::npos) {
                history_text.erase(start_think, end_think + 8 - start_think);
            } else {
                history_text.erase(start_think);
            }
            // Strip any remaining leading/trailing whitespace
            auto ltrim_h = [](std::string& s) {
                size_t i = 0;
                while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
                if (i > 0) s.erase(0, i);
            };
            ltrim_h(history_text);
        }
        history_.add_assistant(history_text);

        return output_text;
    }

    std::string generate_messages(const std::vector<Message>& messages,
                                  const GenerationConfig& gen_config = GenerationConfig(),
                                  std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        if (!backend_) {
            set_error(EngineErrorCode::RuntimeError, "Backend is not initialized");
            return "";
        }
        if (auto* q35_backend = dynamic_cast<Qwen35HybridTextBackend*>(backend_.get())) {
            q35_backend->clear_embedding_overrides();
            q35_backend->clear_mrope_positions();
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

        std::string tmpl_err;
        std::vector<Message> render_messages;
        render_messages.reserve(messages.size());
        std::vector<sycl::ext::oneapi::bfloat16> vision_embeddings_flat;
        size_t total_vision_tokens = 0;
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
                    if (!vision_encoder_->encode_image(p.uri, encoded, &vision_err)) {
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
                set_error(EngineErrorCode::TemplateError, "Unknown content part type");
                return "";
            }
            render_messages.push_back(std::move(out_msg));
        }

        std::vector<int> full_ids;
        if (!template_registry_.render(model_spec_.family, tokenizer_, render_messages,
                                       vision_backend_enabled_, true, full_ids, &tmpl_err)) {
            AILA_LOG_ERROR("[GenerateMessages] Template render failed: %s", tmpl_err.c_str());
            EngineErrorCode code = (tmpl_err.find("Vision content is not enabled") != std::string::npos)
                                       ? EngineErrorCode::VisionNotEnabled
                                       : EngineErrorCode::TemplateError;
            set_error(code, tmpl_err);
            return "";
        }

        if (total_vision_tokens > 0) {
            auto* q35_backend = dynamic_cast<Qwen35HybridTextBackend*>(backend_.get());
            if (!q35_backend) {
                set_error(EngineErrorCode::RuntimeError,
                          "Vision embedding override requires Qwen35 backend");
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

            q35_backend->set_embedding_overrides(image_positions, vision_embeddings_flat, config_.hidden_size);
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
            q35_backend->set_mrope_positions(*ctx_, pos_t, pos_h, pos_w, text_pos_delta);
            AILA_LOG_INFO("[Vision] Injecting %zu image embeddings into prompt", total_vision_tokens);
            AILA_LOG_INFO("[Vision] Applied multimodal text RoPE positions (delta=%d)", text_pos_delta);
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

        backend_->reset();
        cached_ids_.clear();

        auto t_start = std::chrono::high_resolution_clock::now();
        Tensor* logits_ptr = nullptr;
        bool tokenwise_prefill = (model_spec_.family == ModelFamily::Qwen35Hybrid) &&
                                 aila::env::read_flag("AILA_Q35_PREFILL_TOKENWISE", false);
        if (!tokenwise_prefill) {
            int* token_ids_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(full_ids.size()) * sizeof(int)));
            ctx_->memcpy_h2d(token_ids_device, full_ids.data(),
                             static_cast<size_t>(full_ids.size()) * sizeof(int));
            logits_ptr = &backend_->forward(*ctx_, token_ids_device, static_cast<int>(full_ids.size()));
            ctx_->free_device(token_ids_device);
        } else {
            AILA_LOG_INFO("[Qwen3.5] Tokenwise prefill enabled for debug");
            int* one_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            for (size_t i = 0; i < full_ids.size(); ++i) {
                int tok = full_ids[i];
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
        int next_token = ops::sample_with_config(*ctx_, logits, config_.vocab_size,
                                                 tuned_cfg, generated_token_ids);

        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        ctx_->memcpy_h2d(current_token_device, &next_token, sizeof(int));

        std::string output_text;
        bool streaming = (token_callback != nullptr);
        int same_token_run = 0;
        int last_token = -1;

        for (int i = 0; i < max_new_tokens; ++i) {
            int current_token = next_token;
            if (tokenizer_.is_eos(current_token)) break;

            if (current_token == last_token) same_token_run++;
            else {
                same_token_run = 1;
                last_token = current_token;
            }
            if (same_token_run >= 48) {
                AILA_LOG_WARN("[GenerateMessages] Loop guard triggered (token=%d run=%d)",
                              current_token, same_token_run);
                break;
            }

            generated_token_ids.push_back(current_token);
            if (debug_token_ids && i < 64) {
                std::string piece = tokenizer_.decode(current_token);
                AILA_LOG_INFO("[DebugToken] step=%d id=%d text='%s'",
                              i, current_token, piece.c_str());
            }
            if (streaming) {
                std::string token_text = tokenizer_.decode(current_token);
                output_text += token_text;
                token_callback(token_text);
            }

            Tensor& logits_next = backend_->forward(*ctx_, current_token_device, 1);
            next_token = ops::sample_with_config(*ctx_, logits_next, config_.vocab_size,
                                                 tuned_cfg, generated_token_ids);
            ctx_->memcpy_h2d(next_token_device, &next_token, sizeof(int));
            std::swap(current_token_device, next_token_device);
        }
        ctx_->synchronize();
        auto t_end = std::chrono::high_resolution_clock::now();
        ctx_->free_device(current_token_device);
        ctx_->free_device(next_token_device);

        if (!streaming && !generated_token_ids.empty()) {
            output_text = tokenizer_.decode(generated_token_ids);
        }

        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        AILA_LOG_INFO("[GenerateMessages] Prompt=%d Generated=%zu in %.2f ms",
                      total_prompt_len, generated_token_ids.size(), ms);

        cached_ids_ = full_ids;
        cached_ids_.insert(cached_ids_.end(), generated_token_ids.begin(), generated_token_ids.end());
        return output_text;
    }

    std::string generate_messages_json(const std::string& messages_json,
                                       const GenerationConfig& gen_config = GenerationConfig(),
                                       std::function<void(const std::string&)> token_callback = nullptr) {
        clear_error();
        std::vector<Message> messages;
        std::string parse_error;
        if (!parse_messages_json(messages_json, messages, &parse_error)) {
            AILA_LOG_ERROR("[GenerateMessages] Invalid messages JSON: %s", parse_error.c_str());
            set_error(EngineErrorCode::JsonParseError, parse_error);
            return "";
        }
        return generate_messages(messages, gen_config, token_callback);
    }

    // ============================================================
    // Raw prefill for benchmark (no decode, no history)
    // ============================================================
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

        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        ctx_->memcpy_h2d(current_token_device, &benchmark_seed_token_, sizeof(int));

        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(num_tokens));

        auto t0 = std::chrono::high_resolution_clock::now();
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

    Qwen3Config& config() { return config_; }
    const Qwen3Config& config() const { return config_; }
    const ModelSpec& model_spec() const { return model_spec_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    bool vision_enabled() const { return vision_backend_enabled_; }
    EngineErrorCode last_error_code() const { return last_error_code_; }
    const std::string& last_error_message() const { return last_error_message_; }

private:
    void clear_error() {
        last_error_code_ = EngineErrorCode::Ok;
        last_error_message_.clear();
    }

    void set_error(EngineErrorCode code, const std::string& message) {
        last_error_code_ = code;
        last_error_message_ = message;
    }

    bool parse_messages_json(const std::string& messages_json,
                             std::vector<Message>& out_messages,
                             std::string* error_message = nullptr) const {
        auto set_error = [&](const std::string& msg) {
            if (error_message) *error_message = msg;
        };

        out_messages.clear();
        try {
            simdjson::dom::parser parser;
            simdjson::dom::element root = parser.parse(messages_json);
            simdjson::dom::array arr;
            if (root.get_array().get(arr) != simdjson::SUCCESS) {
                set_error("messages root is not an array");
                return false;
            }

            for (auto item : arr) {
                simdjson::dom::object obj;
                if (item.get_object().get(obj) != simdjson::SUCCESS) {
                    set_error("message item is not an object");
                    return false;
                }

                Message msg;
                {
                    simdjson::dom::element role_elem;
                    if (obj.at_key("role").get(role_elem) != simdjson::SUCCESS) {
                        set_error("message.role missing");
                        return false;
                    }
                    std::string_view role_sv;
                    if (role_elem.get_string().get(role_sv) != simdjson::SUCCESS) {
                        set_error("message.role must be string");
                        return false;
                    }
                    msg.role = std::string(role_sv);
                }

                simdjson::dom::element content_elem;
                if (obj.at_key("content").get(content_elem) != simdjson::SUCCESS) {
                    set_error("message.content missing");
                    return false;
                }

                std::string_view content_str;
                if (content_elem.get_string().get(content_str) == simdjson::SUCCESS) {
                    msg.content.push_back(ContentPart{ContentType::Text, std::string(content_str), ""});
                } else {
                    simdjson::dom::array content_arr;
                    if (content_elem.get_array().get(content_arr) != simdjson::SUCCESS) {
                        set_error("message.content must be string or array");
                        return false;
                    }

                    for (auto part_elem : content_arr) {
                        simdjson::dom::object part_obj;
                        if (part_elem.get_object().get(part_obj) != simdjson::SUCCESS) {
                            set_error("content part must be object");
                            return false;
                        }

                        simdjson::dom::element type_elem;
                        if (part_obj.at_key("type").get(type_elem) != simdjson::SUCCESS) {
                            set_error("content part.type missing");
                            return false;
                        }
                        std::string_view type_sv;
                        if (type_elem.get_string().get(type_sv) != simdjson::SUCCESS) {
                            set_error("content part.type must be string");
                            return false;
                        }

                        std::string type(type_sv);
                        if (type == "text" || type == "input_text") {
                            simdjson::dom::element text_elem;
                            if (part_obj.at_key("text").get(text_elem) != simdjson::SUCCESS) {
                                set_error("text content missing text field");
                                return false;
                            }
                            std::string_view text_sv;
                            if (text_elem.get_string().get(text_sv) != simdjson::SUCCESS) {
                                set_error("text content.text must be string");
                                return false;
                            }
                            msg.content.push_back(ContentPart{ContentType::Text, std::string(text_sv), ""});
                        } else if (type == "image" || type == "image_url" || type == "input_image") {
                            std::string uri;
                            simdjson::dom::element image_url_elem;
                            if (part_obj.at_key("image_url").get(image_url_elem) == simdjson::SUCCESS) {
                                std::string_view image_url_sv;
                                if (image_url_elem.get_string().get(image_url_sv) == simdjson::SUCCESS) {
                                    uri = std::string(image_url_sv);
                                } else {
                                    simdjson::dom::object image_obj;
                                    if (image_url_elem.get_object().get(image_obj) == simdjson::SUCCESS) {
                                        simdjson::dom::element url_elem;
                                        if (image_obj.at_key("url").get(url_elem) == simdjson::SUCCESS) {
                                            std::string_view url_sv;
                                            if (url_elem.get_string().get(url_sv) == simdjson::SUCCESS) {
                                                uri = std::string(url_sv);
                                            }
                                        }
                                    }
                                }
                            }
                            if (uri.empty()) {
                                simdjson::dom::element image_elem;
                                if (part_obj.at_key("image").get(image_elem) == simdjson::SUCCESS) {
                                    std::string_view image_sv;
                                    if (image_elem.get_string().get(image_sv) == simdjson::SUCCESS) {
                                        uri = std::string(image_sv);
                                    }
                                }
                            }
                            if (uri.empty()) {
                                set_error("image content missing image/image_url/url field");
                                return false;
                            }
                            msg.content.push_back(ContentPart{ContentType::Image, "", uri});
                        } else if (type == "video" || type == "video_url" || type == "input_video") {
                            std::string uri;
                            simdjson::dom::element video_url_elem;
                            if (part_obj.at_key("video_url").get(video_url_elem) == simdjson::SUCCESS) {
                                std::string_view video_url_sv;
                                if (video_url_elem.get_string().get(video_url_sv) == simdjson::SUCCESS) {
                                    uri = std::string(video_url_sv);
                                } else {
                                    simdjson::dom::object video_obj;
                                    if (video_url_elem.get_object().get(video_obj) == simdjson::SUCCESS) {
                                        simdjson::dom::element url_elem;
                                        if (video_obj.at_key("url").get(url_elem) == simdjson::SUCCESS) {
                                            std::string_view url_sv;
                                            if (url_elem.get_string().get(url_sv) == simdjson::SUCCESS) {
                                                uri = std::string(url_sv);
                                            }
                                        }
                                    }
                                }
                            }
                            if (uri.empty()) {
                                simdjson::dom::element video_elem;
                                if (part_obj.at_key("video").get(video_elem) == simdjson::SUCCESS) {
                                    std::string_view video_sv;
                                    if (video_elem.get_string().get(video_sv) == simdjson::SUCCESS) {
                                        uri = std::string(video_sv);
                                    }
                                }
                            }
                            if (uri.empty()) {
                                set_error("video content missing video/video_url/url field");
                                return false;
                            }
                            msg.content.push_back(ContentPart{ContentType::Video, "", uri});
                        } else {
                            set_error("unknown content part type: " + type);
                            return false;
                        }
                    }
                }

                out_messages.push_back(std::move(msg));
            }
            return true;
        } catch (const std::exception& e) {
            set_error(std::string("JSON parse failed: ") + e.what());
            return false;
        }
    }

    std::string model_dir_;
    std::string system_prompt_ = "You are a helpful assistant.";
    Qwen3Config config_;
    ModelSpec model_spec_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<IModelBackend> backend_;
    std::unique_ptr<aila::vision::Qwen35VisionEncoder> vision_encoder_;
    aila::templating::TemplateRegistry template_registry_;
    Tokenizer tokenizer_;
    bool vision_backend_enabled_ = false;

    // Multi-turn conversation state
    ChatHistory history_;
    std::vector<Message> mm_history_;
    // Exact token IDs currently stored in the KV cache.
    // This is the ground truth for incremental prefill.
    std::vector<int> cached_ids_;

    // Benchmark decode seed (token after prefill argmax)
    int benchmark_seed_token_ = -1;
    bool benchmark_seed_ready_ = false;
    EngineErrorCode last_error_code_ = EngineErrorCode::Ok;
    std::string last_error_message_;
};
