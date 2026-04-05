#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/Qwen3.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "../src/profile/Profiling.hpp"
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
        std::string safetensors_path = model_dir + "/model.safetensors";
        weights_ = std::make_unique<ModelWeights>(LoadSafetensors(safetensors_path, *ctx_));

        // 4. Initialize model
        model_.load(*ctx_, *weights_, config_, max_seq_len);
        auto to_mb = [](size_t bytes) -> double {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };
        AILA_LOG_INFO("[Memory] After model load: current=%.2f MB, peak=%.2f MB",
                      to_mb(ctx_->current_allocated_bytes()),
                      to_mb(ctx_->peak_allocated_bytes()));

        // 5. Warmup to amortize first-run JIT/primitive costs
        {
            model_.reset();
            auto warmup_ids = tokenizer_.apply_chat_template(
                "You are a helpful assistant.", "Warmup.");
            int* warmup_token_ids = static_cast<int*>(
                ctx_->alloc_device(warmup_ids.size() * sizeof(int)));
            ctx_->memcpy_h2d_async(warmup_token_ids, warmup_ids.data(),
                                   warmup_ids.size() * sizeof(int));

            auto t_warmup_start = std::chrono::high_resolution_clock::now();
            Tensor& warmup_logits = model_.forward(*ctx_, warmup_token_ids, static_cast<int>(warmup_ids.size()));
            int* warmup_argmax = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
            ops::argmax(*ctx_, warmup_logits, config_.vocab_size, warmup_argmax);
            ctx_->synchronize();
            auto t_warmup_end = std::chrono::high_resolution_clock::now();

            ctx_->free_device(warmup_argmax);
            ctx_->free_device(warmup_token_ids);
            model_.reset();

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
        cached_ids_.clear();
        model_.reset();
        AILA_LOG_INFO("[Context] Conversation reset");
    }

    int context_length() const { return static_cast<int>(cached_ids_.size()); }
    int max_context_length() const { return model_.max_seq_len(); }
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
        // We construct it incrementally from cached_ids_ to avoid
        // decode→re-encode mismatches.
        //
        // If cached_ids_ is empty (first turn), build from scratch.
        // Otherwise, append:  <|im_end|>\n  (close prev assistant)
        //                     <|im_start|>user\n{msg}<|im_end|>\n
        //                     <|im_start|>assistant\n

        std::vector<int> full_ids;
        int reusable_prefix = 0;

        if (cached_ids_.empty()) {
            // First turn: encode from scratch using chat template
            full_ids = tokenizer_.apply_chat_template(system_prompt_, history_);
            reusable_prefix = 0;
        } else {
            // Subsequent turns: build incrementally
            // The KV cache contains cached_ids_ which ends with the generated tokens
            // of the previous assistant response. We need to close that turn and
            // open the new one.

            std::vector<int> new_suffix;

            // Close previous assistant turn: <|im_end|>\n
            new_suffix.push_back(config_.im_end_id);
            auto nl_tokens = tokenizer_.encode("\n");
            new_suffix.insert(new_suffix.end(), nl_tokens.begin(), nl_tokens.end());

            // New user turn: <|im_start|>user\n{message}<|im_end|>\n
            new_suffix.push_back(config_.im_start_id);
            auto user_turn = tokenizer_.encode("user\n" + user_message);
            new_suffix.insert(new_suffix.end(), user_turn.begin(), user_turn.end());
            new_suffix.push_back(config_.im_end_id);
            new_suffix.insert(new_suffix.end(), nl_tokens.begin(), nl_tokens.end());

            // Open assistant turn: <|im_start|>assistant\n
            new_suffix.push_back(config_.im_start_id);
            auto asst_start = tokenizer_.encode("assistant\n");
            new_suffix.insert(new_suffix.end(), asst_start.begin(), asst_start.end());

            // Handle /no_think suffix injection
            if (no_think_requested) {
                auto end_think_it_pair = tokenizer_.special_token_id("</think>");
                if (end_think_it_pair >= 0) {
                    new_suffix.push_back(end_think_it_pair);
                    new_suffix.insert(new_suffix.end(), nl_tokens.begin(), nl_tokens.end());
                    auto direct = tokenizer_.encode("Please answer directly and briefly.\n");
                    new_suffix.insert(new_suffix.end(), direct.begin(), direct.end());
                }
            }

            // full_ids = cached_ids_ + new_suffix
            full_ids.reserve(cached_ids_.size() + new_suffix.size());
            full_ids.insert(full_ids.end(), cached_ids_.begin(), cached_ids_.end());
            full_ids.insert(full_ids.end(), new_suffix.begin(), new_suffix.end());
            reusable_prefix = static_cast<int>(cached_ids_.size());
        }

        // --- Context overflow: if too long, clear and rebuild from scratch ---
        int max_ctx = model_.max_seq_len();
        if (static_cast<int>(full_ids.size()) > max_ctx - 64) {
            // Truncate history until it fits
            while (static_cast<int>(full_ids.size()) > max_ctx - 64 && history_.size() > 1) {
                history_.truncate_oldest(static_cast<int>(history_.size()) - 2);
                full_ids = tokenizer_.apply_chat_template(system_prompt_, history_);
                AILA_LOG_WARN("[Context] History truncated to fit context window (%zu messages remaining)",
                              history_.size());
            }
            // Must do full prefill since we rebuilt from scratch
            model_.reset();
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
        Tensor& logits = model_.forward(*ctx_, token_ids_device, new_tokens_to_prefill);
        ctx_->synchronize();
        auto t_prefill = std::chrono::high_resolution_clock::now();

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        AILA_LOG_INFO("[Generate] Prefill: %d tokens in %.2f ms (%.1f tok/s)",
                      new_tokens_to_prefill, prefill_ms,
                      new_tokens_to_prefill / (prefill_ms / 1000.0));

        ctx_->free_device(token_ids_device);

        // --- Decode loop ---
        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));

        // Collect generated tokens for penalty tracking
        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(max_new_tokens));

        // Sample first token using unified sampler
        int next_token = ops::sample_with_config(*ctx_, logits, config_.vocab_size,
                                                  gen_config, generated_token_ids);
        ctx_->memcpy_h2d(current_token_device, &next_token, sizeof(int));

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

        int* generated_tokens_device = nullptr;
        if (use_chunked_greedy) {
            generated_tokens_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(max_new_tokens + 1) * sizeof(int)));
        }

        int generated_count = 0;
        std::vector<int> host_tokens(static_cast<size_t>(max_new_tokens));
        auto t_decode_start = std::chrono::high_resolution_clock::now();

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
                    Tensor& logits_next = model_.forward(*ctx_, current_ptr, 1);
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

                    if (streaming) {
                        std::string token_text = tokenizer_.decode(token_id);
                        emit_stream_piece(token_text);
                    }
                }
            }
        } else {
            // Unified path: penalties + sampling (goes through CPU each step)
            while (generated_count < max_new_tokens) {
                Tensor& logits_next = model_.forward(*ctx_, current_token_device, 1);

                ctx_->memcpy_d2h(&next_token, current_token_device, sizeof(int));
                if (tokenizer_.is_eos(next_token)) {
                    break;
                }

                generated_token_ids.push_back(next_token);

                if (streaming) {
                    std::string token_text = tokenizer_.decode(next_token);
                    emit_stream_piece(token_text);
                }
                generated_count++;

                // Unified sampling with penalties
                ctx_->synchronize();
                next_token = ops::sample_with_config(*ctx_, logits_next, config_.vocab_size,
                                                      gen_config, generated_token_ids);
                ctx_->memcpy_h2d(next_token_device, &next_token, sizeof(int));
                std::swap(current_token_device, next_token_device);
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
        history_.add_assistant(output_text);

        return output_text;
    }

    // ============================================================
    // Raw prefill for benchmark (no decode, no history)
    // ============================================================
    double benchmark_prefill(const std::vector<int>& token_ids) {
        model_.reset();
        cached_ids_.clear();

        int* device_ids = static_cast<int*>(ctx_->alloc_device(token_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d(device_ids, token_ids.data(), token_ids.size() * sizeof(int));

        auto t0 = std::chrono::high_resolution_clock::now();
        model_.forward(*ctx_, device_ids, static_cast<int>(token_ids.size()));
        ctx_->synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        ctx_->free_device(device_ids);
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // Raw decode N tokens for benchmark (after prefill)
    double benchmark_decode(int num_tokens) {
        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));

        // Start with argmax of last logits
        Tensor& init_logits = model_.forward(*ctx_, current_token_device, 1);
        ops::argmax(*ctx_, init_logits, config_.vocab_size, current_token_device);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_tokens; i++) {
            Tensor& logits = model_.forward(*ctx_, current_token_device, 1);
            ops::argmax(*ctx_, logits, config_.vocab_size, next_token_device);
            std::swap(current_token_device, next_token_device);
        }
        ctx_->synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        ctx_->free_device(current_token_device);
        ctx_->free_device(next_token_device);
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    Qwen3Config& config() { return config_; }
    const Qwen3Config& config() const { return config_; }
    Tokenizer& tokenizer() { return tokenizer_; }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    Qwen3Model& model() { return model_; }

private:
    std::string model_dir_;
    std::string system_prompt_ = "You are a helpful assistant.";
    Qwen3Config config_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    Qwen3Model model_;
    Tokenizer tokenizer_;

    // Multi-turn conversation state
    ChatHistory history_;
    // Exact token IDs currently stored in the KV cache.
    // This is the ground truth for incremental prefill.
    std::vector<int> cached_ids_;
};
