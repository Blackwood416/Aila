#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/Qwen3.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/ModelConfig.hpp"
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
        weights_ = std::make_unique<ModelWeights>(LoadModelWeightsFromDir(model_dir, *ctx_));

        // 3.5 Load architecture config from config.json when available
        {
            std::string cfg_error;
            if (aila::modelcfg::load_qwen3_config_from_dir(model_dir, config_, &cfg_error)) {
                AILA_LOG_INFO("[Config] Loaded model config from config.json");
                AILA_LOG_INFO("[Config] hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d ffn=%d vocab=%d",
                              config_.hidden_size, config_.num_hidden_layers,
                              config_.num_attention_heads, config_.num_key_value_heads,
                              config_.head_dim, config_.intermediate_size,
                              config_.vocab_size);
            } else {
                AILA_LOG_WARN("[Config] %s (using built-in defaults)", cfg_error.c_str());
            }
        }
        if (max_seq_len > config_.max_position_embeddings) {
            AILA_LOG_WARN("[Config] max_seq_len=%d exceeds model max_position_embeddings=%d, clamping",
                          max_seq_len, config_.max_position_embeddings);
            max_seq_len = config_.max_position_embeddings;
        }

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
        benchmark_seed_ready_ = false;
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
        model_.truncate_kv_cache(reusable_prefix);
        cached_ids_.resize(reusable_prefix);

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

                Tensor& logits_next = model_.forward(*ctx_, current_token_device, 1);

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

    // ============================================================
    // Raw prefill for benchmark (no decode, no history)
    // ============================================================
    double benchmark_prefill(const std::vector<int>& token_ids) {
        model_.reset();
        cached_ids_.clear();
        benchmark_seed_ready_ = false;

        int* device_ids = static_cast<int*>(ctx_->alloc_device(token_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d(device_ids, token_ids.data(), token_ids.size() * sizeof(int));

        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor& logits = model_.forward(*ctx_, device_ids, static_cast<int>(token_ids.size()));
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
            Tensor& logits = model_.forward(*ctx_, current_token_device, 1);
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

    // Benchmark decode seed (token after prefill argmax)
    int benchmark_seed_token_ = -1;
    bool benchmark_seed_ready_ = false;
};
