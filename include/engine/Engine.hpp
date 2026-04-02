#pragma once

#include "../src/core/Context.hpp"
#include "../src/models/Qwen3.hpp"
#include "../src/utils/Tokenizer.hpp"
#include "../src/utils/SafeTensors.hpp"
#include "Types.hpp"
#include <string>
#include <functional>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>

// ============================================================
// Inference Engine: orchestrates loading, tokenization, inference
// ============================================================
class InferenceEngine {
public:
    InferenceEngine() = default;

    // Initialize: load model + tokenizer from model directory
    bool init(const std::string& model_dir, int max_seq_len = 4096) {
        model_dir_ = model_dir;

        std::cout << "========================================" << std::endl;
        std::cout << "  Aila Inference Engine (Qwen3-0.6B)" << std::endl;
        std::cout << "========================================" << std::endl;

        // 1. Create context
        std::cout << "\n[1/3] Initializing GPU context..." << std::endl;
        ctx_ = std::make_unique<Context>();

        // 2. Load tokenizer
        std::cout << "\n[2/3] Loading tokenizer..." << std::endl;
        if (!tokenizer_.load(model_dir)) {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return false;
        }

        // 3. Load model weights
        std::cout << "\n[3/3] Loading model weights..." << std::endl;
        std::string safetensors_path = model_dir + "/model.safetensors";
        weights_ = std::make_unique<ModelWeights>(LoadSafetensors(safetensors_path, *ctx_));

        // 4. Initialize model
        model_.load(*ctx_, *weights_, config_, max_seq_len);

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
            std::cout << "[Warmup] Completed in " << warmup_ms << " ms" << std::endl;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "  Engine ready! Type your message." << std::endl;
        std::cout << "========================================\n" << std::endl;

        return true;
    }

    // Generate response from prompt
    // token_callback is called for each generated token (streaming)
    std::string generate(const std::string& user_message,
                         const GenerationConfig& gen_config = GenerationConfig(),
                         std::function<void(const std::string&)> token_callback = nullptr) {

        // Reset KV cache for new conversation
        model_.reset();

        // Tokenize with ChatML template (thinking mode bypassed via injected tokens)
        auto input_ids = tokenizer_.apply_chat_template(
            "You are a helpful assistant.", user_message);

        std::cout << "[Generate] Prompt tokens: " << input_ids.size() << std::endl;

        // Upload token IDs to GPU (async, will be waited by first kernel)
        int* token_ids_device = static_cast<int*>(ctx_->alloc_device(input_ids.size() * sizeof(int)));
        ctx_->memcpy_h2d_async(token_ids_device, input_ids.data(), input_ids.size() * sizeof(int));

        // Prefill: process all prompt tokens at once
        auto t_start = std::chrono::high_resolution_clock::now();
        Tensor& logits = model_.forward(*ctx_, token_ids_device, static_cast<int>(input_ids.size()));
        ctx_->synchronize();
        auto t_prefill = std::chrono::high_resolution_clock::now();

        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
        std::cout << "[Generate] Prefill: " << prefill_ms << " ms ("
                  << input_ids.size() / (prefill_ms / 1000.0) << " tok/s)" << std::endl;

        ctx_->free_device(token_ids_device);

        // Allocate buffers on GPU for decode loop
        int* current_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));
        int* next_token_device = static_cast<int*>(ctx_->alloc_device(sizeof(int)));

        // Sample first token
        int next_token = 0;
        if (gen_config.do_sample) {
            next_token = ops::topk_sample(*ctx_, logits, config_.vocab_size,
                                          gen_config.temperature, gen_config.top_k);
            ctx_->memcpy_h2d(current_token_device, &next_token, sizeof(int));
        } else {
            ops::argmax(*ctx_, logits, config_.vocab_size, current_token_device);
        }

        // Decode loop (Asynchronous Pipeline)
        std::string output_text;
        std::vector<int> host_tokens(static_cast<size_t>(gen_config.max_new_tokens));
        std::vector<int> generated_token_ids;
        generated_token_ids.reserve(static_cast<size_t>(gen_config.max_new_tokens));
        bool streaming = (token_callback != nullptr);
        int effective_chunk_size = streaming ? std::max(1, gen_config.stream_chunk_size)
                                             : std::max(1, gen_config.decode_chunk_size);
        bool use_chunked_greedy = (!gen_config.do_sample && effective_chunk_size > 1);
        int* generated_tokens_device = nullptr;
        if (use_chunked_greedy) {
            generated_tokens_device = static_cast<int*>(
                ctx_->alloc_device(static_cast<size_t>(gen_config.max_new_tokens + 1) * sizeof(int)));
        }

        int generated_count = 0;
        auto t_decode_start = std::chrono::high_resolution_clock::now();

        if (use_chunked_greedy) {
            const int chunk_size = effective_chunk_size;
            bool eos_reached = false;
            ctx_->queue().memcpy(generated_tokens_device, current_token_device, sizeof(int));

            while (generated_count < gen_config.max_new_tokens && !eos_reached) {
                int chunk_begin = generated_count;
                int chunk_end = std::min(gen_config.max_new_tokens, chunk_begin + chunk_size);

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
                        generated_count = i; // EOS token itself is not counted
                        eos_reached = true;
                        break;
                    }

                    if (streaming) {
                        std::string token_text = tokenizer_.decode(token_id);
                        output_text += token_text;
                        token_callback(token_text);
                    } else {
                        generated_token_ids.push_back(token_id);
                    }
                }
            }
        } else {
            while (generated_count < gen_config.max_new_tokens) {
                Tensor& logits_next = model_.forward(*ctx_, current_token_device, 1);

                ctx_->memcpy_d2h(&next_token, current_token_device, sizeof(int));
                if (tokenizer_.is_eos(next_token)) {
                    break;
                }

                if (streaming) {
                    std::string token_text = tokenizer_.decode(next_token);
                    output_text += token_text;
                    token_callback(token_text);
                } else {
                    generated_token_ids.push_back(next_token);
                }
                generated_count++;

                if (gen_config.do_sample) {
                    ctx_->synchronize();
                    next_token = ops::topk_sample(*ctx_, logits_next, config_.vocab_size,
                                                  gen_config.temperature, gen_config.top_k);
                    ctx_->memcpy_h2d(next_token_device, &next_token, sizeof(int));
                } else {
                    ops::argmax(*ctx_, logits_next, config_.vocab_size, next_token_device);
                }
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

        std::cout << "\n[Generate] Decoded " << generated_count << " tokens in "
                  << decode_ms << " ms ("
                  << (generated_count > 0 ? generated_count / (decode_ms / 1000.0) : 0)
                  << " tok/s)" << std::endl;

        if (!streaming && !generated_token_ids.empty()) {
            output_text = tokenizer_.decode(generated_token_ids);
        }

        return output_text;
    }

private:
    std::string model_dir_;
    Qwen3Config config_;
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<ModelWeights> weights_;
    Qwen3Model model_;
    Tokenizer tokenizer_;
};
