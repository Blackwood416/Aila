#include "engine/Engine.hpp"

double InferenceEngine::benchmark_prefill(const std::vector<int>& token_ids) {
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

double InferenceEngine::benchmark_decode(int num_tokens, const GenerationConfig* decode_config) {
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
