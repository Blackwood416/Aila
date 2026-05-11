#include "Benchmark.hpp"
#include "engine/Engine.hpp"
#include "profile/Profiling.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

// ============================================================
// Generate synthetic prompt tokens
// ============================================================
static std::vector<int> generate_synthetic_prompt(InferenceEngine& engine, int target_len) {
    // Use a simple repeating pattern of common tokens.
    // Encode a base sentence and repeat to desired length.
    auto& tokenizer = engine.tokenizer();
    std::vector<int> base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. "
        "In a world where technology advances rapidly, "
        "artificial intelligence continues to transform "
        "how we live, work, and communicate with each other. ");

    std::vector<int> prompt;
    prompt.reserve(target_len);
    while (static_cast<int>(prompt.size()) < target_len) {
        int remaining = target_len - static_cast<int>(prompt.size());
        int copy_len = std::min(remaining, static_cast<int>(base.size()));
        prompt.insert(prompt.end(), base.begin(), base.begin() + copy_len);
    }
    return prompt;
}

// ============================================================
// Statistics helpers
// ============================================================
static double compute_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double compute_stddev(const std::vector<double>& v, double mean) {
    if (v.size() < 2) return 0.0;
    double sq_sum = 0.0;
    for (double x : v) {
        sq_sum += (x - mean) * (x - mean);
    }
    return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
}

// ============================================================
// Run Benchmark
// ============================================================

BenchmarkResult run_benchmark(InferenceEngine& engine, const BenchmarkConfig& cfg) {
    AILA_LOG_INFO("========================================");
    AILA_LOG_INFO("  Aila Benchmark");
    AILA_LOG_INFO("========================================");
    AILA_LOG_INFO("  pp (prompt tokens):  %d", cfg.prompt_length);
    AILA_LOG_INFO("  tg (gen tokens):     %d", cfg.gen_length);
    AILA_LOG_INFO("  decode mode:         %s", cfg.decode_do_sample ? "sampling" : "greedy");
    if (cfg.decode_do_sample) {
        AILA_LOG_INFO("  sampling params:     temp=%.3f top_k=%d top_p=%.3f seed=%llu fixed_seed=%s",
                      cfg.decode_gen_config.temperature,
                      cfg.decode_gen_config.top_k,
                      cfg.decode_gen_config.top_p,
                      static_cast<unsigned long long>(cfg.decode_gen_config.sampling_seed),
                      cfg.decode_gen_config.use_fixed_seed ? "true" : "false");
    }
    AILA_LOG_INFO("  warmup iterations:   %d", cfg.warmup_iters);
    AILA_LOG_INFO("  bench iterations:    %d", cfg.bench_iters);
    AILA_LOG_INFO("========================================");

    // Generate synthetic prompt
    auto prompt = generate_synthetic_prompt(engine, cfg.prompt_length);
    int actual_pp = static_cast<int>(prompt.size());

    // --- Warmup ---
    AILA_LOG_INFO("[Bench] Running %d warmup iteration(s)...", cfg.warmup_iters);
    for (int i = 0; i < cfg.warmup_iters; i++) {
        engine.benchmark_prefill(prompt);
        engine.benchmark_decode(std::min(cfg.gen_length, 16),
                                cfg.decode_do_sample ? &cfg.decode_gen_config : nullptr);
    }
    AILA_LOG_INFO("[Bench] Warmup complete");

    // --- Prefill benchmark ---
    AILA_LOG_INFO("[Bench] Measuring prefill (%d iterations)...", cfg.bench_iters);
    std::vector<double> pp_times;
    pp_times.reserve(cfg.bench_iters);

    for (int i = 0; i < cfg.bench_iters; i++) {
        double ms = engine.benchmark_prefill(prompt);
        pp_times.push_back(ms);
        AILA_LOG_INFO("  pp iter %d: %.2f ms (%.1f tok/s)",
                      i + 1, ms, actual_pp / (ms / 1000.0));
    }

    // --- Decode benchmark ---
    AILA_LOG_INFO("[Bench] Measuring decode (%d iterations)...", cfg.bench_iters);
    std::vector<double> tg_times;
    tg_times.reserve(cfg.bench_iters);

    for (int i = 0; i < cfg.bench_iters; i++) {
        // Prefill first (not measured), then decode
        engine.benchmark_prefill(prompt);
        double ms = engine.benchmark_decode(cfg.gen_length,
                                            cfg.decode_do_sample ? &cfg.decode_gen_config : nullptr);
        tg_times.push_back(ms);
        AILA_LOG_INFO("  tg iter %d: %.2f ms (%.1f tok/s)",
                      i + 1, ms, cfg.gen_length / (ms / 1000.0));
    }

    // --- Compute statistics ---
    BenchmarkResult result;
    result.prompt_tokens = actual_pp;
    result.gen_tokens = cfg.gen_length;
    result.decode_do_sample = cfg.decode_do_sample;

    result.prefill_ms_avg = compute_mean(pp_times);
    result.prefill_ms_stddev = compute_stddev(pp_times, result.prefill_ms_avg);
    result.prefill_tok_per_sec = actual_pp / (result.prefill_ms_avg / 1000.0);

    result.decode_ms_avg = compute_mean(tg_times);
    result.decode_ms_stddev = compute_stddev(tg_times, result.decode_ms_avg);
    result.decode_tok_per_sec = cfg.gen_length / (result.decode_ms_avg / 1000.0);

    result.ttft_ms_avg = result.prefill_ms_avg + result.decode_ms_avg / cfg.gen_length;

    return result;
}

// ============================================================
// Print Benchmark Results
// ============================================================

void print_benchmark_results(const BenchmarkResult& r) {
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "  Benchmark Results" << std::endl;
    std::cout << "  Decode mode: " << (r.decode_do_sample ? "sampling" : "greedy") << std::endl;
    std::cout << "========================================" << std::endl;

    // Header
    std::cout << std::left << std::setw(12) << "  test"
              << std::right
              << std::setw(10) << "tokens"
              << std::setw(14) << "tok/s"
              << std::setw(14) << "ms/tok"
              << std::setw(14) << "ms/run"
              << std::setw(14) << "stddev"
              << std::endl;
    std::cout << "  " << std::string(74, '-') << std::endl;

    // Prefill row
    double pp_ms_per_tok = (r.prompt_tokens > 0) ? r.prefill_ms_avg / r.prompt_tokens : 0.0;
    std::cout << std::left << std::setw(12) << "  pp"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.prompt_tokens
              << std::setw(14) << r.prefill_tok_per_sec
              << std::setw(14) << pp_ms_per_tok
              << std::setw(14) << r.prefill_ms_avg
              << std::setw(14) << r.prefill_ms_stddev
              << std::endl;

    // Decode row
    double tg_ms_per_tok = (r.gen_tokens > 0) ? r.decode_ms_avg / r.gen_tokens : 0.0;
    std::cout << std::left << std::setw(12) << "  tg"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << r.gen_tokens
              << std::setw(14) << r.decode_tok_per_sec
              << std::setw(14) << tg_ms_per_tok
              << std::setw(14) << r.decode_ms_avg
              << std::setw(14) << r.decode_ms_stddev
              << std::endl;

    // TTFT row (prefill + first decode step)
    std::cout << std::left << std::setw(12) << "  ttft"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(10) << (r.prompt_tokens + 1)
              << std::setw(14) << "-"
              << std::setw(14) << "-"
              << std::setw(14) << r.ttft_ms_avg
              << std::setw(14) << "-"
              << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
}
