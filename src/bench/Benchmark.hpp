#pragma once

#include <string>
#include "engine/Types.hpp"

class InferenceEngine;

struct BenchmarkConfig {
    int prompt_length   = 512;   // pp: number of tokens to prefill
    int gen_length      = 128;   // tg: number of tokens to decode
    int warmup_iters    = 1;     // warmup iterations
    int bench_iters     = 5;     // measurement iterations
    bool decode_do_sample = false; // decode benchmark mode
    GenerationConfig decode_gen_config{};
};

struct BenchmarkResult {
    double prefill_tok_per_sec = 0.0;
    double decode_tok_per_sec  = 0.0;
    double prefill_ms_avg      = 0.0;
    double decode_ms_avg       = 0.0;
    double prefill_ms_stddev   = 0.0;
    double decode_ms_stddev    = 0.0;
    double ttft_ms_avg         = 0.0;
    int    prompt_tokens       = 0;
    int    gen_tokens          = 0;
    bool   decode_do_sample    = false;
};

// Run benchmark: prefill + decode measurements
BenchmarkResult run_benchmark(InferenceEngine& engine, const BenchmarkConfig& cfg);

// Print benchmark results in formatted table
void print_benchmark_results(const BenchmarkResult& result);
