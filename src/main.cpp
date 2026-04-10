#include "engine/Engine.hpp"
#include "cli/CLI.hpp"
#include "bench/Benchmark.hpp"
#include "profile/Device.hpp"
#include "profile/Profiling.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char** argv) {
    // Parse command line arguments
    CLIOptions opts;
    if (!parse_cli_args(argc, argv, opts)) {
        return 1;
    }

    if (opts.show_help) {
        print_help();
        return 0;
    }

    if (opts.show_version) {
        print_version();
        return 0;
    }

    // Validate model directory
    if (opts.model_dir.empty()) {
        std::cerr << "Error: Model directory is required." << std::endl;
        std::cerr << "Use -m <path> or set AILA_MODEL_DIR environment variable." << std::endl;
        std::cerr << "Use --help for usage information." << std::endl;
        return 1;
    }

    // Check GPU device
    CheckDevice();
    AILA_LOG_INFO("[Config] max_seq_len=%d", opts.max_seq_len);

    // Initialize engine
    InferenceEngine engine;
    if (!engine.init(opts.model_dir, opts.max_seq_len)) {
        AILA_LOG_ERROR("Failed to initialize inference engine");
        return 1;
    }

    // Benchmark mode
    if (opts.bench_mode) {
        BenchmarkConfig bench_cfg;
        bench_cfg.prompt_length = opts.bench_pp;
        bench_cfg.gen_length    = opts.bench_tg;
        bench_cfg.bench_iters   = opts.bench_iters;
        bench_cfg.warmup_iters  = opts.bench_warmup;
        bench_cfg.decode_do_sample = opts.bench_sample;
        bench_cfg.decode_gen_config.max_new_tokens = opts.bench_tg;
        bench_cfg.decode_gen_config.temperature = opts.temperature;
        bench_cfg.decode_gen_config.top_k = opts.top_k;
        bench_cfg.decode_gen_config.top_p = opts.top_p;
        bench_cfg.decode_gen_config.do_sample = opts.bench_sample;
        bench_cfg.decode_gen_config.repetition_penalty = opts.repetition_penalty;
        bench_cfg.decode_gen_config.presence_penalty = opts.presence_penalty;
        bench_cfg.decode_gen_config.frequency_penalty = opts.frequency_penalty;
        bench_cfg.decode_gen_config.sampling_seed = opts.sampling_seed;
        bench_cfg.decode_gen_config.use_fixed_seed = opts.use_fixed_seed || opts.bench_sample;

        auto result = run_benchmark(engine, bench_cfg);
        print_benchmark_results(result);
        return 0;
    }

    // Build generation config from CLI options
    GenerationConfig gen_config;
    gen_config.max_new_tokens    = opts.max_new_tokens;
    gen_config.temperature       = opts.temperature;
    gen_config.top_k             = opts.top_k;
    gen_config.top_p             = opts.top_p;
    gen_config.do_sample         = opts.do_sample;
    gen_config.sampling_seed     = opts.sampling_seed;
    gen_config.use_fixed_seed    = opts.use_fixed_seed;
    gen_config.decode_chunk_size = opts.decode_chunk_size;
    gen_config.stream_chunk_size = opts.stream_chunk_size;
    gen_config.repetition_penalty = opts.repetition_penalty;
    gen_config.presence_penalty   = opts.presence_penalty;
    gen_config.frequency_penalty  = opts.frequency_penalty;

    // Single-shot messages JSON mode
    if (!opts.messages_json_path.empty()) {
        std::ifstream in(opts.messages_json_path, std::ios::binary);
        if (!in.is_open()) {
            AILA_LOG_ERROR("Failed to open messages JSON file: %s", opts.messages_json_path.c_str());
            return 1;
        }
        std::string messages_json((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());

        if (opts.stream_output) {
            std::cout << "\nAila: ";
            engine.generate_messages_json(messages_json, gen_config,
                [](const std::string& token_text) {
                    std::cout << token_text << std::flush;
                });
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("messages-json generation failed: %s",
                               engine.last_error_message().c_str());
                return 2;
            }
            std::cout << std::endl;
        } else {
            std::string out = engine.generate_messages_json(messages_json, gen_config, nullptr);
            if (engine.last_error_code() != EngineErrorCode::Ok) {
                AILA_LOG_ERROR("messages-json generation failed: %s",
                               engine.last_error_message().c_str());
                return 2;
            }
            std::cout << out << std::endl;
        }
        return 0;
    }

    // Run interactive loop
    return run_interactive(engine, gen_config, opts.stream_output);
}
